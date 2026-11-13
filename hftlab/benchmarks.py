"""Paired, correctness-gated native timings; startup stays outside kernel speedups."""
from __future__ import annotations

import math
import statistics
import threading
from dataclasses import replace
from pathlib import Path

from hftlab.configuration import ENGINES, SimulationConfig
from hftlab.engines import capabilities, engine_available
from hftlab.orchestration import JobCancelled, JobManager
from hftlab.utilities import atomic_json, utc_now


def compare_results(reference: dict, candidate: dict) -> dict:
    differences = []
    for name in ("events_processed", "trades", "orders", "checksum"):
        if name in reference and reference.get(name) != candidate.get(name):
            differences.append(name)
    for name in ("pnl", "equity", "fees", "drawdown"):
        if name in reference and not math.isclose(reference[name], candidate.get(name, math.inf), rel_tol=1e-9, abs_tol=1e-6):
            differences.append(name)
    return {"passed": not differences, "differences": differences, "oracle": "sequential"}


def summarize(samples: list[float]) -> dict:
    if not samples or any(value <= 0 or not math.isfinite(value) for value in samples):
        raise ValueError("Timing samples must be positive and finite")
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    mean = statistics.mean(samples)
    # Student t 95% two-tailed critical values, conservative small-sample intervals.
    t95 = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]
    critical = t95[min(len(samples)-1, 10)] if len(samples) <= 11 else 1.96
    margin = critical * deviation / math.sqrt(len(samples))
    return {"samples_seconds": samples, "mean_seconds": mean, "median_seconds": statistics.median(samples),
            "stddev_seconds": deviation, "ci95_seconds": [max(0.0, mean-margin), mean+margin] if len(samples) > 1 else None,
            "repeats": len(samples)}


def benchmark(config: SimulationConfig, repeats: int = 3, engines: list[str] | None = None,
              output: Path = Path("reports/benchmark.json"), warmups: int = 1,
              cancel_event: threading.Event | None = None) -> dict:
    if type(repeats) is not int or type(warmups) is not int or not 1 <= repeats <= 100 or not 0 <= warmups <= 10:
        raise ValueError("repeats must be 1..100 and warmups 0..10")
    config.validate()
    requested = list(dict.fromkeys(["sequential"] + (engines or ["sequential", "threaded", "openmp"])))
    if any(engine not in ENGINES for engine in requested):
        raise ValueError("Unknown engine in benchmark request")
    capability = capabilities()
    if capability.get("error"):
        raise RuntimeError(f"Native capability probe failed: {capability['error']}")
    if not engine_available(capability, "sequential"):
        raise RuntimeError("Sequential oracle is unavailable")
    report = {"schema_version": 1, "timestamp": utc_now(), "config": config.to_dict(),
              "methodology": {"scope": "native wall_seconds: feature/inference and matching execution, excludes CSV parsing and partition allocation; load_seconds and full process/orchestration durations are recorded separately",
                              "warmups": warmups, "repeats": repeats, "live_visualization": False, "native_progress": False,
                              "correctness": "compare each repeat to the sequential reference before reporting speedup",
                              "ci": "95% Student t for n<=11; normal approximation for larger samples; n=1 has no interval"},
              "capabilities": capability, "engines": []}
    reference = None
    baseline = None
    for engine in requested:
        if not engine_available(capability, engine):
            report["engines"].append({"engine": engine, "status": "UNAVAILABLE"})
            continue
        candidate_config = replace(config, engine=engine,
                                   cpu_mode="single-single" if engine == "sequential" else "multi-multi")
        try:
            for _ in range(warmups):
                JobManager().run(candidate_config, quiet=True, cancel_event=cancel_event)
            runs = [JobManager().run(candidate_config, quiet=True, cancel_event=cancel_event) for _ in range(repeats)]
            if reference is None:
                reference = runs[0]
            checks = [compare_results(reference, run) for run in runs]
            if not all(check["passed"] for check in checks):
                report["engines"].append({"engine": engine, "status": "INCORRECT", "checks": checks, "runs": runs})
                if engine == "sequential":
                    break
                continue
            timings = summarize([run["wall_seconds"] for run in runs])
            if engine == "sequential":
                baseline = timings["median_seconds"]
            workers = runs[0].get("workers", runs[0].get("profiling", {}).get("workers"))
            if engine == "mpi" and isinstance(workers, int):
                workers *= runs[0].get("profiling", {}).get("ranks", 1)
            speedup = baseline / timings["median_seconds"] if baseline else None
            row = {"engine": engine, "status": "COMPLETED", "timing": timings, "correctness": checks,
                   "speedup": speedup, "workers": workers,
                   "efficiency": speedup / workers if engine not in ("cuda", "hybrid") and speedup is not None and isinstance(workers, int) and workers > 0 else None,
                   "throughput": statistics.median(run["throughput"] for run in runs), "runs": runs}
            if engine not in ("cuda", "hybrid") and isinstance(workers, int) and workers > 1 and speedup:
                row["karp_flatt_fraction"] = (1 / speedup - 1 / workers) / (1 - 1 / workers)
            report["engines"].append(row)
        except JobCancelled as error:
            report["engines"].append({"engine": engine, "status": "CANCELLED", "error": str(error)})
            report["status"] = "CANCELLED"
            atomic_json(output, report)
            raise
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
            report["engines"].append({"engine": engine, "status": "FAILED", "error": str(error)})
            if engine == "sequential":
                break
    report["status"] = "FAILED" if any(row["status"] in ("FAILED", "INCORRECT") for row in report["engines"]) else "COMPLETED"
    atomic_json(output, report)
    report["report_path"] = str(Path(output).resolve())
    return report


def scaling(config: SimulationConfig, workers: list[int], *, weak: bool = False, repeats: int = 3,
            events_per_worker: int = 10000, output: Path = Path("reports/scaling.json")) -> dict:
    if not workers or any(type(count) is not int or not 1 <= count <= 1024 for count in workers):
        raise ValueError("Scaling worker counts must be between 1 and 1024")
    from hftlab.datasets import generate_dataset
    rows = []
    for count in workers:
        current = replace(config, threads=count)
        if weak:
            dataset = Path(output).resolve().parent / "scaling-data" / f"weak-{count}.csv"
            generate_dataset(dataset, events=events_per_worker*count, instruments=max(2, count), seed=42, scenario="normal")
            current = replace(current, dataset=dataset)
        result = benchmark(current, repeats=repeats, engines=[config.engine],
                           output=Path(output).with_name(f"{Path(output).stem}-{count}.json"))
        rows.append({"workers": count, "benchmark": result})
    timings = {}
    for row in rows:
        candidate = next((item for item in row["benchmark"]["engines"]
                          if item["engine"] == config.engine and item["status"] == "COMPLETED"), None)
        if candidate:
            timings[row["workers"]] = candidate["timing"]["median_seconds"]
    if timings:
        reference_workers = min(timings)
        reference_time = timings[reference_workers]
        for row in rows:
            elapsed = timings.get(row["workers"])
            if elapsed:
                ratio = row["workers"] / reference_workers
                row.update(reference_workers=reference_workers,
                           scaling_speedup=reference_time / elapsed,
                           scaling_efficiency=(reference_time / elapsed) if weak else (reference_time / elapsed / ratio))
    completed = all(row["benchmark"].get("status") == "COMPLETED" and any(
        item["engine"] == config.engine and item["status"] == "COMPLETED"
        for item in row["benchmark"]["engines"]) for row in rows)
    value = {"schema_version": 1, "status": "COMPLETED" if completed else "FAILED",
             "kind": "weak" if weak else "strong", "rows": rows,
             "definition": "weak grows events and independent instruments with workers; strong holds input fixed",
             "efficiency_definition": "T(reference)/T(p)" if weak else "T(reference)/T(p)/(p/reference_workers)",
             "engine": config.engine}
    atomic_json(output, value)
    return value
