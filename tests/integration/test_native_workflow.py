from dataclasses import replace
import csv
import subprocess
from pathlib import Path

import pytest

from hftlab.benchmarks import benchmark, compare_results
from hftlab.configuration import SimulationConfig
from hftlab.engines import capabilities, engine_available, find_runner
from hftlab.experiments import ExperimentRegistry
from hftlab.orchestration import JobManager


@pytest.fixture(scope="module")
def native_capabilities():
    try:
        find_runner()
    except RuntimeError as error:
        pytest.skip(str(error))
    return capabilities()


@pytest.fixture
def market(tmp_path):
    from hftlab.datasets import generate_dataset
    path = tmp_path / "market.csv"
    generate_dataset(path, events=1200, instruments=4, seed=11)
    return path


@pytest.mark.native
@pytest.mark.parametrize("strategy", ["momentum", "mean-reversion", "market-making", "stat-arb"])
def test_engine_parity(native_capabilities, market, tmp_path, strategy):
    config = SimulationConfig(market, strategy=strategy, output_dir=tmp_path / "results")
    registry = ExperimentRegistry(tmp_path / "experiments")
    events = []
    baseline = JobManager(registry).run(config, on_event=events.append)
    assert baseline["events_processed"] == 1200
    assert any(event.get("state") == "COMPLETED" for event in events)
    for engine, mode in [("threaded", "single-multi"), ("threaded", "multi-multi"),
                         ("sequential", "multi-single"), ("openmp", "multi-multi"), ("hybrid", "multi-multi"),
                         ("mpi", "multi-multi"), ("cuda", "multi-multi")]:
        if not engine_available(native_capabilities, engine):
            continue
        if engine == "mpi":
            from hftlab.engines import mpi_launcher
            if not mpi_launcher():
                continue
        candidate = JobManager(registry).run(replace(config, engine=engine, cpu_mode=mode, threads=2))
        assert compare_results(baseline, candidate)["passed"], (engine, strategy, baseline, candidate)
    assert len(registry.list()) >= 4


@pytest.mark.native
def test_unavailable_engine_rejected(native_capabilities, market, tmp_path):
    unavailable = [name for name in ("cuda", "mpi") if not engine_available(native_capabilities, name)]
    if not unavailable:
        pytest.skip("All optional engines compiled")
    manager = JobManager(ExperimentRegistry(tmp_path / "registry"))
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        manager.run(SimulationConfig(market, engine=unavailable[0], output_dir=tmp_path / "out"))
    assert manager.state == "FAILED"


@pytest.mark.native
def test_benchmark_export(native_capabilities, market, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = benchmark(SimulationConfig(market, output_dir=tmp_path / "runs"), repeats=2,
                       engines=["sequential", "threaded"], output=tmp_path / "benchmark.json", warmups=0)
    assert [row["status"] for row in result["engines"]] == ["COMPLETED", "COMPLETED"]
    assert result["engines"][0]["speedup"] == 1
    assert (tmp_path / "benchmark.json").is_file()


def test_result_validation_rejects_corruption():
    from hftlab.orchestration import _validate_result
    with pytest.raises(ValueError):
        _validate_result({"schema_version": 2})


@pytest.mark.parametrize("changes", [{"trades": -1}, {"orders": True}, {"fees": -1},
                                      {"checksum": "bad"}, {"throughput": 999}, {"status": "FAILED"}])
def test_result_validation_checks_contract(changes):
    from hftlab.orchestration import _validate_result
    result = {"schema_version": 1, "status": "COMPLETED", "wall_seconds": .1,
              "events_processed": 100, "throughput": 1000, "trades": 3, "orders": 5,
              "pnl": 1, "equity": 100001, "fees": .1, "drawdown": 2, "checksum": "0123456789abcdef"}
    _validate_result(result)
    with pytest.raises(ValueError):
        _validate_result({**result, **changes})


@pytest.mark.native
def test_binary_import_to_native_simulation(native_capabilities, market, tmp_path):
    from hftlab.datasets import export_binary
    binary = tmp_path / "events.hftdata"
    export_binary(market, binary)
    registry = ExperimentRegistry(tmp_path / "experiments")
    config = SimulationConfig(market, output_dir=tmp_path / "runs")
    reference = JobManager(registry).run(config)
    imported = JobManager(registry).run(replace(config, dataset=binary))
    assert compare_results(reference, imported)["passed"]
    assert (Path(imported["output_dir"]) / "input.csv").is_file()


@pytest.mark.native
def test_accepted_csv_variants_replay_through_canonical_snapshot(native_capabilities, market, tmp_path):
    with market.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    variant = tmp_path / "quoted-bom-lowercase.csv"
    with variant.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
        writer.writerow(rows[0])
        writer.writerows([[value.lower() for value in row] for row in rows[1:]])
    registry = ExperimentRegistry(tmp_path / "registry")
    config = SimulationConfig(market, output_dir=tmp_path / "runs")
    reference = JobManager(registry).run(config)
    candidate = JobManager(registry).run(replace(config, dataset=variant))
    assert compare_results(reference, candidate)["passed"]
    assert Path(candidate["output_dir"], "input.csv").read_bytes() == market.read_bytes()
    profiling = candidate["profiling"]
    for key in ("cuda_h2d_seconds", "cuda_kernel_seconds", "cuda_d2h_seconds",
                "mpi_communication_seconds", "bytes_transferred", "feature_seconds"):
        assert profiling[key] is None, key


@pytest.mark.native
@pytest.mark.parametrize("flag", ["--fee-bps", "--max-loss", "--latency-ns", "--slippage-ticks"])
def test_native_empty_numeric_argument_is_rejected(native_capabilities, flag):
    completed = subprocess.run([str(find_runner()), flag, ""], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 2
    assert "outside supported range" in completed.stdout


@pytest.mark.native
def test_native_oversized_model_header_is_rejected(native_capabilities, tmp_path):
    model = tmp_path / "bad.hftmodel"
    model.write_text("HFTMODEL 1 6 " + "9" * 1000 + " 3\n", encoding="utf-8")
    features = tmp_path / "features.txt"
    features.write_text("0 0 0 0 0 0\n", encoding="utf-8")
    completed = subprocess.run([str(find_runner()), "--infer", "--model", str(model), "--features", str(features)],
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 2
    assert "Invalid native model" in completed.stdout


def test_command_has_no_shell_interpolation(tmp_path):
    from hftlab import engines
    path = tmp_path / "odd $(name); data.csv"
    config = SimulationConfig(path)
    original = engines.find_runner
    try:
        engines.find_runner = lambda: Path("runner")
        command = engines.command(config, tmp_path / "result.json")
        assert str(path) in command
    finally:
        engines.find_runner = original
