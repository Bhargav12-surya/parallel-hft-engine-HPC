"""Supervised native jobs with progress events, cancellation, and atomic artifacts."""
from __future__ import annotations

import json
import math
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from hftlab.configuration import SimulationConfig
from hftlab.engines import command, find_runner, validate_engine
from hftlab.experiments import ExperimentRegistry
from hftlab.utilities import atomic_json, file_hash, read_json, utc_now

STATES = ("CREATED", "VALIDATING", "QUEUED", "STARTING", "RUNNING", "FINALIZING", "COMPLETED", "FAILED", "CANCELLED")


class JobCancelled(RuntimeError):
    pass


class JobManager:
    def __init__(self, registry: ExperimentRegistry | None = None):
        self.registry = registry or ExperimentRegistry()
        self.state = "CREATED"
        self.last_experiment_id: str | None = None
        self._cancel = threading.Event()
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._cancel.set()
        if self._process and self._process.poll() is None:
            self._terminate()

    def _terminate(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            import psutil
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            parent.terminate()
            _, alive = psutil.wait_procs(children + [parent], timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.Error:
                    pass
        except Exception:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
        finally:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def run(self, config: SimulationConfig, on_event: Callable[[dict], None] | None = None, *, quiet: bool = False,
            cancel_event: threading.Event | None = None) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("This job manager already has an active job")
        manifest = None
        model_metadata = None
        started = time.perf_counter()
        self._cancel.clear()
        self._process = None
        callback_error = None

        def emit(event: dict) -> None:
            nonlocal callback_error
            if on_event:
                try:
                    on_event(event)
                except Exception as error:
                    callback_error = str(error)

        def transition(state: str) -> None:
            self.state = state
            emit({"schema_version": 1, "type": "state", "state": state})

        try:
            transition("VALIDATING")
            config.validate()
            capability = validate_engine(config)
            from hftlab.datasets import validate_dataset
            data_manifest = validate_dataset(config.dataset)
            if data_manifest.get("valid") is False:
                raise ValueError(f"Dataset validation failed: {data_manifest.get('errors')}")
            if config.strategy == "ai":
                from hftlab.ai import load_model
                model_metadata = load_model(config.model).get("metadata") or {}
            identifier = utc_now().replace(":", "").replace(".", "").replace("+", "_") + "_" + uuid.uuid4().hex[:8]
            self.last_experiment_id = identifier
            run_dir = config.output_dir / identifier
            run_dir.mkdir(parents=True, exist_ok=False)
            manifest = {"schema_version": 1, "id": identifier, "timestamp": utc_now(),
                        "status": "QUEUED", "config": config.to_dict(), "dataset_sha256": file_hash(config.dataset),
                        "runner_sha256": file_hash(find_runner()), "capabilities": capability,
                        "output_dir": str(run_dir), "result": None}
            manifest["environment"] = {"os": platform.platform(), "processor": platform.processor(),
                                       "logical_processors": os.cpu_count(), "python": sys.version,
                                       "dataset_validation": data_manifest}
            if config.model:
                manifest["model_sha256"] = file_hash(config.model)
            self.registry.save(manifest)
            atomic_json(run_dir / "manifest.json", manifest)
            atomic_json(run_dir / "config.json", config.to_dict())
            transition("QUEUED")
            if self._cancel.is_set() or (cancel_event is not None and cancel_event.is_set()):
                raise JobCancelled("Job cancelled before launch")
            if config.dataset.suffix.lower() != ".csv":
                from dataclasses import replace
                from hftlab.datasets import import_dataset
                imported = run_dir / "input.csv"
                import_dataset(config.dataset, imported)
                config = replace(config, dataset=imported)
            arguments = command(config, run_dir / "result.json")
            if quiet:
                arguments.append("--quiet")
            manifest["command"] = arguments
            manifest["progress_suppressed"] = quiet
            transition("STARTING")
            flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            launch_started = time.perf_counter()
            self._process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace", bufsize=1, **flags)
            transition("RUNNING")
            events: queue.Queue[str | None] = queue.Queue(maxsize=512)

            def read_output() -> None:
                assert self._process is not None and self._process.stdout is not None
                try:
                    for line in self._process.stdout:
                        while True:
                            try:
                                events.put(line, timeout=.2)
                                break
                            except queue.Full:
                                if self._cancel.is_set():
                                    return
                finally:
                    try:
                        events.put(None, timeout=1)
                    except queue.Full:
                        pass

            reader = threading.Thread(target=read_output, name="hftlab-jsonl", daemon=True)
            reader.start()
            peak_rss = 0
            with (run_dir / "native.jsonl").open("w", encoding="utf-8") as log:
                while True:
                    if self._cancel.is_set() or (cancel_event is not None and cancel_event.is_set()):
                        raise JobCancelled("Native process cancelled")
                    if time.perf_counter() - launch_started > config.timeout:
                        raise TimeoutError(f"Native job exceeded timeout ({config.timeout:g}s)")
                    try:
                        line = events.get(timeout=.1)
                    except queue.Empty:
                        if not reader.is_alive() and self._process.poll() is not None:
                            break
                        continue
                    if line is None:
                        break
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict) or event.get("schema_version", 1) != 1:
                            raise ValueError("Unsupported progress event")
                    except (ValueError, TypeError):
                        event = {"schema_version": 1, "type": "log", "message": line.strip()}
                    try:
                        import psutil
                        process = psutil.Process(self._process.pid)
                        rss = process.memory_info().rss
                        peak_rss = max(peak_rss, rss)
                        event["memory_bytes"] = rss
                    except Exception:
                        pass
                    emit(event)
            code = self._process.wait(timeout=5)
            reader.join(timeout=2)
            if self._cancel.is_set():
                raise JobCancelled("Native process cancelled")
            if code != 0:
                tail = (run_dir / "native.jsonl").read_text(encoding="utf-8")[-4000:]
                raise RuntimeError(f"Native runner exited with code {code}: {tail.strip()}")
            transition("FINALIZING")
            result = read_json(run_dir / "result.json")
            _validate_result(result, config, data_manifest.get("events"))
            result.update(experiment_id=identifier, output_dir=str(run_dir),
                          orchestration_seconds=time.perf_counter() - started,
                          process_seconds=time.perf_counter() - launch_started,
                          sampled_peak_rss_bytes=peak_rss or None)
            if config.strategy == "ai":
                result["ai"] = {"model_path": str(config.model), "sha256": manifest["model_sha256"],
                                "kind": model_metadata.get("kind"), "backend": model_metadata.get("backend"),
                                "classification_validation": model_metadata.get("validation"),
                                "classification_test": model_metadata.get("test"),
                                "training_dataset_sha256": model_metadata.get("dataset_sha256"),
                                "note": "Classification metrics describe the model's saved chronological splits; current trading replay is reported separately."}
            manifest.update(status="COMPLETED", ended_at=utc_now(), result=result,
                            callback_error=callback_error)
            atomic_json(run_dir / "result.json", result)
            atomic_json(run_dir / "manifest.json", manifest)
            self.registry.save(manifest)
            transition("COMPLETED")
            return result
        except BaseException as error:
            self._terminate()
            state = "CANCELLED" if isinstance(error, (JobCancelled, KeyboardInterrupt)) else "FAILED"
            transition(state)
            if manifest is not None:
                manifest.update(status=state, ended_at=utc_now(), error=str(error))
                atomic_json(Path(manifest["output_dir"]) / "manifest.json", manifest)
                self.registry.save(manifest)
            raise
        finally:
            if self._process and self._process.stdout:
                self._process.stdout.close()
            self._lock.release()


def _validate_result(result: dict, config: SimulationConfig | None = None, expected_events: int | None = None) -> None:
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported native result schema")
    for key in ("wall_seconds", "throughput", "pnl", "equity", "fees", "drawdown"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Native result field {key} is missing or nonfinite")
    for key in ("events_processed", "trades", "orders"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Native result field {key} must be a nonnegative integer")
    if result["wall_seconds"] <= 0 or result["events_processed"] <= 0:
        raise ValueError("Native result contains invalid timing or event count")
    if result["fees"] < 0 or result["drawdown"] < 0 or not re.fullmatch(r"[0-9a-f]{16}", str(result.get("checksum", ""))):
        raise ValueError("Native result contains invalid accounting or checksum")
    if result.get("status") != "COMPLETED":
        raise ValueError("Native result does not report a completed run")
    if not math.isclose(result["throughput"], result["events_processed"] / result["wall_seconds"], rel_tol=1e-9):
        raise ValueError("Native throughput is inconsistent with its event count/timing")
    if expected_events is not None and result["events_processed"] != expected_events:
        raise ValueError("Native processed event count differs from the validated dataset")
    if config is not None and (result.get("engine") != config.engine or result.get("strategy") != config.strategy):
        raise ValueError("Native result engine/strategy differs from the requested configuration")
