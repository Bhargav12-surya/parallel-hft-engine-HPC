import sys
import threading
from dataclasses import replace

import pytest

from hftlab.configuration import SimulationConfig
from hftlab.experiments import ExperimentRegistry
from hftlab.orchestration import JobCancelled, JobManager


def setup_process(monkeypatch, tmp_path):
    from hftlab import orchestration
    from hftlab.datasets import generate_dataset
    path = tmp_path / "market.csv"
    generate_dataset(path, events=100, instruments=2)
    stub = tmp_path / "deliberately_slow_backend.py"
    stub.write_text("import time\nprint('{\"schema_version\":1,\"type\":\"progress\",\"events_processed\":1}',flush=True)\ntime.sleep(30)\n")
    monkeypatch.setattr(orchestration, "validate_engine", lambda config: {"test_process": True})
    monkeypatch.setattr(orchestration, "find_runner", lambda: stub)
    monkeypatch.setattr(orchestration, "command", lambda config, output: [sys.executable, str(stub)])
    return SimulationConfig(path, output_dir=tmp_path / "results", timeout=2)


def test_timeout_kills_child_and_persists_failure(monkeypatch, tmp_path):
    config = setup_process(monkeypatch, tmp_path)
    registry = ExperimentRegistry(tmp_path / "registry")
    manager = JobManager(registry)
    with pytest.raises(TimeoutError):
        manager.run(config)
    assert manager.state == "FAILED"
    assert manager._process.poll() is not None
    assert registry.list()[0]["status"] == "FAILED"


def test_cancel_reaps_process(monkeypatch, tmp_path):
    config = setup_process(monkeypatch, tmp_path)
    manager = JobManager(ExperimentRegistry(tmp_path / "registry"))
    failures = []
    ready = threading.Event()

    def run():
        try:
            manager.run(config, on_event=lambda event: ready.set() if event.get("type") == "progress" else None)
        except JobCancelled as error:
            failures.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    assert ready.wait(timeout=5)
    manager.cancel()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert failures
    assert manager.state == "CANCELLED"
    assert manager._process.poll() is not None


def test_external_cancellation_interrupts_quiet_supervised_job(monkeypatch, tmp_path):
    config = setup_process(monkeypatch, tmp_path)
    manager = JobManager(ExperimentRegistry(tmp_path / "registry"))
    cancelled = threading.Event()

    def on_event(event):
        if event.get("type") == "progress":
            cancelled.set()

    with pytest.raises(JobCancelled):
        manager.run(config, on_event=on_event, quiet=True, cancel_event=cancelled)
    assert manager.state == "CANCELLED"
    assert manager._process.poll() is not None


def test_timeout_reaps_reader_when_output_queue_is_full(monkeypatch, tmp_path):
    import time

    config = setup_process(monkeypatch, tmp_path)
    (tmp_path / "deliberately_slow_backend.py").write_text(
        "import time\nfor _ in range(10000): print('{\"schema_version\":1,\"type\":\"progress\"}',flush=True)\ntime.sleep(30)\n")
    readers_before = {thread.ident for thread in threading.enumerate() if thread.name == "hftlab-jsonl"}
    manager = JobManager(ExperimentRegistry(tmp_path / "registry"))
    with pytest.raises(TimeoutError):
        manager.run(replace(config, timeout=.5), on_event=lambda event: time.sleep(.05)
                    if event.get("type") == "progress" else None)
    assert manager._process.poll() is not None
    readers_after = {thread.ident for thread in threading.enumerate() if thread.name == "hftlab-jsonl"}
    assert readers_after == readers_before
