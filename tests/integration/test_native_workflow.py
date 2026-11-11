from dataclasses import replace
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
                         ("sequential", "multi-single"), ("openmp", "multi-multi"), ("hybrid", "multi-multi")]:
        if not engine_available(native_capabilities, engine):
            continue
        candidate = JobManager(registry).run(replace(config, engine=engine, cpu_mode=mode))
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
