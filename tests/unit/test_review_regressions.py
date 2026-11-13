"""Regressions found by reviewing configuration, artifacts and failure paths."""
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hftlab import benchmarks, cli, engines
from hftlab.ai import ModelError, load_model
from hftlab.ai.learning import FEATURE_NAMES, LABELS, _export_model, _initial_model
from hftlab.configuration import SimulationConfig
from hftlab.datasets import DatasetError, generate_dataset, export_binary, validate_dataset
from hftlab.datasets.market import _atomic_json
from hftlab.utilities import file_hash


@pytest.fixture
def market(tmp_path):
    path = tmp_path / "market.csv"
    generate_dataset(path, events=100, instruments=2)
    return path


def test_explicit_runner_never_selects_another_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("HFTLAB_RUNNER", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(engines, "PROJECT_ROOT", Path(__file__).resolve().parents[2])
    with pytest.raises(RuntimeError, match="Configured HFTLAB_RUNNER"):
        engines.find_runner()


def test_explicit_mpi_launcher_never_selects_mpirun(monkeypatch):
    monkeypatch.setenv("HFTLAB_MPI_LAUNCHER", "missing-launcher")
    monkeypatch.setattr(engines.shutil, "which", lambda name: "/available/mpirun" if name == "mpirun" else None)
    assert engines.mpi_launcher() is None


@pytest.mark.parametrize("capability", [
    {"engines": {}}, {"schema_version": True, "engines": {}},
    {"schema_version": 1, "engines": []},
    {"schema_version": 1, "engines": {"cuda": "UNAVAILABLE"}},
])
def test_malformed_capabilities_are_rejected(monkeypatch, capability):
    monkeypatch.setattr(engines, "find_runner", lambda: Path("runner"))
    monkeypatch.setattr(engines.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(capability)))
    assert "error" in engines.capabilities()


@pytest.mark.parametrize("changes", [
    {"threads": True}, {"schema_version": True}, {"schema_version": 1.0},
    {"capital": True}, {"capital": "100000"}, {"max_loss": None},
    {"latency_ns": 10**12 + 1}, {"position_limit": 10**8 + 1},
    {"max_order": 10**8 + 1, "position_limit": 10**8 + 1},
    {"components": None}, {"components": [["cpu"]]}, {"components": "cpu,openmp"},
    {"strategy": "ai", "model": []}, {"output_dir": None},
])
def test_configuration_rejects_native_mismatches_and_wrong_types(market, changes):
    with pytest.raises(ValueError):
        SimulationConfig(market, **changes).validate()


@pytest.mark.parametrize("field,value", [("price_ticks", 10**12 + 1), ("quantity", 10**9 + 1)])
def test_dataset_cannot_validate_values_native_rejects(market, field, value):
    with market.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0][field] = value
    with market.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetError, match="native limits"):
        validate_dataset(market)


def test_binary_generation_preserves_existing_source_named_file(tmp_path):
    destination = tmp_path / "market.hftdata"
    existing = tmp_path / "market.hftdata.source.csv"
    existing.write_text("user-owned data", encoding="utf-8")
    generate_dataset(destination, events=40, instruments=2)
    assert existing.read_text(encoding="utf-8") == "user-owned data"
    assert validate_dataset(destination)["events"] == 40


def test_binary_export_cannot_overwrite_source(market):
    before = market.read_bytes()
    with pytest.raises(DatasetError, match="differ"):
        export_binary(market, market)
    assert market.read_bytes() == before


def test_metadata_writers_can_share_a_destination(tmp_path):
    destination = tmp_path / "manifest.json"
    values = [{"writer": index, "payload": str(index) * 1000} for index in range(24)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda value: _atomic_json(destination, value), values))
    assert json.loads(destination.read_text(encoding="utf-8")) in values
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json"]


@pytest.fixture
def numerical_model(tmp_path):
    source = tmp_path / "standalone.hftmodel"
    _export_model(_initial_model(0, random.Random(7)), source)
    return source


def test_unrelated_folder_metadata_does_not_attach_to_standalone_model(numerical_model):
    numerical_model.with_name("metadata.json").write_text('{"sha256":"unrelated"}', encoding="utf-8")
    assert load_model(numerical_model)["metadata"] is None


@pytest.mark.parametrize("metadata", [[], {"schema_version": 2}, {"schema_version": True},
                                       {"schema_version": 1, "feature_names": [], "labels": list(LABELS)}])
def test_incompatible_model_metadata_is_rejected(numerical_model, metadata):
    if isinstance(metadata, dict):
        metadata["sha256"] = file_hash(numerical_model)
    numerical_model.with_suffix(".hftmodel.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelError, match="metadata"):
        load_model(numerical_model)


def test_valid_model_metadata_roundtrip(numerical_model):
    metadata = {"schema_version": 1, "feature_names": list(FEATURE_NAMES),
                "labels": list(LABELS), "sha256": file_hash(numerical_model)}
    numerical_model.with_suffix(".hftmodel.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert load_model(numerical_model)["metadata"] == metadata


def test_failed_capability_probe_cannot_produce_a_successful_benchmark(monkeypatch, market, tmp_path):
    monkeypatch.setattr(benchmarks, "capabilities", lambda: {"engines": {}, "error": "empty response"})
    with pytest.raises(RuntimeError, match="probe failed"):
        benchmarks.benchmark(SimulationConfig(market), output=tmp_path / "report.json")


def test_inconsistent_oracle_stops_candidate_measurement(monkeypatch, market, tmp_path):
    monkeypatch.setattr(benchmarks, "capabilities", lambda: {"engines": {"sequential": True, "threaded": True}})
    calls = []

    class UnstableRunner:
        def run(self, config, **kwargs):
            calls.append(config.engine)
            return {"events_processed": 100, "trades": 1, "orders": 1,
                    "checksum": f"{len(calls):016x}", "wall_seconds": .1, "throughput": 1000}

    monkeypatch.setattr(benchmarks, "JobManager", UnstableRunner)
    result = benchmarks.benchmark(SimulationConfig(market), repeats=2, warmups=0,
                                  engines=["threaded"], output=tmp_path / "report.json")
    assert result["status"] == "FAILED"
    assert calls == ["sequential", "sequential"]


def test_scaling_failure_sets_status_and_cli_exit_code(monkeypatch, market, tmp_path):
    monkeypatch.setattr(benchmarks, "benchmark", lambda *args, **kwargs: {
        "status": "COMPLETED", "engines": [{"engine": "sequential", "status": "COMPLETED"},
                                            {"engine": "cuda", "status": "UNAVAILABLE"}]})
    result = benchmarks.scaling(SimulationConfig(market, engine="cuda"), [1], output=tmp_path / "scaling.json")
    assert result["status"] == "FAILED"
    monkeypatch.setattr(benchmarks, "scaling", lambda *args, **kwargs: result)
    assert cli.main(["scaling", "--dataset", str(market), "--engine", "cuda"]) == 1


def test_failed_demo_comparison_stops_before_training(monkeypatch, tmp_path):
    monkeypatch.setattr("hftlab.datasets.generate_dataset", lambda *args, **kwargs: {})
    monkeypatch.setattr(benchmarks, "benchmark", lambda *args, **kwargs: {"status": "FAILED"})
    with pytest.raises(RuntimeError, match="Demo benchmark failed"):
        cli.run_demo(tmp_path / "demo")


def test_invalid_yaml_returns_a_friendly_cli_error(tmp_path, capsys):
    config = tmp_path / "invalid.yaml"
    config.write_text("dataset: [\n", encoding="utf-8")
    assert cli.main(["simulate", "--config", str(config)]) == 1
    assert "Invalid YAML" in capsys.readouterr().err
