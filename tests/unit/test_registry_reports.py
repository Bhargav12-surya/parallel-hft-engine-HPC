import csv

import pytest

from hftlab.benchmarks import compare_results, summarize
from hftlab.experiments import ExperimentRegistry
from hftlab.reports import export_report
from hftlab.utilities import atomic_json, read_json


def test_registry_transactional_lifecycle(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry")
    value = {"id": "test_123", "timestamp": "2026-11-10", "config": {"engine": "sequential", "strategy": "momentum"}, "status": "RUNNING"}
    registry.save(value)
    value.update(status="COMPLETED", result={"wall_seconds": 0.01, "pnl": 3})
    registry.save(value)
    assert registry.get("test_123")["status"] == "COMPLETED"
    assert len(registry.list(search="momentum")) == 1
    assert registry.list(search="absent") == []
    registry.delete("test_123")
    assert registry.list() == []
    with pytest.raises(ValueError):
        registry.get("test_123")


def test_registry_id_cannot_traverse(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    with pytest.raises(ValueError, match="Invalid"):
        registry.save({"id": "../escape"})


def test_reports_escape_and_roundtrip(tmp_path):
    value = {"engine": "<script>alert(1)</script>", "pnl": -3.5, "source": "=HYPERLINK(\"bad\")"}
    html = export_report(value, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert read_json(export_report(value, tmp_path / "report.json")) == value
    with export_report(value, tmp_path / "report.csv").open(newline="", encoding="utf-8") as stream:
        rows = dict(list(csv.reader(stream))[1:])
    assert rows["source"].startswith("'=")
    assert rows["pnl"] == "-3.5"


def test_atomic_json_rejects_nonfinite(tmp_path):
    path = atomic_json(tmp_path / "value.json", {"ok": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"nan": float("nan")})
    assert read_json(path) == {"ok": 1}


def test_correctness_gate():
    baseline = {"events_processed": 10, "trades": 3, "orders": 5, "checksum": "abc", "pnl": 2.0}
    assert compare_results(baseline, baseline)["passed"]
    assert not compare_results(baseline, {**baseline, "checksum": "bad"})["passed"]
    assert not compare_results(baseline, {**baseline, "pnl": 4})["passed"]


def test_timing_statistics():
    stats = summarize([1, 2, 3])
    assert stats["median_seconds"] == 2
    assert stats["stddev_seconds"] == 1
    assert stats["ci95_seconds"][0] < 2 < stats["ci95_seconds"][1]
    assert summarize([1])["ci95_seconds"] is None
    with pytest.raises(ValueError):
        summarize([0])
