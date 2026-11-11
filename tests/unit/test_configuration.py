from pathlib import Path

import pytest

from hftlab.configuration import SimulationConfig


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("event_id,timestamp_ns,instrument_id,event_type,side,price_ticks,quantity,order_id,reference_order_id\n1,1,0,QUOTE,BUY,100,10,1,0\n")
    return path


@pytest.mark.parametrize("fields", [
    {"engine": "fake"}, {"strategy": "fake"}, {"threads": 0}, {"threads": 1025},
    {"capital": float("nan")}, {"capital": -1}, {"fee_bps": float("inf")},
    {"position_limit": 0}, {"max_order": 101}, {"latency_ns": -1}, {"slippage_ticks": -1},
    {"engine": "hybrid", "components": ["cpu"]},
    {"engine": "hybrid", "components": ["cpu", "cpu"]},
    {"engine": "hybrid", "components": ["cpu", "fake"]}, {"strategy": "ai"},
    {"timeout": 0}, {"schema_version": 2}, {"cpu_mode": "fictional"},
])
def test_reject_invalid(dataset, fields):
    with pytest.raises(ValueError):
        SimulationConfig(dataset=dataset, **fields).validate()


def test_serializable_roundtrip(dataset):
    value = SimulationConfig(dataset=dataset).validate()
    assert SimulationConfig.from_dict(value.to_dict()).validate().to_dict() == value.to_dict()


def test_unknown_fields_rejected(dataset):
    with pytest.raises(ValueError, match="Unknown"):
        SimulationConfig.from_dict({"dataset": str(dataset), "shady": True})


def test_missing_dataset():
    with pytest.raises(ValueError, match="does not exist"):
        SimulationConfig(Path("missing-file-for-test")).validate()
