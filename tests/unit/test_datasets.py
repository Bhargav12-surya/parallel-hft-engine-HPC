import csv
import hashlib

import pytest

from hftlab.datasets import (DatasetError, EVENT_FIELDS, SCENARIOS, export_binary,
                             generate_dataset, import_dataset, iter_events,
                             validate_dataset)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def event(event_id=1, timestamp_ns=1, **kwargs):
    result = dict(zip(EVENT_FIELDS, (event_id, timestamp_ns, 0, "QUOTE", "BUY", 10000, 10, 0, 0)))
    result.update(kwargs)
    return result


def test_generation_is_byte_reproducible(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    one = generate_dataset(first, events=300, instruments=3, seed=19)
    two = generate_dataset(second, events=300, instruments=3, seed=19)
    assert first.read_bytes() == second.read_bytes()
    assert one["sha256"] == two["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert validate_dataset(first)["events"] == 300


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios_validate_and_emit_native_flow(tmp_path, scenario):
    path = tmp_path / f"{scenario}.csv"
    manifest = generate_dataset(path, events=500, instruments=3, scenario=scenario)
    metadata = validate_dataset(path)
    assert metadata["instruments"] == 3
    assert metadata["event_types"]["QUOTE"] > 0
    assert metadata["event_types"]["LIMIT"] > 0
    assert metadata["event_types"]["MARKET"] > 0
    assert manifest["educational_pattern"] == (scenario == "predictable")


def test_binary_roundtrip_and_payload_corruption(tmp_path):
    csv_path, binary_path, imported_path = (tmp_path / name for name in ("a.csv", "b.hftdata", "c.csv"))
    generate_dataset(csv_path, events=100, instruments=2)
    export_binary(csv_path, binary_path)
    assert list(iter_events(csv_path)) == list(iter_events(binary_path))
    assert import_dataset(binary_path, imported_path)["events"] == 100
    assert imported_path.read_bytes() == csv_path.read_bytes()
    data = bytearray(binary_path.read_bytes())
    data[-1] ^= 1
    binary_path.write_bytes(data)
    with pytest.raises(DatasetError, match="checksum"):
        list(iter_events(binary_path))


def test_binary_count_overflow_rejected(tmp_path):
    from hftlab.datasets.market import _HEADER, _MAGIC, _RECORD
    path = tmp_path / "bad.hftdata"
    path.write_bytes(_HEADER.pack(_MAGIC, 1, _RECORD.size, (1 << 64) - 1, bytes(32)))
    with pytest.raises(DatasetError, match="length/count"):
        list(iter_events(path))


@pytest.mark.parametrize("rows,message", [
    ([], "no events"),
    ([event(), event()], "Duplicate event"),
    ([event(timestamp_ns=3), event(2, 2)], "timestamp decreases"),
    ([event(quantity=0)], "positive"),
    ([event(price_ticks=-1)], "int64"),
    ([event(side="LEFT")], "BUY or SELL"),
    ([event(event_type="CANCEL", quantity=0, reference_order_id=999)], "invalid reference"),
    ([event(event_type="LIMIT", order_id=2), event(2,2,event_type="LIMIT",order_id=2)], "Duplicate order"),
])
def test_invalid_datasets_rejected(tmp_path, rows, message):
    path = tmp_path / "invalid.csv"
    write_rows(path, rows)
    with pytest.raises(DatasetError, match=message):
        validate_dataset(path)


def test_valid_reference_and_stable_equal_timestamp(tmp_path):
    path = tmp_path / "flow.csv"
    rows = [event(event_type="LIMIT", order_id=42),
            event(2, 1, event_type="CANCEL", price_ticks=0, quantity=0, reference_order_id=42)]
    write_rows(path, rows)
    assert validate_dataset(path)["events"] == 2
    assert list(iter_events(path)) == rows


def test_cross_instrument_reference_rejected(tmp_path):
    path = tmp_path / "flow.csv"
    write_rows(path, [event(event_type="LIMIT", order_id=42), event(2,2,instrument_id=1,event_type="CANCEL",price_ticks=0,quantity=0,reference_order_id=42)])
    with pytest.raises(DatasetError, match="invalid reference"):
        validate_dataset(path)


def test_header_and_integer_validation(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("price,volume\n1,2\n")
    with pytest.raises(DatasetError, match="header"):
        list(iter_events(path))
    write_rows(path, [event(timestamp_ns="1.5")])
    with pytest.raises(DatasetError, match="integer"):
        list(iter_events(path))


@pytest.mark.parametrize("kwargs", [{"events": 0}, {"events": 100_000_001}, {"instruments": 0}, {"volatility": float("nan")}, {"correlation": 2}, {"cancel_rate": -.1}, {"arrival_intensity": 0}, {"unknown": 1}])
def test_generation_configuration_rejected(tmp_path, kwargs):
    with pytest.raises(DatasetError):
        generate_dataset(tmp_path / "invalid.csv", **kwargs)


def test_import_preserves_source(tmp_path):
    source = tmp_path / "source.csv"
    destination = tmp_path / "import.csv"
    generate_dataset(source, events=100)
    before = source.read_bytes()
    import_dataset(source, destination)
    assert source.read_bytes() == destination.read_bytes() == before
    with pytest.raises(DatasetError, match="must differ"):
        import_dataset(source, source)
