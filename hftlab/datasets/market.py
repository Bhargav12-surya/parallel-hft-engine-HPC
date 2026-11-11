"""Deterministic offline event generation and streaming interchange.

CSV is the canonical native interface. HFTDATA binary files use explicitly packed
little-endian records and SHA256 payload checksums, never persisted C structs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import struct
import tempfile
from collections import Counter, deque
from typing import Iterator

EVENT_FIELDS = ("event_id", "timestamp_ns", "instrument_id", "event_type", "side",
                "price_ticks", "quantity", "order_id", "reference_order_id")
SCENARIOS = ("normal", "high-volatility", "high-liquidity", "low-liquidity",
             "shock", "correlated", "stress", "predictable")
EVENT_TYPES = {"QUOTE": 0, "LIMIT": 1, "MARKET": 2, "CANCEL": 3, "MODIFY": 4}
_TYPE_NAMES = {value: key for key, value in EVENT_TYPES.items()}
_HEADER = struct.Struct("<8sIIQ32s")
_RECORD = struct.Struct("<QQIBBqQQQ")
_MAGIC = b"HFTDATA\x00"
MAX_INTEGER = (1 << 63) - 1


class DatasetError(ValueError):
    """A dataset violates its explicit interchange contract."""


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(row: dict, line: int) -> dict:
    if set(row) != set(EVENT_FIELDS):
        raise DatasetError(f"Row {line}: expected exactly {', '.join(EVENT_FIELDS)}")
    event = {}
    for field in EVENT_FIELDS:
        raw = row[field]
        if raw is None:
            raise DatasetError(f"Row {line}: missing {field}")
        if field in ("event_type", "side"):
            event[field] = str(raw).strip().upper()
        else:
            try:
                number = int(raw)
            except (ValueError, TypeError, OverflowError) as exc:
                raise DatasetError(f"Row {line}: {field} must be an integer") from exc
            if isinstance(raw, float) and raw != number:
                raise DatasetError(f"Row {line}: fractional {field}")
            if number < 0 or number > MAX_INTEGER:
                raise DatasetError(f"Row {line}: {field} outside nonnegative int64 range")
            event[field] = number
    if event["event_type"] not in EVENT_TYPES:
        raise DatasetError(f"Row {line}: unsupported event type {event['event_type']!r}")
    if event["side"] not in ("BUY", "SELL"):
        raise DatasetError(f"Row {line}: side must be BUY or SELL")
    if event["event_id"] == 0:
        raise DatasetError(f"Row {line}: event ID must be positive")
    if event["instrument_id"] > 65535:
        raise DatasetError(f"Row {line}: instrument ID exceeds 65535")
    if event["event_type"] != "CANCEL":
        if event["price_ticks"] <= 0 or event["quantity"] <= 0:
            raise DatasetError(f"Row {line}: price and quantity must be positive")
    if event["event_type"] in ("LIMIT", "MARKET", "MODIFY") and event["order_id"] == 0:
        raise DatasetError(f"Row {line}: order ID must be positive")
    if event["event_type"] in ("CANCEL", "MODIFY") and event["reference_order_id"] == 0:
        raise DatasetError(f"Row {line}: cancellation/modification requires reference ID")
    if event["event_type"] not in ("CANCEL", "MODIFY") and event["reference_order_id"] != 0:
        raise DatasetError(f"Row {line}: unexpected reference ID")
    return event


def iter_events(path: str | Path) -> Iterator[dict]:
    """Read canonical records in file order without loading the dataset in memory."""
    source = Path(path)
    if source.suffix.lower() in (".hft", ".hftdata", ".bin"):
        yield from _iter_binary(source)
        return
    if source.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise DatasetError("Parquet import requires optional pyarrow") from exc
        row_number = 1
        for batch in parquet.ParquetFile(source).iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                row_number += 1
                yield _normalise(row, row_number)
        return
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(EVENT_FIELDS):
                raise DatasetError(f"CSV header must be exactly {','.join(EVENT_FIELDS)}")
            for line, row in enumerate(reader, 2):
                yield _normalise(row, line)
    except (UnicodeError, csv.Error) as exc:
        raise DatasetError(f"Invalid CSV: {exc}") from exc


def validate_dataset(path: str | Path) -> dict:
    """Validate ordering, unique IDs and references with disk-backed indexes.

    References must identify an earlier LIMIT for the same instrument. A valid
    reference may already have filled during strategy-dependent replay; the
    exchange then records its cancellation as rejected rather than changing time.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise DatasetError(f"Dataset does not exist: {source}")
    count = 0
    last_timestamp = -1
    first_timestamp = None
    instruments: Counter = Counter()
    kinds: Counter = Counter()
    minimum, maximum, volume = MAX_INTEGER, 0, 0
    with tempfile.TemporaryDirectory(prefix="hftlab-validation-") as directory:
        connection = sqlite3.connect(str(Path(directory) / "ids.sqlite"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE events(id INTEGER PRIMARY KEY)")
            connection.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY,instrument INTEGER,kind TEXT)")
            for event in iter_events(source):
                count += 1
                timestamp = event["timestamp_ns"]
                if timestamp < last_timestamp:
                    raise DatasetError(f"Event {event['event_id']}: timestamp decreases")
                first_timestamp = timestamp if first_timestamp is None else first_timestamp
                last_timestamp = timestamp
                try:
                    connection.execute("INSERT INTO events VALUES (?)", (event["event_id"],))
                except sqlite3.IntegrityError as exc:
                    raise DatasetError(f"Duplicate event ID {event['event_id']}") from exc
                kind = event["event_type"]
                reference = event["reference_order_id"]
                if reference:
                    target = connection.execute("SELECT instrument,kind FROM orders WHERE id=?", (reference,)).fetchone()
                    if target != (event["instrument_id"], "LIMIT"):
                        raise DatasetError(f"Event {event['event_id']}: invalid reference {reference}")
                if kind in ("LIMIT", "MARKET", "MODIFY"):
                    try:
                        connection.execute("INSERT INTO orders VALUES (?,?,?)", (event["order_id"], event["instrument_id"], "LIMIT" if kind == "MODIFY" else kind))
                    except sqlite3.IntegrityError as exc:
                        raise DatasetError(f"Duplicate order ID {event['order_id']}") from exc
                instruments[event["instrument_id"]] += 1
                kinds[kind] += 1
                if event["price_ticks"]:
                    minimum = min(minimum, event["price_ticks"])
                    maximum = max(maximum, event["price_ticks"])
                volume += event["quantity"]
        finally:
            connection.close()
    if count == 0:
        raise DatasetError("Dataset contains no events")
    return {"schema_version": 1, "path": str(source), "valid": True,
            "events": count, "instruments": len(instruments),
            "instrument_counts": dict(sorted(instruments.items())),
            "event_types": dict(sorted(kinds.items())),
            "start_timestamp_ns": first_timestamp, "end_timestamp_ns": last_timestamp,
            "price_min_ticks": minimum, "price_max_ticks": maximum,
            "total_quantity": volume, "sha256": file_hash(source),
            "bytes": source.stat().st_size}


def inspect_dataset(path: str | Path, preview: int = 8) -> dict:
    result = validate_dataset(path)
    samples = []
    for event in iter_events(path):
        samples.append(event)
        if len(samples) >= preview:
            break
    result["preview"] = samples
    manifest = Path(str(path) + ".manifest.json")
    if manifest.exists():
        result["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    return result


def generate_dataset(path: str | Path, events: int = 10000, instruments: int = 4,
                     seed: int = 42, scenario: str = "normal", **kwargs) -> dict:
    """Write reproducible, bounded-memory synthetic market events.

    Correlated Gaussian innovations, sinusoidal intraday intensity, regime
    volatility, Poisson arrivals and optional shocks generate offline marks.
    QUOTE refreshes native external bid/ask liquidity; LIMIT/MARKET/CANCEL add
    actual exchange flow. Predictable is deliberately educational synthetic data.
    """
    if not isinstance(events, int) or not 1 <= events <= 100_000_000:
        raise DatasetError("events must be an integer in [1, 100000000]")
    if not isinstance(instruments, int) or not 1 <= instruments <= 65536:
        raise DatasetError("instruments must be in [1, 65536]")
    if scenario not in SCENARIOS:
        raise DatasetError(f"scenario must be one of {', '.join(SCENARIOS)}")
    initial_price = int(kwargs.pop("initial_price", 10000))
    volatility = float(kwargs.pop("volatility", 2.0))
    intensity = float(kwargs.pop("arrival_intensity", 50000.0))
    correlation = float(kwargs.pop("correlation", 0.75 if scenario == "correlated" else 0.25))
    buy_imbalance = float(kwargs.pop("buy_imbalance", 0.5))
    cancel_rate = float(kwargs.pop("cancel_rate", 0.07))
    shock_fraction = float(kwargs.pop("shock_fraction", 0.04))
    if kwargs:
        raise DatasetError(f"Unknown generation parameters: {', '.join(kwargs)}")
    if initial_price < 100 or not math.isfinite(volatility) or volatility < 0:
        raise DatasetError("initial_price must be >=100; volatility finite and nonnegative")
    if not math.isfinite(intensity) or intensity <= 0 or not 0 <= correlation <= 1:
        raise DatasetError("arrival intensity must be positive; correlation in [0,1]")
    if not 0 <= buy_imbalance <= 1 or not 0 <= cancel_rate <= .5:
        raise DatasetError("buy_imbalance must be in [0,1]; cancel_rate in [0,.5]")
    if not math.isfinite(shock_fraction) or not 0 <= shock_fraction < 1:
        raise DatasetError("shock_fraction must be in [0,1)")
    destination = Path(path).resolve()
    if destination.suffix.lower() in (".bin", ".hft", ".hftdata"):
        temporary_csv = destination.with_name(destination.name + ".source.csv")
        manifest = generate_dataset(temporary_csv, events, instruments, seed, scenario,
                                    initial_price=initial_price, volatility=volatility,
                                    arrival_intensity=intensity, correlation=correlation,
                                    buy_imbalance=buy_imbalance, cancel_rate=cancel_rate,
                                    shock_fraction=shock_fraction)
        export_binary(temporary_csv, destination)
        temporary_csv.unlink()
        Path(str(temporary_csv) + ".manifest.json").unlink(missing_ok=True)
        manifest.update(path=str(destination), sha256=file_hash(destination), bytes=destination.stat().st_size, format="hftdata-v1")
        _atomic_json(Path(str(destination) + ".manifest.json"), manifest)
        return manifest
    if destination.suffix.lower() != ".csv":
        raise DatasetError("Generate to .csv or .hftdata; use import for Parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    rng = random.Random(seed)
    prices = [float(initial_price + i * 100) for i in range(instruments)]
    references = [deque(maxlen=32) for _ in range(instruments)]
    timestamp = 0
    order_id = 0
    kinds: Counter = Counter()
    scale = {"high-volatility": 5.0, "low-liquidity": 2.0, "stress": 3.0}.get(scenario, 1.0)
    liquidity = {"high-liquidity": 1000, "low-liquidity": 10, "stress": 500}.get(scenario, 100)
    common = 0.0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, lineterminator="\n")
            writer.writeheader()
            for index in range(events):
                instrument = index % instruments if scenario in ("stress", "correlated", "predictable") else rng.randrange(instruments)
                if index % instruments == 0:
                    common = rng.gauss(0.0, 1.0)
                local_time = index / max(instruments, 1)
                intraday = 0.65 + 0.7 * abs(math.cos(math.pi * index / max(events, 1)))
                timestamp += max(1, int(rng.expovariate(intensity * intraday) * 1e9))
                innovation = math.sqrt(correlation) * common + math.sqrt(1 - correlation) * rng.gauss(0, 1)
                regime = 1.8 if (index // max(128, instruments * 32)) % 3 == 1 else 0.8
                movement = volatility * scale * regime * innovation
                if scenario == "predictable":
                    movement = 3.5 * math.sin(local_time / 24.0) + rng.gauss(0, .25)
                elif scenario == "normal":
                    movement += .002 * (initial_price + instrument * 100 - prices[instrument])
                if scenario == "shock" and events // 2 <= index < events // 2 + instruments:
                    prices[instrument] *= 1 - shock_fraction
                prices[instrument] = max(100.0, prices[instrument] + movement)
                price = max(1, round(prices[instrument]))
                side = "BUY" if rng.random() < buy_imbalance else "SELL"
                if scenario == "predictable":
                    side = "BUY" if movement > 0 else "SELL"
                quantity = max(1, int(rng.expovariate(1.0 / liquidity)))
                draw = rng.random()
                reference = 0
                event_order = 0
                if index < instruments or draw < .65:
                    kind = "QUOTE"
                elif draw < .65 + cancel_rate and references[instrument]:
                    kind = "CANCEL"
                    reference = references[instrument].popleft()
                    quantity = 0
                    price = 0
                elif draw < .9:
                    kind = "LIMIT"
                    order_id += 1
                    event_order = order_id
                    # Far passive flow does not intentionally cross fresh marks.
                    price = max(1, price + (20 if side == "SELL" else -20))
                    if len(references[instrument]) == references[instrument].maxlen:
                        # Emit cancellation in place of another passive order so
                        # resting background orders remain bounded per instrument.
                        kind = "CANCEL"
                        reference = references[instrument].popleft()
                        quantity = 0
                        price = 0
                        event_order = 0
                    else:
                        references[instrument].append(event_order)
                else:
                    kind = "MARKET"
                    order_id += 1
                    event_order = order_id
                row = dict(zip(EVENT_FIELDS, (index + 1, timestamp, instrument, kind,
                                             side, price, quantity, event_order, reference)))
                writer.writerow(row)
                kinds[kind] += 1
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest = {"schema_version": 1, "format": "csv", "path": str(destination),
                "events": events, "instruments": instruments, "seed": seed,
                "scenario": scenario, "event_types": dict(kinds),
                "sha256": file_hash(destination), "bytes": destination.stat().st_size,
                "parameters": {"initial_price": initial_price, "volatility": volatility,
                               "arrival_intensity": intensity, "correlation": correlation,
                               "buy_imbalance": buy_imbalance, "cancel_rate": cancel_rate,
                               "shock_fraction": shock_fraction},
                "educational_pattern": scenario == "predictable",
                "note": "Offline synthetic events; no evidence of real-market profitability."}
    _atomic_json(Path(str(destination) + ".manifest.json"), manifest)
    return manifest


def import_dataset(source: str | Path, destination: str | Path) -> dict:
    source_path, output_path = Path(source).resolve(), Path(destination).resolve()
    if source_path == output_path:
        raise DatasetError("Import source and destination must differ")
    validate_dataset(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".hft", ".hftdata", ".bin"):
        export_binary(source_path, output_path)
    else:
        if output_path.suffix.lower() != ".csv":
            raise DatasetError("Import destination must be .csv or .hftdata")
        temporary = output_path.with_name(output_path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(iter_events(source_path))
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    result = validate_dataset(output_path)
    result.update(source=str(source_path), source_sha256=file_hash(source_path), format=output_path.suffix.lower().lstrip("."))
    _atomic_json(Path(str(output_path) + ".manifest.json"), result)
    return result


def export_binary(source: str | Path, destination: str | Path) -> dict:
    """Export explicit HFTDATA v1: 56-byte header and 54-byte records."""
    output = Path(destination).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    digest = hashlib.sha256()
    count = 0
    try:
        with temporary.open("w+b") as handle:
            handle.write(b"\0" * _HEADER.size)
            for event in iter_events(source):
                payload = _RECORD.pack(event["event_id"], event["timestamp_ns"],
                                       event["instrument_id"], EVENT_TYPES[event["event_type"]],
                                       int(event["side"] == "BUY"), event["price_ticks"],
                                       event["quantity"], event["order_id"], event["reference_order_id"])
                handle.write(payload)
                digest.update(payload)
                count += 1
            handle.seek(0)
            handle.write(_HEADER.pack(_MAGIC, 1, _RECORD.size, count, digest.digest()))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"schema_version": 1, "events": count, "record_bytes": _RECORD.size,
            "payload_sha256": digest.hexdigest(), "path": str(output)}


def _iter_binary(path: Path) -> Iterator[dict]:
    with path.open("rb") as handle:
        header = handle.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise DatasetError("Truncated binary header")
        magic, version, record_size, count, expected = _HEADER.unpack(header)
        if magic != _MAGIC or version != 1 or record_size != _RECORD.size:
            raise DatasetError("Unsupported binary magic, version, or record layout")
        if count > 100_000_000 or path.stat().st_size != _HEADER.size + count * record_size:
            raise DatasetError("Binary length/count mismatch")
        # Validate the full checksum before exposing any records to consumers.
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.digest() != expected:
            raise DatasetError("Binary payload checksum mismatch")
        handle.seek(_HEADER.size)
        for index in range(count):
            fields = list(_RECORD.unpack(handle.read(record_size)))
            if fields[3] not in _TYPE_NAMES or fields[4] not in (0, 1):
                raise DatasetError(f"Binary record {index}: invalid event type/side")
            fields[3] = _TYPE_NAMES[fields[3]]
            fields[4] = "BUY" if fields[4] else "SELL"
            yield _normalise(dict(zip(EVENT_FIELDS, fields)), index + 1)
