"""Small dependency-free helpers shared by command and job layers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporary_path(path: str | Path, suffix: str = ".tmp") -> Path:
    """Reserve a unique sibling file for a caller to replace or remove."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(handle)
    return Path(temporary)


def replace_file(source: str | Path, destination: str | Path) -> None:
    """Replace atomically, allowing short Windows sharing-lock contention."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if os.name != "nt" or error.winerror not in (5, 32) or attempt == 4:
                raise
            time.sleep(.01 * 2**attempt)


def atomic_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def finite_number(value: Any, name: str, *, minimum: float = 0, maximum: float = 1e15) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside the supported numeric range") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}")
    return number
