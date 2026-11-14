"""Transactional SQLite index; larger artifacts remain ordinary JSON/CSV files."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ExperimentRegistry:
    def __init__(self, root: Path = Path("experiments")):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "registry.sqlite3"
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, engine TEXT NOT NULL,
                strategy TEXT NOT NULL, status TEXT NOT NULL, runtime REAL, pnl REAL,
                manifest TEXT NOT NULL)""")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=15)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, manifest: dict) -> None:
        identifier = str(manifest["id"])
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", identifier):
            raise ValueError("Invalid experiment ID")
        result = manifest.get("result") or {}
        with self._connect() as connection:
            connection.execute("""INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET timestamp=excluded.timestamp,
                engine=excluded.engine,strategy=excluded.strategy,status=excluded.status,
                runtime=excluded.runtime,pnl=excluded.pnl,manifest=excluded.manifest""",
                (identifier, manifest["timestamp"], manifest["config"]["engine"],
                 manifest["config"]["strategy"], manifest["status"], result.get("wall_seconds"),
                 result.get("pnl"), json.dumps(manifest, allow_nan=False, default=str)))

    def list(self, limit: int = 100, search: str = "", sort: str = "date") -> list[dict]:
        columns = {"date": "timestamp DESC", "runtime": "runtime ASC", "pnl": "pnl DESC"}
        if sort not in columns:
            raise ValueError("sort must be date, runtime, or pnl")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT manifest FROM experiments WHERE id LIKE ? OR engine LIKE ? OR strategy LIKE ? "
                "OR json_extract(manifest, '$.config.dataset') LIKE ? "
                f"ORDER BY {columns[sort]} LIMIT ?", (f"%{search}%",)*4 + (max(1, min(10000, limit)),))
            return [json.loads(row["manifest"]) for row in rows]

    def get(self, identifier: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT manifest FROM experiments WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError(f"Experiment not found: {identifier}")
        return json.loads(row["manifest"])

    def delete(self, identifier: str) -> None:
        self.get(identifier)
        with self._connect() as connection:
            connection.execute("DELETE FROM experiments WHERE id=?", (identifier,))
