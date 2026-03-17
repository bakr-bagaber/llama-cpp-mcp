"""Lightweight SQLite-backed state store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import BenchmarkRecord


class StateStore:
    """Stores benchmark and routing state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmarks (
                    alias_id TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    placement TEXT NOT NULL,
                    prompt_tps REAL NOT NULL,
                    generation_tps REAL NOT NULL,
                    load_seconds REAL NOT NULL,
                    peak_ram_bytes INTEGER,
                    peak_vram_bytes INTEGER,
                    collected_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS route_events (
                    alias_id TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def add_benchmark(self, record: BenchmarkRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmarks (
                    alias_id, backend, placement, prompt_tps, generation_tps,
                    load_seconds, peak_ram_bytes, peak_vram_bytes, collected_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.alias_id,
                    record.backend.value,
                    record.placement.value,
                    record.prompt_tps,
                    record.generation_tps,
                    record.load_seconds,
                    record.peak_ram_bytes,
                    record.peak_vram_bytes,
                    record.collected_at.isoformat(),
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()

    def list_benchmarks(self, alias_id: str | None = None) -> list[BenchmarkRecord]:
        query = "SELECT alias_id, backend, placement, prompt_tps, generation_tps, load_seconds, peak_ram_bytes, peak_vram_bytes, collected_at, metadata_json FROM benchmarks"
        params: tuple[str, ...] = ()
        if alias_id:
            query += " WHERE alias_id = ?"
            params = (alias_id,)
        query += " ORDER BY collected_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        records: list[BenchmarkRecord] = []
        for row in rows:
            records.append(
                BenchmarkRecord(
                    alias_id=row[0],
                    backend=row[1],
                    placement=row[2],
                    prompt_tps=row[3],
                    generation_tps=row[4],
                    load_seconds=row[5],
                    peak_ram_bytes=row[6],
                    peak_vram_bytes=row[7],
                    collected_at=row[8],
                    metadata=json.loads(row[9]),
                )
            )
        return records

    def record_route(self, alias_id: str, decision_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO route_events (alias_id, decision_json) VALUES (?, ?)",
                (alias_id, decision_json),
            )
            conn.commit()
