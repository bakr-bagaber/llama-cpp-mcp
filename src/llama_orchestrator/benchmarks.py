"""Helpers for recording and querying benchmark data."""

from __future__ import annotations

from .models import Backend, BenchmarkRecord, PlacementKind
from .state import StateStore


class BenchmarkService:
    """Small service layer around the state store.

    Keeping benchmark logic in one place makes it easier to swap manual
    records for external `llama-bench` integration later.
    """

    def __init__(self, state: StateStore) -> None:
        self.state = state

    def record_manual_benchmark(
        self,
        *,
        alias_id: str,
        backend: Backend,
        placement: PlacementKind,
        prompt_tps: float,
        generation_tps: float,
        load_seconds: float = 0.0,
        peak_ram_bytes: int | None = None,
        peak_vram_bytes: int | None = None,
        metadata: dict | None = None,
    ) -> BenchmarkRecord:
        record = BenchmarkRecord(
            alias_id=alias_id,
            backend=backend,
            placement=placement,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            load_seconds=load_seconds,
            peak_ram_bytes=peak_ram_bytes,
            peak_vram_bytes=peak_vram_bytes,
            metadata=metadata or {"source": "manual"},
        )
        self.state.add_benchmark(record)
        return record
