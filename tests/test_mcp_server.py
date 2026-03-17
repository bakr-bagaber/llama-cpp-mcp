from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llama_orchestrator.mcp_server import _benchmark_summary_payload, _runtime_diagnostics_payload
from llama_orchestrator.settings import AppSettings


def test_runtime_diagnostics_payload_summarizes_runtime_health() -> None:
    settings = AppSettings.load()
    now = datetime.now(timezone.utc)
    payload = _runtime_diagnostics_payload(
        settings,
        {
            "system_ram_free_bytes": 8 * 1024**3,
            "system_ram_total_bytes": 32 * 1024**3,
            "backends_available": ["cpu", "cuda"],
        },
        [
            {
                "runtime_key": "demo",
                "alias_id": "demo/alias",
                "status": "ready",
                "backend": "cuda",
                "placement": "cpu_dgpu_hybrid",
                "experimental": False,
                "pinned": True,
                "estimated_ram_bytes": 2 * 1024**3,
                "estimated_vram_bytes": 1 * 1024**3,
                "endpoint_url": "http://127.0.0.1:1234",
                "last_used_at": (now - timedelta(seconds=15)).isoformat(),
                "launched_at": (now - timedelta(seconds=120)).isoformat(),
            }
        ],
    )

    assert payload["summary"]["loaded_runtime_count"] == 1
    assert payload["summary"]["system_ram_free_gib"] == 8.0
    assert payload["runtimes"][0]["alias_id"] == "demo/alias"
    assert payload["runtimes"][0]["pinned"] is True
    assert payload["runtimes"][0]["estimated_vram_gib"] == 1.0


def test_benchmark_summary_payload_prefers_verified_records() -> None:
    payload = _benchmark_summary_payload(
        "demo/alias",
        [
            {
                "backend": "vulkan",
                "placement": "igpu_only",
                "prompt_tps": 100.0,
                "generation_tps": 20.0,
                "collected_at": "2026-01-01T00:00:00+00:00",
                "metadata": {"verified": False},
            },
            {
                "backend": "cuda",
                "placement": "cpu_dgpu_hybrid",
                "prompt_tps": 500.0,
                "generation_tps": 80.0,
                "collected_at": "2026-01-02T00:00:00+00:00",
                "metadata": {"verified": True},
            },
            {
                "backend": "cuda",
                "placement": "cpu_dgpu_hybrid",
                "prompt_tps": 400.0,
                "generation_tps": 70.0,
                "collected_at": "2026-01-03T00:00:00+00:00",
                "metadata": {"verified": True},
            },
        ],
    )

    assert payload["alias_id"] == "demo/alias"
    assert payload["total_records"] == 3
    assert payload["unverified_records"] == 1
    assert payload["best_verified_by_placement"][0]["combined_tps"] == 580.0


def test_runtime_diagnostics_handles_empty_runtime_list() -> None:
    settings = AppSettings.load()

    payload = _runtime_diagnostics_payload(
        settings,
        {
            "system_ram_free_bytes": 4 * 1024**3,
            "system_ram_total_bytes": 16 * 1024**3,
            "backends_available": ["cpu"],
        },
        [],
    )

    assert payload["summary"]["loaded_runtime_count"] == 0
    assert payload["runtimes"] == []
