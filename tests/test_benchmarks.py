from __future__ import annotations

import json
from pathlib import Path

from llama_orchestrator.benchmarks import BenchmarkService
from llama_orchestrator.catalog import CatalogStore
from llama_orchestrator.models import (
    AliasDefinition,
    BaseModelDefinition,
    Backend,
    GenerationPreset,
    HardwareDevice,
    HardwareInventory,
    LoadProfile,
    PlacementKind,
)
from llama_orchestrator.settings import AppSettings
from llama_orchestrator.state import StateStore


def test_run_llama_bench_parses_prompt_and_generation_rows(sandbox_path: Path, monkeypatch) -> None:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "orchestrator.db",
        cpu_bench_executable="fake-bench",
    )
    settings.ensure_directories()
    catalog = CatalogStore(settings.catalog_path)
    catalog.load()
    model_path = sandbox_path / "demo.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    catalog.upsert_model(BaseModelDefinition(id="demo-model", display_name="Demo", local_path=model_path))
    catalog.upsert_profile(LoadProfile(id="balanced"))
    catalog.upsert_preset(GenerationPreset(id="default"))
    catalog.upsert_alias(
        AliasDefinition(
            id="demo/alias",
            base_model_id="demo-model",
            load_profile_id="balanced",
            preset_id="default",
        )
    )
    state = StateStore(settings.state_path)
    service = BenchmarkService(settings, catalog, state)

    class Result:
        stdout = json.dumps(
            [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 123.4},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 56.7},
            ]
        )

    def fake_run(command, capture_output, text, check, timeout):
        assert "--output" in command
        return Result()

    monkeypatch.setattr("llama_orchestrator.benchmarks.subprocess.run", fake_run)

    record = service.run_llama_bench(alias_id="demo/alias", backend=Backend.CPU)

    assert record.prompt_tps == 123.4
    assert record.generation_tps == 56.7
    assert record.metadata["source"] == "llama-bench"


def test_run_llama_bench_uses_vulkan_device_selectors(sandbox_path: Path, monkeypatch) -> None:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "orchestrator.db",
        vulkan_bench_executable="fake-vulkan-bench",
    )
    settings.ensure_directories()
    catalog = CatalogStore(settings.catalog_path)
    catalog.load()
    model_path = sandbox_path / "demo.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    catalog.upsert_model(BaseModelDefinition(id="demo-model", display_name="Demo", local_path=model_path))
    catalog.upsert_profile(LoadProfile(id="balanced", gpu_layers=99))
    catalog.upsert_preset(GenerationPreset(id="default"))
    catalog.upsert_alias(
        AliasDefinition(
            id="demo/alias",
            base_model_id="demo-model",
            load_profile_id="balanced",
            preset_id="default",
        )
    )
    state = StateStore(settings.state_path)
    service = BenchmarkService(settings, catalog, state)
    inventory = HardwareInventory(
        devices=[
            HardwareDevice(
                id="igpu0",
                name="Intel Arc",
                kind="igpu",
                ordinal=0,
                selectors={"vulkan": "vulkan0"},
                backend_candidates=[Backend.VULKAN],
                metadata={"vulkan_runtime_selector": "Vulkan0", "vulkan_main_gpu_index": 0},
            )
        ]
    )

    class Result:
        stdout = json.dumps(
            [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 11.1},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 22.2},
            ]
        )

    captured: list[str] = []

    def fake_run(command, capture_output, text, check, timeout):
        captured.extend(command)
        return Result()

    monkeypatch.setattr("llama_orchestrator.benchmarks.subprocess.run", fake_run)

    record = service.run_llama_bench(
        alias_id="demo/alias",
        backend=Backend.VULKAN,
        placement=PlacementKind.IGPU_ONLY,
        inventory=inventory,
        device_ids=["igpu0"],
    )

    assert "--device" in captured
    assert "Vulkan0" in captured
    assert "--main-gpu" in captured
    assert record.placement == PlacementKind.IGPU_ONLY
    assert record.metadata["selectors"] == ["Vulkan0"]
