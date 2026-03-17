from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from llama_orchestrator.catalog import CatalogStore
from llama_orchestrator.models import (
    AliasDefinition,
    BaseModelDefinition,
    Backend,
    GenerationPreset,
    HardwareInventory,
    LoadProfile,
    PlacementKind,
    RuntimeRecord,
    RuntimeStatus,
    SupportLevel,
)
from llama_orchestrator.router import Router
from llama_orchestrator.runtime import RuntimeManager
from llama_orchestrator.settings import AppSettings
from llama_orchestrator.state import StateStore


def _build_manager(sandbox_path: Path) -> tuple[RuntimeManager, CatalogStore]:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "orchestrator.db",
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
    manager = RuntimeManager(settings, catalog, state, Router(settings))
    return manager, catalog

@pytest.mark.anyio
async def test_runtime_manager_reuses_warm_runtime(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _catalog = _build_manager(sandbox_path)
    inventory = HardwareInventory(system_ram_total_bytes=32 * 1024**3, system_ram_free_bytes=24 * 1024**3, backends_available=[Backend.CPU])

    async def fake_launch(alias, model, profile, preset, selected, inventory):
        return RuntimeRecord(
            runtime_key="demo/alias:cpu:cpu_only",
            alias_id=alias.id,
            model_id=alias.base_model_id,
            profile_id=alias.load_profile_id,
            backend=Backend.CPU,
            placement=PlacementKind.CPU_ONLY,
            endpoint_url="http://127.0.0.1:1234",
            support_level=SupportLevel.STABLE,
            status=RuntimeStatus.READY,
        )

    monkeypatch.setattr(manager, "_launch_runtime", fake_launch)

    first = await manager.ensure_runtime("demo/alias", inventory=inventory)
    second = await manager.ensure_runtime("demo/alias", inventory=inventory)

    assert first.runtime_key == second.runtime_key
    assert len(manager.list_runtimes()) == 1


@pytest.mark.anyio
async def test_runtime_manager_unloads_idle_runtime(sandbox_path: Path) -> None:
    manager, _catalog = _build_manager(sandbox_path)
    runtime = RuntimeRecord(
        runtime_key="demo/alias:cpu:cpu_only",
        alias_id="demo/alias",
        model_id="demo-model",
        profile_id="balanced",
        backend=Backend.CPU,
        placement=PlacementKind.CPU_ONLY,
        endpoint_url="http://127.0.0.1:1234",
        support_level=SupportLevel.STABLE,
        status=RuntimeStatus.READY,
        last_used_at=manager._now() - timedelta(seconds=10_000),
    )
    manager._runtimes[runtime.runtime_key] = runtime

    unloaded = await manager.unload_idle()

    assert runtime.runtime_key in unloaded
    assert manager.list_runtimes() == []


def test_runtime_manager_builds_vulkan_device_args() -> None:
    inventory = HardwareInventory(
        devices=[
            # The experimental iGPU path uses Vulkan selectors discovered
            # from llama-bench's device list output.
            # We keep the test tiny so the launch rule stays easy to follow.
            {
                "id": "igpu0",
                "ordinal": 0,
                "name": "Intel Arc",
                "kind": "igpu",
                "selectors": {"vulkan": "vulkan0"},
                "backend_candidates": [Backend.VULKAN],
                "metadata": {"vulkan_runtime_selector": "Vulkan0", "vulkan_main_gpu_index": 0},
            }
        ]
    )

    args = RuntimeManager._device_args_for_selection(Backend.VULKAN, ["igpu0"], inventory)

    assert args == ["--device", "Vulkan0", "--main-gpu", "0"]
