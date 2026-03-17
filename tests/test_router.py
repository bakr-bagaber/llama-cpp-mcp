from __future__ import annotations

from llama_orchestrator.models import (
    AliasDefinition,
    Backend,
    BackendPreference,
    BenchmarkRecord,
    HardwareDevice,
    HardwareInventory,
    LoadProfile,
    PlacementKind,
)
from llama_orchestrator.router import RouteContext, Router
from llama_orchestrator.settings import AppSettings


def make_alias() -> AliasDefinition:
    return AliasDefinition(
        id="alias",
        base_model_id="base-model",
        load_profile_id="balanced",
        preset_id="default",
        backend_preference=BackendPreference.AUTO,
    )


def make_profile() -> LoadProfile:
    return LoadProfile(id="balanced", backend_preference=BackendPreference.AUTO)


def make_inventory(include_dgpu: bool = False, include_igpu: bool = False) -> HardwareInventory:
    devices = []
    backends = [Backend.CPU]
    if include_dgpu:
        devices.append(
            HardwareDevice(
                id="dgpu-0",
                name="RTX",
                kind="dgpu",
                backend_candidates=[Backend.CUDA],
                total_memory_bytes=8 * 1024**3,
                free_memory_bytes=8 * 1024**3,
            )
        )
        backends.append(Backend.CUDA)
    if include_igpu:
        devices.append(
            HardwareDevice(
                id="igpu-0",
                name="Intel Iris",
                kind="igpu",
                backend_candidates=[Backend.VULKAN],
                total_memory_bytes=4 * 1024**3,
                free_memory_bytes=4 * 1024**3,
                experimental=True,
            )
        )
        backends.append(Backend.VULKAN)
    return HardwareInventory(
        cpu_count=8,
        system_ram_total_bytes=32 * 1024**3,
        system_ram_free_bytes=24 * 1024**3,
        backends_available=backends,
        devices=devices,
        warnings=[],
    )


def test_router_prefers_cpu_when_gpu_reserve_would_be_exceeded() -> None:
    settings = AppSettings.load()
    settings.policy.min_free_dgpu_vram_bytes = 2 * 1024**3
    router = Router(settings)
    inventory = make_inventory(include_dgpu=True)
    alias = make_alias()
    profile = make_profile()

    decision = router.choose_placement(
        alias=alias,
        profile=profile,
        model_ram_bytes=4 * 1024**3,
        model_vram_bytes=32 * 1024**3,
        context=RouteContext(
            inventory=inventory,
            warm_runtimes=[],
            benchmarks=[],
            requested_backend_preference=BackendPreference.AUTO,
        ),
    )

    assert decision.selected is not None
    assert decision.selected.placement == PlacementKind.CPU_ONLY


def test_router_hides_igpu_candidates_until_experimental_flag_is_enabled() -> None:
    settings = AppSettings.load()
    settings.policy.allow_experimental_igpu = False
    router = Router(settings)
    alias = make_alias()
    profile = make_profile()
    inventory = make_inventory(include_igpu=True)

    decision = router.choose_placement(
        alias=alias,
        profile=profile,
        model_ram_bytes=2 * 1024**3,
        model_vram_bytes=1 * 1024**3,
        context=RouteContext(
            inventory=inventory,
            warm_runtimes=[],
            benchmarks=[],
            requested_backend_preference=BackendPreference.AUTO,
        ),
    )

    assert all(candidate.placement != PlacementKind.IGPU_ONLY for candidate in decision.candidates)


def test_router_exposes_igpu_candidates_when_flag_enabled() -> None:
    settings = AppSettings.load()
    settings.policy.allow_experimental_igpu = True
    router = Router(settings)
    alias = make_alias()
    profile = make_profile()
    inventory = make_inventory(include_igpu=True)

    decision = router.choose_placement(
        alias=alias,
        profile=profile,
        model_ram_bytes=2 * 1024**3,
        model_vram_bytes=1 * 1024**3,
        context=RouteContext(
            inventory=inventory,
            warm_runtimes=[],
            benchmarks=[],
            requested_backend_preference=BackendPreference.AUTO,
        ),
    )

    assert any(candidate.placement == PlacementKind.IGPU_ONLY for candidate in decision.candidates)


def test_router_uses_benchmark_bonus_when_available() -> None:
    settings = AppSettings.load()
    router = Router(settings)
    alias = make_alias()
    profile = make_profile()
    inventory = make_inventory(include_dgpu=True)

    decision = router.choose_placement(
        alias=alias,
        profile=profile,
        model_ram_bytes=2 * 1024**3,
        model_vram_bytes=2 * 1024**3,
        context=RouteContext(
            inventory=inventory,
            warm_runtimes=[],
            benchmarks=[
                BenchmarkRecord(
                    alias_id="alias",
                    backend=Backend.CUDA,
                    placement=PlacementKind.DGPU_ONLY,
                    prompt_tps=500.0,
                    generation_tps=500.0,
                )
            ],
            requested_backend_preference=BackendPreference.AUTO,
        ),
    )

    assert decision.selected is not None
    assert decision.selected.placement == PlacementKind.DGPU_ONLY
