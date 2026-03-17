"""Routing decisions for model placement."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    AliasDefinition,
    Backend,
    BackendPreference,
    BenchmarkRecord,
    CandidatePlacement,
    HardwareInventory,
    LoadProfile,
    PlacementEstimate,
    PlacementKind,
    RoutingDecision,
    RuntimeRecord,
    SupportLevel,
)
from .settings import AppSettings


@dataclass(slots=True)
class RouteContext:
    inventory: HardwareInventory
    warm_runtimes: list[RuntimeRecord]
    benchmarks: list[BenchmarkRecord]
    requested_backend_preference: BackendPreference


class Router:
    """Selects the most appropriate placement for a request."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def choose_placement(
        self,
        alias: AliasDefinition,
        profile: LoadProfile,
        model_ram_bytes: int,
        model_vram_bytes: int,
        context: RouteContext,
    ) -> RoutingDecision:
        candidates = self._build_candidates(alias, profile, model_ram_bytes, model_vram_bytes, context)
        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = next((item for item in candidates if item.feasible), None)
        reused_runtime_key = self._find_reusable_runtime(alias.id, selected, context.warm_runtimes)
        if reused_runtime_key and selected:
            selected.score += 100.0
        reason = selected.reason if selected else "No feasible placement matched the current resource and policy constraints."
        return RoutingDecision(
            alias_id=alias.id,
            selected=selected,
            candidates=candidates,
            reason_summary=reason,
            reused_runtime_key=reused_runtime_key,
        )

    def _build_candidates(
        self,
        alias: AliasDefinition,
        profile: LoadProfile,
        model_ram_bytes: int,
        model_vram_bytes: int,
        context: RouteContext,
    ) -> list[CandidatePlacement]:
        inventory = context.inventory
        policy = self.settings.policy
        requested = alias.backend_preference or context.requested_backend_preference
        candidates: list[CandidatePlacement] = []
        candidates.append(
            self._candidate(
                backend=Backend.CPU,
                placement=PlacementKind.CPU_ONLY,
                support_level=SupportLevel.STABLE,
                devices=[],
                estimate=PlacementEstimate(ram_bytes=model_ram_bytes, vram_bytes=0),
                inventory=inventory,
                requested=requested,
            )
        )
        dgpus = inventory.devices_by_kind("dgpu")
        if dgpus:
            if Backend.CUDA in inventory.backends_available:
                candidates.append(
                    self._candidate(
                        backend=Backend.CUDA,
                        placement=PlacementKind.DGPU_ONLY,
                        support_level=SupportLevel.STABLE,
                        devices=[device.id for device in dgpus],
                        estimate=PlacementEstimate(ram_bytes=max(model_ram_bytes // 2, 0), vram_bytes=model_vram_bytes),
                        inventory=inventory,
                        requested=requested,
                    )
                )
                candidates.append(
                    self._candidate(
                        backend=Backend.CUDA,
                        placement=PlacementKind.CPU_DGPU_HYBRID,
                        support_level=SupportLevel.STABLE,
                        devices=[device.id for device in dgpus],
                        estimate=PlacementEstimate(ram_bytes=model_ram_bytes, vram_bytes=max(model_vram_bytes // 2, 0)),
                        inventory=inventory,
                        requested=requested,
                    )
                )
            elif Backend.VULKAN in inventory.backends_available:
                candidates.append(
                    self._candidate(
                        backend=Backend.VULKAN,
                        placement=PlacementKind.DGPU_ONLY,
                        support_level=SupportLevel.STABLE,
                        devices=[device.id for device in dgpus],
                        estimate=PlacementEstimate(ram_bytes=max(model_ram_bytes // 2, 0), vram_bytes=model_vram_bytes),
                        inventory=inventory,
                        requested=requested,
                    )
                )

        igpus = inventory.devices_by_kind("igpu")
        if igpus and (Backend.VULKAN in inventory.backends_available or Backend.SYCL in inventory.backends_available):
            backend = Backend.SYCL if Backend.SYCL in inventory.backends_available else Backend.VULKAN
            if policy.allow_experimental_igpu:
                candidates.append(
                    self._candidate(
                        backend=backend,
                        placement=PlacementKind.IGPU_ONLY,
                        support_level=SupportLevel.EXPERIMENTAL,
                        devices=[device.id for device in igpus],
                        estimate=PlacementEstimate(ram_bytes=model_ram_bytes, vram_bytes=max(model_vram_bytes // 2, 0)),
                        inventory=inventory,
                        requested=requested,
                    )
                )
                candidates.append(
                    self._candidate(
                        backend=backend,
                        placement=PlacementKind.CPU_IGPU_HYBRID,
                        support_level=SupportLevel.EXPERIMENTAL,
                        devices=[device.id for device in igpus],
                        estimate=PlacementEstimate(ram_bytes=model_ram_bytes, vram_bytes=max(model_vram_bytes // 3, 0)),
                        inventory=inventory,
                        requested=requested,
                    )
                )
        if dgpus and igpus and policy.allow_experimental_mixed_gpu and Backend.VULKAN in inventory.backends_available:
            candidates.append(
                self._candidate(
                    backend=Backend.VULKAN,
                    placement=PlacementKind.DGPU_IGPU_MIXED,
                    support_level=SupportLevel.EXPERIMENTAL,
                    devices=[device.id for device in dgpus + igpus],
                    estimate=PlacementEstimate(ram_bytes=max(model_ram_bytes // 2, 0), vram_bytes=max(model_vram_bytes // 2, 0)),
                    inventory=inventory,
                    requested=requested,
                )
            )

        for candidate in candidates:
            candidate.score += self._benchmark_bonus(alias.id, candidate.backend, candidate.placement, context.benchmarks)
            if candidate.support_level is SupportLevel.STABLE:
                candidate.score += 20.0
            if candidate.support_level is SupportLevel.EXPERIMENTAL:
                candidate.score -= 10.0
        return candidates

    def _candidate(
        self,
        *,
        backend: Backend,
        placement: PlacementKind,
        support_level: SupportLevel,
        devices: list[str],
        estimate: PlacementEstimate,
        inventory: HardwareInventory,
        requested: BackendPreference,
    ) -> CandidatePlacement:
        feasible = True
        reasons: list[str] = []
        score = 0.0
        policy = self.settings.policy

        if placement in {PlacementKind.CPU_ONLY, PlacementKind.CPU_DGPU_HYBRID, PlacementKind.CPU_IGPU_HYBRID}:
            free_after = inventory.system_ram_free_bytes - estimate.ram_bytes
            if free_after < policy.min_free_system_ram_bytes:
                feasible = False
                reasons.append("System RAM reserve would be exceeded.")
            else:
                score += max(0.0, free_after / (1024**3))

        if placement in {PlacementKind.DGPU_ONLY, PlacementKind.CPU_DGPU_HYBRID, PlacementKind.DGPU_IGPU_MIXED}:
            dgpu = max(((device.free_memory_bytes or 0) for device in inventory.devices_by_kind("dgpu")), default=0)
            if dgpu - estimate.vram_bytes < policy.min_free_dgpu_vram_bytes:
                feasible = False
                reasons.append("Discrete GPU VRAM reserve would be exceeded.")
            else:
                score += max(0.0, (dgpu - estimate.vram_bytes) / (1024**3))

        if placement in {PlacementKind.IGPU_ONLY, PlacementKind.CPU_IGPU_HYBRID, PlacementKind.DGPU_IGPU_MIXED}:
            free_after = inventory.system_ram_free_bytes - estimate.ram_bytes
            if free_after < policy.min_free_igpu_shared_ram_bytes:
                feasible = False
                reasons.append("Integrated GPU shared RAM reserve would be exceeded.")
            else:
                score += max(0.0, free_after / (1024**3))

        score += self._preference_bonus(requested, placement)
        reason = "; ".join(reasons) if reasons else f"Placement {placement.value} is feasible."
        return CandidatePlacement(
            backend=backend,
            placement=placement,
            support_level=support_level,
            devices=devices,
            feasible=feasible,
            reason=reason,
            estimated=estimate,
            score=score,
        )

    @staticmethod
    def _preference_bonus(preference: BackendPreference, placement: PlacementKind) -> float:
        if preference is BackendPreference.AUTO:
            return 0.0
        mapping = {
            BackendPreference.PREFER_CPU: {PlacementKind.CPU_ONLY},
            BackendPreference.PREFER_DGPU: {PlacementKind.DGPU_ONLY, PlacementKind.CPU_DGPU_HYBRID},
            BackendPreference.PREFER_IGPU: {PlacementKind.IGPU_ONLY, PlacementKind.CPU_IGPU_HYBRID, PlacementKind.DGPU_IGPU_MIXED},
            BackendPreference.FORCE_CPU: {PlacementKind.CPU_ONLY},
            BackendPreference.FORCE_DGPU: {PlacementKind.DGPU_ONLY, PlacementKind.CPU_DGPU_HYBRID},
            BackendPreference.FORCE_IGPU: {PlacementKind.IGPU_ONLY, PlacementKind.CPU_IGPU_HYBRID, PlacementKind.DGPU_IGPU_MIXED},
        }
        preferred = mapping.get(preference, set())
        if preferred and placement not in preferred and preference.name.startswith("FORCE"):
            return -10_000.0
        return 25.0 if placement in preferred else 0.0

    @staticmethod
    def _benchmark_bonus(
        alias_id: str,
        backend: Backend,
        placement: PlacementKind,
        benchmarks: list[BenchmarkRecord],
    ) -> float:
        for benchmark in benchmarks:
            if benchmark.alias_id == alias_id and benchmark.backend == backend and benchmark.placement == placement:
                return benchmark.generation_tps + benchmark.prompt_tps
        return 0.0

    @staticmethod
    def _find_reusable_runtime(
        alias_id: str,
        selected: CandidatePlacement | None,
        warm_runtimes: list[RuntimeRecord],
    ) -> str | None:
        if not selected:
            return None
        for runtime in warm_runtimes:
            if runtime.alias_id == alias_id and runtime.backend == selected.backend and runtime.placement == selected.placement:
                return runtime.runtime_key
        return None
