"""Helpers for recording and querying benchmark data."""

from __future__ import annotations

import json
import subprocess

from .catalog import CatalogStore
from .models import Backend, BenchmarkRecord, HardwareInventory, PlacementKind
from .settings import AppSettings
from .state import StateStore


class BenchmarkService:
    """Small service layer around the state store.

    Keeping benchmark logic in one place makes it easier to swap manual
    records for external `llama-bench` integration later.
    """

    def __init__(self, settings: AppSettings, catalog: CatalogStore, state: StateStore) -> None:
        self.settings = settings
        self.catalog = catalog
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

    def run_llama_bench(
        self,
        *,
        alias_id: str,
        backend: Backend,
        n_gpu_layers: int | None = None,
        placement: PlacementKind | None = None,
        inventory: HardwareInventory | None = None,
        device_ids: list[str] | None = None,
    ) -> BenchmarkRecord:
        """Run a real llama-bench process and store the parsed result."""
        executable = self.settings.bench_executable_for_backend(backend)
        if not executable:
            raise RuntimeError(f"No llama-bench executable is available for backend '{backend.value}'.")

        _alias, model, profile, _preset = self.catalog.resolve_alias(alias_id)
        if not model.local_path:
            raise RuntimeError(f"Alias '{alias_id}' does not point to a local model file.")

        command = [
            executable,
            "--model",
            str(model.local_path),
            "--output",
            "json",
        ]
        if n_gpu_layers is not None:
            command.extend(["--n-gpu-layers", str(n_gpu_layers)])
        elif backend is not Backend.CPU and profile.gpu_layers is not None:
            command.extend(["--n-gpu-layers", str(profile.gpu_layers)])

        selectors, main_gpu_index = self._resolve_vulkan_selectors(backend, inventory, device_ids or [])
        if selectors:
            command.extend(["--device", ",".join(selectors)])
            if main_gpu_index is not None:
                command.extend(["--main-gpu", str(main_gpu_index)])

        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
        payload = json.loads(result.stdout)
        entries = payload if isinstance(payload, list) else [payload]
        prompt_entry = next((item for item in entries if int(item.get("n_prompt", 0)) > 0), {})
        generation_entry = next((item for item in entries if int(item.get("n_gen", 0)) > 0), {})

        resolved_placement = placement or self._infer_placement(
            backend=backend,
            profile_gpu_layers=n_gpu_layers if n_gpu_layers is not None else profile.gpu_layers,
            inventory=inventory,
            device_ids=device_ids or [],
        )

        prompt_tps = float(prompt_entry.get("avg_ts") or 0.0)
        generation_tps = float(generation_entry.get("avg_ts") or 0.0)
        load_seconds = 0.0

        record = BenchmarkRecord(
            alias_id=alias_id,
            backend=backend,
            placement=resolved_placement,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            load_seconds=load_seconds,
            metadata={
                "source": "llama-bench",
                "raw": entries,
                "device_ids": device_ids or [],
                "selectors": selectors,
            },
        )
        self.state.add_benchmark(record)
        return record

    @staticmethod
    def _resolve_vulkan_selectors(
        backend: Backend,
        inventory: HardwareInventory | None,
        device_ids: list[str],
    ) -> tuple[list[str], int | None]:
        if backend is not Backend.VULKAN or inventory is None or not device_ids:
            return [], None
        selectors: list[str] = []
        main_gpu_index: int | None = None
        for device_id in device_ids:
            device = inventory.find_device(device_id, backend=backend)
            if not device:
                continue
            selector = BenchmarkService._runtime_selector_for_backend(device, backend)
            if not selector:
                continue
            selectors.append(selector)
            if main_gpu_index is None:
                raw_index = device.metadata.get("vulkan_main_gpu_index")
                if isinstance(raw_index, int):
                    main_gpu_index = raw_index
        return selectors, main_gpu_index

    @staticmethod
    def _infer_placement(
        *,
        backend: Backend,
        profile_gpu_layers: int | None,
        inventory: HardwareInventory | None,
        device_ids: list[str],
    ) -> PlacementKind:
        if backend is Backend.CPU:
            return PlacementKind.CPU_ONLY
        if backend is Backend.CUDA:
            return PlacementKind.CPU_DGPU_HYBRID if (profile_gpu_layers or 0) > 0 else PlacementKind.DGPU_ONLY
        if backend is not Backend.VULKAN or inventory is None or not device_ids:
            return PlacementKind.DGPU_ONLY

        selected_devices = [inventory.find_device(device_id, backend=backend) for device_id in device_ids]
        selected_devices = [device for device in selected_devices if device is not None]
        kinds = {device.kind for device in selected_devices}
        if kinds == {"igpu"}:
            return PlacementKind.IGPU_ONLY
        if kinds == {"dgpu"}:
            return PlacementKind.DGPU_ONLY
        if "igpu" in kinds and "dgpu" in kinds:
            return PlacementKind.DGPU_IGPU_MIXED
        return PlacementKind.DGPU_ONLY

    @staticmethod
    def _runtime_selector_for_backend(device, backend: Backend) -> str:
        selector = str(device.selectors.get(backend.value, "")).strip()
        if not selector:
            return ""
        suffix = selector[len(backend.value) :]
        if backend is Backend.VULKAN:
            return f"Vulkan{suffix}"
        return selector
