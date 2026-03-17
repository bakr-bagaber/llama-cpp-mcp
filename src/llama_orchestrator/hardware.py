"""Hardware discovery and backend availability probes."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path

import psutil

from .models import Backend, HardwareDevice, HardwareInventory
from .settings import AppSettings


class HardwareProbe:
    """Collects system and backend availability information."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def collect(self) -> HardwareInventory:
        """Collect a fresh inventory snapshot.

        We keep this as a pure snapshot method so routing can make decisions
        from current machine state instead of stale cached data.
        """
        virtual_memory = psutil.virtual_memory()
        inventory = HardwareInventory(
            cpu_count=psutil.cpu_count(logical=True) or 1,
            system_ram_total_bytes=int(virtual_memory.total),
            system_ram_free_bytes=int(virtual_memory.available),
        )
        inventory.devices.extend(self._probe_nvidia())
        inventory.devices.extend(self._probe_windows_video_controllers(existing_names={device.name for device in inventory.devices}))
        self._attach_vulkan_metadata(inventory)
        self._assign_generic_ids(inventory)
        inventory.backends_available = self._detect_backends(inventory)
        if not inventory.devices:
            inventory.warnings.append("No GPU devices were detected. CPU-only routing remains available.")
        return inventory

    def _detect_backends(self, inventory: HardwareInventory) -> list[Backend]:
        available: list[Backend] = [Backend.CPU]
        for backend in (Backend.CUDA, Backend.VULKAN, Backend.SYCL):
            executable = self.settings.executable_for_backend(backend)
            if executable and self._which(executable):
                available.append(backend)
                continue
            # CUDA gets a slightly friendlier fallback because an NVIDIA
            # machine often already has a usable local llama.cpp build.
            if backend is Backend.CUDA and inventory.devices_by_kind("dgpu") and self._which("nvidia-smi"):
                available.append(backend)
        return available

    def _which(self, command: str) -> str | None:
        direct = Path(command)
        if direct.exists():
            return str(direct)
        return shutil.which(command)

    def _probe_nvidia(self) -> list[HardwareDevice]:
        if not self._which("nvidia-smi"):
            return []
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
        except Exception:
            return []
        devices: list[HardwareDevice] = []
        reader = csv.reader(result.stdout.splitlines())
        for row in reader:
            if len(row) < 4:
                continue
            device_id = row[0].strip()
            name = row[1].strip()
            total_mib = int(row[2].strip())
            free_mib = int(row[3].strip())
            driver = row[4].strip() if len(row) > 4 else None
            devices.append(
                HardwareDevice(
                    id="",
                    name=name,
                    kind="dgpu",
                    ordinal=int(device_id),
                    selectors={Backend.CUDA.value: f"cuda{device_id}"},
                    backend_candidates=[Backend.CUDA, Backend.VULKAN],
                    total_memory_bytes=total_mib * 1024 * 1024,
                    free_memory_bytes=free_mib * 1024 * 1024,
                    driver=driver,
                    metadata={"probe_source": "nvidia-smi"},
                )
            )
        return devices

    def _probe_windows_video_controllers(self, existing_names: set[str]) -> list[HardwareDevice]:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
        except Exception:
            return []
        payload = result.stdout.strip()
        if not payload:
            return []
        try:
            import json

            items = json.loads(payload)
        except Exception:
            return []
        if isinstance(items, dict):
            items = [items]
        devices: list[HardwareDevice] = []
        for index, item in enumerate(items):
            name = str(item.get("Name", "")).strip()
            if not name or name in existing_names:
                continue
            lower_name = name.lower()
            if "intel" in lower_name or "iris" in lower_name or "uhd" in lower_name:
                kind = "igpu"
                candidates = [Backend.VULKAN, Backend.SYCL]
                experimental = True
            elif "nvidia" in lower_name:
                kind = "dgpu"
                candidates = [Backend.CUDA, Backend.VULKAN]
                experimental = False
            else:
                kind = "igpu" if "integrated" in lower_name else "dgpu"
                candidates = [Backend.VULKAN, Backend.SYCL]
                experimental = kind == "igpu"
            memory = item.get("AdapterRAM")
            devices.append(
                HardwareDevice(
                    id="",
                    name=name,
                    kind=kind,
                    ordinal=index,
                    backend_candidates=candidates,
                    total_memory_bytes=int(memory) if isinstance(memory, int) and memory > 0 else None,
                    experimental=experimental,
                    metadata={"probe_source": "windows-video-controller"},
                )
            )
        return devices

    def _attach_vulkan_metadata(self, inventory: HardwareInventory) -> None:
        """Annotate inventory devices with Vulkan selectors when available.

        llama.cpp's Vulkan binaries expose selectors like `Vulkan0` that we
        need later for targeted iGPU launches. We store those selectors in
        metadata so the router and runtime manager can stay backend-agnostic.
        """
        for selector, name in self._probe_vulkan_devices():
            matched = self._match_device_by_name(inventory, name)
            if matched is not None:
                if Backend.VULKAN not in matched.backend_candidates:
                    matched.backend_candidates.append(Backend.VULKAN)
                matched.selectors[Backend.VULKAN.value] = self._canonical_selector(Backend.VULKAN, selector)
                matched.metadata["vulkan_runtime_selector"] = selector
                matched.metadata["vulkan_main_gpu_index"] = self._selector_index(selector)
                continue

            inferred_kind = self._infer_device_kind(name)
            inventory.devices.append(
                HardwareDevice(
                    id="",
                    name=name,
                    kind=inferred_kind,
                    selectors={Backend.VULKAN.value: self._canonical_selector(Backend.VULKAN, selector)},
                    backend_candidates=[Backend.VULKAN],
                    experimental=inferred_kind == "igpu",
                    metadata={
                        "probe_source": "vulkan-device-list",
                        "vulkan_runtime_selector": selector,
                        "vulkan_main_gpu_index": self._selector_index(selector),
                    },
                )
            )

    def _probe_vulkan_devices(self) -> list[tuple[str, str]]:
        executable = self.settings.bench_executable_for_backend(Backend.VULKAN)
        if not executable or not self._which(executable):
            return []
        try:
            result = subprocess.run([executable, "--list-devices"], capture_output=True, text=True, check=True, timeout=10)
        except Exception:
            return []
        devices: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            match = re.match(r"^(Vulkan\d+):\s+(.+)$", line.strip())
            if match:
                devices.append((match.group(1), self._normalize_vulkan_name(match.group(2))))
        return devices

    @staticmethod
    def _match_device_by_name(inventory: HardwareInventory, name: str) -> HardwareDevice | None:
        lower_name = name.lower()
        for device in inventory.devices:
            if device.name.lower() == lower_name:
                return device
        for device in inventory.devices:
            candidate = device.name.lower()
            if candidate in lower_name or lower_name in candidate:
                return device
        return None

    @staticmethod
    def _infer_device_kind(name: str) -> str:
        lower_name = name.lower()
        if "intel" in lower_name or "iris" in lower_name or "uhd" in lower_name or "arc" in lower_name:
            return "igpu"
        return "dgpu"

    @staticmethod
    def _selector_index(selector: str) -> int | None:
        suffix = selector.removeprefix("Vulkan")
        return int(suffix) if suffix.isdigit() else None

    @staticmethod
    def _normalize_vulkan_name(name: str) -> str:
        return re.sub(r"\s+\(\d+\s+MiB,\s+\d+\s+MiB free\)$", "", name).strip()

    @staticmethod
    def _canonical_selector(backend: Backend, runtime_selector: str) -> str:
        prefix = backend.value
        suffix = "".join(character for character in runtime_selector if character.isdigit())
        return f"{prefix}{suffix}" if suffix else backend.value

    @staticmethod
    def _assign_generic_ids(inventory: HardwareInventory) -> None:
        """Assign stable generic ids like `dgpu0` and `igpu0`.

        Backend-specific names such as `cuda0` and `vulkan1` remain available
        through the `selectors` field, while `id` stays backend-agnostic.
        """
        kind_counters: dict[str, int] = {}
        for device in inventory.devices:
            if device.ordinal is None:
                device.ordinal = kind_counters.get(device.kind, 0)
            kind_counters[device.kind] = max(kind_counters.get(device.kind, 0), device.ordinal + 1)
        for device in sorted(inventory.devices, key=lambda item: (item.kind, item.ordinal or 0, item.name.lower())):
            if not device.id:
                device.id = f"{device.kind}{device.ordinal or 0}"
