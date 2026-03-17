"""Hardware discovery and backend availability probes."""

from __future__ import annotations

import csv
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
                    id=f"nvidia-{device_id}",
                    name=name,
                    kind="dgpu",
                    backend_candidates=[Backend.CUDA, Backend.VULKAN],
                    total_memory_bytes=total_mib * 1024 * 1024,
                    free_memory_bytes=free_mib * 1024 * 1024,
                    driver=driver,
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
                    id=f"video-{index}",
                    name=name,
                    kind=kind,
                    backend_candidates=candidates,
                    total_memory_bytes=int(memory) if isinstance(memory, int) and memory > 0 else None,
                    experimental=experimental,
                )
            )
        return devices
