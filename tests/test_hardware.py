from __future__ import annotations

from llama_orchestrator.hardware import HardwareProbe
from llama_orchestrator.models import Backend, HardwareDevice, HardwareInventory


def test_normalize_vulkan_name_strips_memory_suffix() -> None:
    name = "Intel(R) Arc(TM) Graphics (18361 MiB, 17593 MiB free)"

    normalized = HardwareProbe._normalize_vulkan_name(name)

    assert normalized == "Intel(R) Arc(TM) Graphics"


def test_inventory_finds_device_by_generic_id_and_backend_selector() -> None:
    inventory = HardwareInventory(
        devices=[
            HardwareDevice(
                id="dgpu0",
                name="NVIDIA RTX",
                kind="dgpu",
                ordinal=0,
                selectors={"cuda": "cuda0", "vulkan": "vulkan1"},
                backend_candidates=[Backend.CUDA, Backend.VULKAN],
            )
        ]
    )

    assert inventory.find_device("dgpu0") is not None
    assert inventory.find_device("cuda0", backend=Backend.CUDA) is not None
    assert inventory.find_device("vulkan1", backend=Backend.VULKAN) is not None
