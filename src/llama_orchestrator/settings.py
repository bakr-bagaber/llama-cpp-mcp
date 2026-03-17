"""Application settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from .models import Backend, MemoryPolicy, StrictModel


class AppSettings(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key: str | None = None
    catalog_path: Path = Path("catalog/catalog.yaml")
    state_path: Path = Path("state/orchestrator.db")
    idle_scan_interval_seconds: int = Field(default=30, ge=1)
    runtime_start_timeout_seconds: int = Field(default=20, ge=1)
    http_timeout_seconds: float = Field(default=120.0, gt=0)
    default_idle_unload_seconds: int = Field(default=900, ge=1)
    cpu_executable: str | None = None
    cuda_executable: str | None = None
    vulkan_executable: str | None = None
    sycl_executable: str | None = None
    policy: MemoryPolicy = Field(default_factory=MemoryPolicy)

    @classmethod
    def load(cls) -> "AppSettings":
        return cls(
            host=os.getenv("LLAMA_ORCH_HOST", "127.0.0.1"),
            port=int(os.getenv("LLAMA_ORCH_PORT", "8080")),
            api_key=os.getenv("LLAMA_ORCH_API_KEY") or None,
            catalog_path=Path(os.getenv("LLAMA_ORCH_CATALOG_PATH", "catalog/catalog.yaml")),
            state_path=Path(os.getenv("LLAMA_ORCH_STATE_PATH", "state/orchestrator.db")),
            idle_scan_interval_seconds=int(os.getenv("LLAMA_ORCH_IDLE_SCAN_SECONDS", "30")),
            runtime_start_timeout_seconds=int(os.getenv("LLAMA_ORCH_RUNTIME_START_TIMEOUT", "20")),
            http_timeout_seconds=float(os.getenv("LLAMA_ORCH_HTTP_TIMEOUT", "120")),
            default_idle_unload_seconds=int(os.getenv("LLAMA_ORCH_DEFAULT_IDLE_UNLOAD", "900")),
            cpu_executable=os.getenv("LLAMA_SERVER_CPU") or None,
            cuda_executable=os.getenv("LLAMA_SERVER_CUDA") or None,
            vulkan_executable=os.getenv("LLAMA_SERVER_VULKAN") or None,
            sycl_executable=os.getenv("LLAMA_SERVER_SYCL") or None,
            policy=MemoryPolicy(
                min_free_system_ram_bytes=int(os.getenv("LLAMA_ORCH_MIN_FREE_RAM", str(4 * 1024**3))),
                min_free_dgpu_vram_bytes=int(os.getenv("LLAMA_ORCH_MIN_FREE_DGPU_VRAM", str(1 * 1024**3))),
                min_free_igpu_shared_ram_bytes=int(os.getenv("LLAMA_ORCH_MIN_FREE_IGPU_RAM", str(2 * 1024**3))),
                max_loaded_instances=int(os.getenv("LLAMA_ORCH_MAX_LOADED", "4")),
                max_concurrent_requests_per_runtime=int(os.getenv("LLAMA_ORCH_MAX_CONCURRENCY", "4")),
                allow_experimental_igpu=os.getenv("LLAMA_ORCH_ALLOW_EXPERIMENTAL_IGPU", "").lower() in {"1", "true", "yes"},
                allow_experimental_mixed_gpu=os.getenv("LLAMA_ORCH_ALLOW_EXPERIMENTAL_MIXED", "").lower() in {"1", "true", "yes"},
            ),
        )

    def ensure_directories(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def executable_for_backend(self, backend: Backend) -> str | None:
        mapping = {
            Backend.CPU: self.cpu_executable,
            Backend.CUDA: self.cuda_executable,
            Backend.VULKAN: self.vulkan_executable,
            Backend.SYCL: self.sycl_executable,
        }
        return mapping[backend]
