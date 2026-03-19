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
    state_path: Path = Path("state/mcp.db")
    models_dir: Path = Path(r"C:\llama.cpp\models")
    idle_scan_interval_seconds: int = Field(default=30, ge=1)
    runtime_start_timeout_seconds: int = Field(default=20, ge=1)
    http_timeout_seconds: float = Field(default=120.0, gt=0)
    default_idle_unload_seconds: int = Field(default=900, ge=1)
    cpu_executable: str | None = None
    cuda_executable: str | None = None
    vulkan_executable: str | None = None
    sycl_executable: str | None = None
    cpu_bench_executable: str | None = None
    cuda_bench_executable: str | None = None
    vulkan_bench_executable: str | None = None
    sycl_bench_executable: str | None = None
    policy: MemoryPolicy = Field(default_factory=MemoryPolicy)

    @classmethod
    def load(cls) -> "AppSettings":
        dotenv = cls._load_dotenv()
        return cls(
            host=cls._env("LLAMA_MCP_HOST", dotenv, "127.0.0.1"),
            port=int(cls._env("LLAMA_MCP_PORT", dotenv, "8080")),
            api_key=cls._env("LLAMA_MCP_API_KEY", dotenv) or None,
            catalog_path=Path(cls._env("LLAMA_MCP_CATALOG_PATH", dotenv, "catalog/catalog.yaml")),
            state_path=Path(cls._env("LLAMA_MCP_STATE_PATH", dotenv, "state/mcp.db")),
            models_dir=Path(cls._env("LLAMA_MCP_MODELS_DIR", dotenv, r"C:\llama.cpp\models")),
            idle_scan_interval_seconds=int(cls._env("LLAMA_MCP_IDLE_SCAN_SECONDS", dotenv, "30")),
            runtime_start_timeout_seconds=int(cls._env("LLAMA_MCP_RUNTIME_START_TIMEOUT", dotenv, "20")),
            http_timeout_seconds=float(cls._env("LLAMA_MCP_HTTP_TIMEOUT", dotenv, "120")),
            default_idle_unload_seconds=int(cls._env("LLAMA_MCP_DEFAULT_IDLE_UNLOAD", dotenv, "900")),
            cpu_executable=cls._env("LLAMA_SERVER_CPU", dotenv) or None,
            cuda_executable=cls._env("LLAMA_SERVER_CUDA", dotenv) or None,
            vulkan_executable=cls._env("LLAMA_SERVER_VULKAN", dotenv) or None,
            sycl_executable=cls._env("LLAMA_SERVER_SYCL", dotenv) or None,
            cpu_bench_executable=cls._env("LLAMA_BENCH_CPU", dotenv) or None,
            cuda_bench_executable=cls._env("LLAMA_BENCH_CUDA", dotenv) or None,
            vulkan_bench_executable=cls._env("LLAMA_BENCH_VULKAN", dotenv) or None,
            sycl_bench_executable=cls._env("LLAMA_BENCH_SYCL", dotenv) or None,
            policy=MemoryPolicy(
                min_free_system_ram_bytes=int(cls._env("LLAMA_MCP_MIN_FREE_RAM", dotenv, str(4 * 1024**3))),
                min_free_dgpu_vram_bytes=int(cls._env("LLAMA_MCP_MIN_FREE_DGPU_VRAM", dotenv, str(1 * 1024**3))),
                min_free_igpu_shared_ram_bytes=int(cls._env("LLAMA_MCP_MIN_FREE_IGPU_RAM", dotenv, str(2 * 1024**3))),
                max_loaded_instances=int(cls._env("LLAMA_MCP_MAX_LOADED", dotenv, "4")),
                max_concurrent_requests_per_runtime=int(cls._env("LLAMA_MCP_MAX_CONCURRENCY", dotenv, "4")),
                allow_experimental_igpu=cls._env("LLAMA_MCP_ALLOW_EXPERIMENTAL_IGPU", dotenv, "").lower() in {"1", "true", "yes"},
                allow_experimental_mixed_gpu=cls._env("LLAMA_MCP_ALLOW_EXPERIMENTAL_MIXED", dotenv, "").lower() in {"1", "true", "yes"},
            ),
        )

    @staticmethod
    def _load_dotenv(path: Path | None = None) -> dict[str, str]:
        dotenv_path = path or Path(".env")
        if not dotenv_path.exists():
            return {}
        values: dict[str, str] = {}
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
        return values

    @staticmethod
    def _env(name: str, dotenv: dict[str, str], default: str | None = None) -> str:
        return os.getenv(name) or dotenv.get(name) or (default if default is not None else "")

    def ensure_directories(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def executable_for_backend(self, backend: Backend) -> str | None:
        """Resolve a backend executable.

        We prefer explicit environment overrides first. If none are set,
        we fall back to common local Windows llama.cpp install locations
        so the project works out of the box on this machine.
        """
        mapping = {
            Backend.CPU: self.cpu_executable,
            Backend.CUDA: self.cuda_executable,
            Backend.VULKAN: self.vulkan_executable,
            Backend.SYCL: self.sycl_executable,
        }
        explicit = mapping[backend]
        if explicit:
            return explicit

        defaults = {
            Backend.CPU: [Path(r"C:\llama.cpp\cpu\llama-server.exe")],
            Backend.CUDA: [Path(r"C:\llama.cpp\cuda\llama-server.exe")],
            Backend.VULKAN: [Path(r"C:\llama.cpp\vulkan\llama-server.exe")],
            Backend.SYCL: [Path(r"C:\llama.cpp\sycl\llama-server.exe")],
        }
        for candidate in defaults[backend]:
            if candidate.exists():
                return str(candidate)
        return None

    def bench_executable_for_backend(self, backend: Backend) -> str | None:
        mapping = {
            Backend.CPU: self.cpu_bench_executable,
            Backend.CUDA: self.cuda_bench_executable,
            Backend.VULKAN: self.vulkan_bench_executable,
            Backend.SYCL: self.sycl_bench_executable,
        }
        explicit = mapping[backend]
        if explicit:
            return explicit

        defaults = {
            Backend.CPU: [Path(r"C:\llama.cpp\cpu\llama-bench.exe")],
            Backend.CUDA: [Path(r"C:\llama.cpp\cuda\llama-bench.exe")],
            Backend.VULKAN: [Path(r"C:\llama.cpp\vulkan\llama-bench.exe")],
            Backend.SYCL: [Path(r"C:\llama.cpp\sycl\llama-bench.exe")],
        }
        for candidate in defaults[backend]:
            if candidate.exists():
                return str(candidate)
        return None
