"""Runtime process management."""

from __future__ import annotations

import asyncio
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .catalog import CatalogStore
from .models import AliasDefinition, Backend, RuntimeRecord, RuntimeStatus, SupportLevel
from .router import RouteContext, Router
from .settings import AppSettings
from .state import StateStore


class RuntimeErrorBase(RuntimeError):
    """Base runtime error."""


class RuntimeLaunchError(RuntimeErrorBase):
    """Raised when a runtime cannot be launched."""


class RuntimeManager:
    """Coordinates runtime reuse, startup, and teardown."""

    def __init__(self, settings: AppSettings, catalog: CatalogStore, state: StateStore, router: Router) -> None:
        self.settings = settings
        self.catalog = catalog
        self.state = state
        self.router = router
        self._runtimes: dict[str, RuntimeRecord] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

    def list_runtimes(self) -> list[RuntimeRecord]:
        return sorted(self._runtimes.values(), key=lambda item: item.alias_id)

    async def ensure_runtime(self, alias_id: str, inventory, backend_preference=None) -> RuntimeRecord:
        async with self._lock:
            alias, model, profile, _preset = self.catalog.resolve_alias(alias_id)
            benchmarks = self.state.list_benchmarks(alias_id)
            model_ram = model.estimated_ram_bytes or model.size_bytes or 4 * 1024**3
            model_vram = model.estimated_vram_bytes or max(model_ram // 2, 0)
            decision = self.router.choose_placement(
                alias=alias,
                profile=profile,
                model_ram_bytes=model_ram,
                model_vram_bytes=model_vram,
                context=RouteContext(
                    inventory=inventory,
                    warm_runtimes=list(self._runtimes.values()),
                    benchmarks=benchmarks,
                    requested_backend_preference=backend_preference or profile.backend_preference,
                ),
            )
            self.state.record_route(alias_id, decision.model_dump_json())
            if decision.reused_runtime_key:
                runtime = self._runtimes[decision.reused_runtime_key]
                runtime.last_used_at = self._now()
                return runtime
            if not decision.selected:
                raise RuntimeLaunchError(decision.reason_summary)
            await self._evict_if_needed(alias, decision.selected)
            runtime = await self._launch_runtime(alias, model.local_path, profile, decision.selected)
            runtime.last_used_at = self._now()
            self._runtimes[runtime.runtime_key] = runtime
            return runtime

    async def unload_runtime(self, runtime_key: str) -> None:
        runtime = self._runtimes.pop(runtime_key, None)
        process = self._processes.pop(runtime_key, None)
        if process:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except Exception:
                process.kill()
        if runtime:
            runtime.status = RuntimeStatus.STOPPED

    async def unload_idle(self) -> list[str]:
        unloaded: list[str] = []
        now = self._now()
        for runtime in list(self._runtimes.values()):
            alias = self.catalog.get_alias(runtime.alias_id)
            profile = self.catalog.get_profile(alias.load_profile_id)
            idle_limit = profile.idle_unload_seconds or self.settings.default_idle_unload_seconds
            idle_seconds = (now - runtime.last_used_at).total_seconds()
            if runtime.pinned or idle_seconds < idle_limit:
                continue
            await self.unload_runtime(runtime.runtime_key)
            unloaded.append(runtime.runtime_key)
        return unloaded

    async def mark_pinned(self, alias_id: str, pinned: bool) -> None:
        for runtime in self._runtimes.values():
            if runtime.alias_id == alias_id:
                runtime.pinned = pinned

    async def proxy_json(self, runtime: RuntimeRecord, path: str, payload: dict[str, Any], stream: bool = False):
        timeout = httpx.Timeout(self.settings.http_timeout_seconds)
        url = runtime.endpoint_url.rstrip("/") + path
        client = httpx.AsyncClient(timeout=timeout)
        if stream:
            return client.stream("POST", url, json=payload)
        response = await client.post(url, json=payload)
        runtime.last_used_at = self._now()
        return response

    async def _evict_if_needed(self, alias: AliasDefinition, selected) -> None:
        if len(self._runtimes) < self.settings.policy.max_loaded_instances:
            return
        candidates = [runtime for runtime in self._runtimes.values() if not runtime.pinned]
        if not candidates:
            raise RuntimeLaunchError("All loaded runtimes are pinned and the max loaded instance limit was reached.")
        # Prefer removing the oldest non-pinned runtime first so active models stay warm.
        candidates.sort(key=lambda item: (item.alias_id == alias.id, item.last_used_at))
        await self.unload_runtime(candidates[0].runtime_key)

    async def _launch_runtime(
        self,
        alias: AliasDefinition,
        model_path: Path | None,
        profile,
        selected,
    ) -> RuntimeRecord:
        executable = self._resolve_executable(selected.backend)
        if not executable:
            raise RuntimeLaunchError(f"No executable is configured or discoverable for backend '{selected.backend.value}'.")
        if not model_path:
            raise RuntimeLaunchError(f"Alias '{alias.id}' does not point to a local model path yet.")
        if not model_path.exists():
            raise RuntimeLaunchError(f"Model path '{model_path}' does not exist.")

        port = self._choose_port()
        endpoint_url = f"http://127.0.0.1:{port}"
        runtime_key = f"{alias.id}:{selected.backend.value}:{selected.placement.value}"
        command = [
            executable,
            "-m",
            str(model_path),
            "--port",
            str(port),
            "--ctx-size",
            str(profile.context_size),
        ]
        if profile.threads:
            command.extend(["--threads", str(profile.threads)])
        if profile.batch_size:
            command.extend(["--batch-size", str(profile.batch_size)])
        if profile.ubatch_size:
            command.extend(["--ubatch-size", str(profile.ubatch_size)])
        if profile.gpu_layers is not None and selected.backend is not Backend.CPU:
            command.extend(["--n-gpu-layers", str(profile.gpu_layers)])
        if profile.embedding_mode:
            command.append("--embeddings")
        if profile.flash_attention:
            command.append("--flash-attn")
        command.extend(profile.extra_args)

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        runtime = RuntimeRecord(
            runtime_key=runtime_key,
            alias_id=alias.id,
            model_id=alias.base_model_id,
            profile_id=alias.load_profile_id,
            backend=selected.backend,
            placement=selected.placement,
            endpoint_url=endpoint_url,
            support_level=selected.support_level,
            process_id=process.pid,
            command=command,
            estimated_ram_bytes=selected.estimated.ram_bytes,
            estimated_vram_bytes=selected.estimated.vram_bytes,
            experimental=selected.support_level is SupportLevel.EXPERIMENTAL,
        )
        self._processes[runtime_key] = process
        try:
            await self._wait_until_ready(endpoint_url)
        except Exception as exc:  # pragma: no cover - depends on local binaries
            runtime.status = RuntimeStatus.FAILED
            runtime.failure_reason = str(exc)
            await self.unload_runtime(runtime_key)
            raise RuntimeLaunchError(str(exc)) from exc
        runtime.status = RuntimeStatus.READY
        return runtime

    async def _wait_until_ready(self, endpoint_url: str) -> None:
        timeout = self.settings.runtime_start_timeout_seconds
        async with httpx.AsyncClient(timeout=2.0) as client:
            for _ in range(timeout * 2):
                try:
                    response = await client.get(f"{endpoint_url}/health")
                    if response.status_code < 500:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        raise RuntimeLaunchError(f"Runtime at {endpoint_url} did not become ready within {timeout} seconds.")

    def _resolve_executable(self, backend: Backend) -> str | None:
        preferred = self.settings.executable_for_backend(backend)
        if preferred:
            return preferred
        return "llama-server"

    @staticmethod
    def _choose_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
