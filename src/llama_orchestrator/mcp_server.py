"""MCP control-plane server."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .benchmarks import BenchmarkService
from .catalog import CatalogStore
from .downloads import download_model
from .hardware import HardwareProbe
from .models import AliasDefinition, BaseModelDefinition, GenerationPreset, LoadProfile
from .router import RouteContext
from .runtime import RuntimeManager
from .settings import AppSettings
from .state import StateStore

try:  # pragma: no cover - depends on optional dependency availability
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - handled at runtime
    FastMCP = None


def create_mcp_server(
    settings: AppSettings,
    catalog: CatalogStore,
    state: StateStore,
    hardware_probe: HardwareProbe,
    runtime_manager: RuntimeManager,
):
    """Create the FastMCP server if the SDK is available."""

    if FastMCP is None:  # pragma: no cover - depends on environment
        raise RuntimeError("The 'mcp' package is not installed. Install dependencies before starting the MCP server.")

    mcp = FastMCP("llama_orchestrator")
    benchmark_service = BenchmarkService(settings, catalog, state)

    @mcp.tool(name="llama_list_models", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_models() -> dict[str, Any]:
        return {"models": [model.model_dump(mode="json") for model in catalog.list_models()]}

    @mcp.tool(name="llama_list_profiles", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_profiles() -> dict[str, Any]:
        return {"profiles": [profile.model_dump(mode="json") for profile in catalog.list_profiles()]}

    @mcp.tool(name="llama_list_presets", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_presets() -> dict[str, Any]:
        return {"presets": [preset.model_dump(mode="json") for preset in catalog.list_presets()]}

    @mcp.tool(name="llama_list_aliases", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_aliases() -> dict[str, Any]:
        return {"aliases": [alias.model_dump(mode="json") for alias in catalog.list_aliases()]}

    @mcp.tool(name="llama_get_model", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_model(params: dict[str, Any]) -> dict[str, Any]:
        model = catalog.get_model(str(params["model_id"]))
        return {"model": model.model_dump(mode="json")}

    @mcp.tool(name="llama_get_profile", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_profile(params: dict[str, Any]) -> dict[str, Any]:
        profile = catalog.get_profile(str(params["profile_id"]))
        return {"profile": profile.model_dump(mode="json")}

    @mcp.tool(name="llama_get_preset", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_preset(params: dict[str, Any]) -> dict[str, Any]:
        preset = catalog.get_preset(str(params["preset_id"]))
        return {"preset": preset.model_dump(mode="json")}

    @mcp.tool(name="llama_get_alias", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_alias(params: dict[str, Any]) -> dict[str, Any]:
        alias, model, profile, preset = catalog.resolve_alias(str(params["alias_id"]))
        return {
            "alias": alias.model_dump(mode="json"),
            "resolved_model": model.model_dump(mode="json"),
            "resolved_profile": profile.model_dump(mode="json"),
            "resolved_preset": preset.model_dump(mode="json"),
        }

    @mcp.tool(name="llama_import_model", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def import_model(params: dict[str, Any]) -> dict[str, Any]:
        model = BaseModelDefinition.model_validate(params)
        catalog.upsert_model(model)
        return {"ok": True, "model": model.model_dump(mode="json")}

    @mcp.tool(name="llama_delete_model", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def delete_model(params: dict[str, Any]) -> dict[str, Any]:
        model_id = str(params["model_id"])
        catalog.delete_model(model_id)
        return {"ok": True, "model_id": model_id}

    @mcp.tool(name="llama_download_model", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True})
    async def download_remote_model(params: dict[str, Any]) -> dict[str, Any]:
        model = BaseModelDefinition.model_validate(params["model"])
        destination_dir = settings.catalog_path.parent / "models"
        downloaded_path = download_model(model, destination_dir)
        model.local_path = downloaded_path
        catalog.upsert_model(model)
        return {"ok": True, "model": model.model_dump(mode="json"), "local_path": str(downloaded_path)}

    @mcp.tool(name="llama_create_profile", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def create_profile(params: dict[str, Any]) -> dict[str, Any]:
        profile = LoadProfile.model_validate(params)
        catalog.upsert_profile(profile)
        return {"ok": True, "profile": profile.model_dump(mode="json")}

    @mcp.tool(name="llama_create_preset", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def create_preset(params: dict[str, Any]) -> dict[str, Any]:
        preset = GenerationPreset.model_validate(params)
        catalog.upsert_preset(preset)
        return {"ok": True, "preset": preset.model_dump(mode="json")}

    @mcp.tool(name="llama_delete_profile", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def delete_profile(params: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(params["profile_id"])
        catalog.delete_profile(profile_id)
        return {"ok": True, "profile_id": profile_id}

    @mcp.tool(name="llama_delete_preset", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def delete_preset(params: dict[str, Any]) -> dict[str, Any]:
        preset_id = str(params["preset_id"])
        catalog.delete_preset(preset_id)
        return {"ok": True, "preset_id": preset_id}

    @mcp.tool(name="llama_create_alias", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def create_alias(params: dict[str, Any]) -> dict[str, Any]:
        alias = AliasDefinition.model_validate(params)
        catalog.upsert_alias(alias)
        return {"ok": True, "alias": alias.model_dump(mode="json")}

    @mcp.tool(name="llama_get_hardware", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_hardware() -> dict[str, Any]:
        return hardware_probe.collect().model_dump(mode="json")

    @mcp.tool(name="llama_get_runtime_status", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_runtime_status() -> dict[str, Any]:
        return {"runtimes": [runtime.model_dump(mode="json") for runtime in runtime_manager.list_runtimes()]}

    @mcp.tool(name="llama_get_runtime_diagnostics", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_runtime_diagnostics() -> dict[str, Any]:
        inventory = hardware_probe.collect()
        runtimes = runtime_manager.list_runtimes()
        return _runtime_diagnostics_payload(settings, inventory.model_dump(mode="json"), [runtime.model_dump(mode="json") for runtime in runtimes])

    @mcp.tool(name="llama_load_alias", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def load_alias(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        runtime = await runtime_manager.ensure_runtime(alias_id, inventory=hardware_probe.collect())
        return {"ok": True, "runtime": runtime.model_dump(mode="json")}

    @mcp.tool(name="llama_unload_alias", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def unload_alias(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        unloaded = []
        for runtime in list(runtime_manager.list_runtimes()):
            if runtime.alias_id == alias_id:
                await runtime_manager.unload_runtime(runtime.runtime_key)
                unloaded.append(runtime.runtime_key)
        return {"ok": True, "unloaded": unloaded}

    @mcp.tool(name="llama_unload_idle", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def unload_idle() -> dict[str, Any]:
        unloaded = await runtime_manager.unload_idle()
        return {"ok": True, "unloaded": unloaded}

    @mcp.tool(name="llama_pin_alias", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def pin_alias(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        pinned = bool(params.get("pinned", True))
        await runtime_manager.mark_pinned(alias_id, pinned)
        alias = catalog.get_alias(alias_id)
        alias.pinned = pinned
        catalog.upsert_alias(alias)
        return {"ok": True, "alias_id": alias_id, "pinned": pinned}

    @mcp.tool(name="llama_delete_alias", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def delete_alias(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        catalog.delete_alias(alias_id)
        return {"ok": True, "alias_id": alias_id}

    @mcp.tool(name="llama_list_benchmarks", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_benchmarks(params: dict[str, Any] | None = None) -> dict[str, Any]:
        alias_id = str(params["alias_id"]) if params and "alias_id" in params else None
        records = state.list_benchmarks(alias_id)
        return {"benchmarks": [record.model_dump(mode="json") for record in records]}

    @mcp.tool(name="llama_benchmark_summary", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def benchmark_summary(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        records = state.list_benchmarks(alias_id)
        return _benchmark_summary_payload(alias_id, [record.model_dump(mode="json") for record in records])

    @mcp.tool(name="llama_verify_benchmark", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def verify_benchmark(params: dict[str, Any]) -> dict[str, Any]:
        record = benchmark_service.mark_benchmark_verified(
            alias_id=str(params["alias_id"]),
            backend=params["backend"],
            placement=params["placement"],
            collected_at=str(params["collected_at"]),
            verified=bool(params.get("verified", True)),
            note=str(params["note"]) if params.get("note") is not None else None,
        )
        return {"ok": True, "benchmark": record.model_dump(mode="json")}

    @mcp.tool(name="llama_delete_benchmark", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def delete_benchmark(params: dict[str, Any]) -> dict[str, Any]:
        benchmark_service.delete_benchmark(
            alias_id=str(params["alias_id"]),
            backend=params["backend"],
            placement=params["placement"],
            collected_at=str(params["collected_at"]),
        )
        return {"ok": True}

    @mcp.tool(name="llama_record_benchmark", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def record_benchmark(params: dict[str, Any]) -> dict[str, Any]:
        record = benchmark_service.record_manual_benchmark(
            alias_id=str(params["alias_id"]),
            backend=params["backend"],
            placement=params["placement"],
            prompt_tps=float(params["prompt_tps"]),
            generation_tps=float(params["generation_tps"]),
            load_seconds=float(params.get("load_seconds", 0.0)),
            peak_ram_bytes=int(params["peak_ram_bytes"]) if params.get("peak_ram_bytes") is not None else None,
            peak_vram_bytes=int(params["peak_vram_bytes"]) if params.get("peak_vram_bytes") is not None else None,
            metadata=dict(params.get("metadata", {})),
        )
        return {"ok": True, "benchmark": record.model_dump(mode="json")}

    @mcp.tool(name="llama_run_benchmark", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def run_benchmark(params: dict[str, Any]) -> dict[str, Any]:
        record = benchmark_service.run_llama_bench(
            alias_id=str(params["alias_id"]),
            backend=params["backend"],
            n_gpu_layers=int(params["n_gpu_layers"]) if params.get("n_gpu_layers") is not None else None,
            placement=params.get("placement"),
            inventory=hardware_probe.collect(),
            device_ids=[str(item) for item in params.get("device_ids", [])],
        )
        return {"ok": True, "benchmark": record.model_dump(mode="json")}

    @mcp.tool(name="llama_get_memory_policy", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def get_memory_policy() -> dict[str, Any]:
        return settings.policy.model_dump(mode="json")

    @mcp.tool(name="llama_set_memory_policy", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def set_memory_policy(params: dict[str, Any]) -> dict[str, Any]:
        updated = settings.policy.model_copy(update=params)
        settings.policy = updated
        return {"ok": True, "policy": updated.model_dump(mode="json")}

    @mcp.tool(name="llama_route_explain", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def route_explain(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        alias, model, profile, _ = catalog.resolve_alias(alias_id)
        inventory = hardware_probe.collect()
        decision = runtime_manager.router.choose_placement(
            alias=alias,
            profile=profile,
            model_ram_bytes=model.estimated_ram_bytes or model.size_bytes or 4 * 1024**3,
            model_vram_bytes=model.estimated_vram_bytes or 2 * 1024**3,
            context=RouteContext(
                inventory=inventory,
                warm_runtimes=runtime_manager.list_runtimes(),
                benchmarks=state.list_benchmarks(alias_id),
                requested_backend_preference=alias.backend_preference or profile.backend_preference,
            ),
        )
        selected = decision.selected
        summary = {
            "alias_id": alias_id,
            "selected_backend": selected.backend.value if selected else None,
            "selected_placement": selected.placement.value if selected else None,
            "selected_devices": selected.devices if selected else [],
            "reason_summary": decision.reason_summary,
            "warm_runtime_reused": bool(decision.reused_runtime_key),
            "inventory_backends": [backend.value for backend in inventory.backends_available],
        }
        return {"summary": summary, "decision": decision.model_dump(mode="json")}

    @mcp.tool(name="llama_list_route_events", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_route_events(params: dict[str, Any] | None = None) -> dict[str, Any]:
        alias_id = str(params["alias_id"]) if params and "alias_id" in params else None
        limit = int(params["limit"]) if params and "limit" in params else 20
        events = state.list_route_events(alias_id=alias_id, limit=limit)
        return {"route_events": events}

    @mcp.tool(name="llama_route_simulate", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def route_simulate(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        alias, model, profile, _ = catalog.resolve_alias(alias_id)
        inventory = hardware_probe.collect()
        model_ram_bytes = int(params.get("model_ram_bytes", model.estimated_ram_bytes or model.size_bytes or 4 * 1024**3))
        model_vram_bytes = int(params.get("model_vram_bytes", model.estimated_vram_bytes or 2 * 1024**3))
        decision = runtime_manager.router.choose_placement(
            alias=alias,
            profile=profile,
            model_ram_bytes=model_ram_bytes,
            model_vram_bytes=model_vram_bytes,
            context=RouteContext(
                inventory=inventory,
                warm_runtimes=runtime_manager.list_runtimes(),
                benchmarks=state.list_benchmarks(alias_id),
                requested_backend_preference=alias.backend_preference or profile.backend_preference,
            ),
        )
        return {
            "decision": decision.model_dump(mode="json"),
            "inventory": inventory.model_dump(mode="json"),
        }

    return mcp


def _runtime_diagnostics_payload(settings: AppSettings, inventory: dict[str, Any], runtimes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact operator-facing runtime summary."""
    now = datetime.now(timezone.utc)
    diagnostic_rows: list[dict[str, Any]] = []
    for runtime in runtimes:
        last_used = _parse_datetime(runtime.get("last_used_at"))
        launched_at = _parse_datetime(runtime.get("launched_at"))
        diagnostic_rows.append(
            {
                "runtime_key": runtime.get("runtime_key"),
                "alias_id": runtime.get("alias_id"),
                "status": runtime.get("status"),
                "backend": runtime.get("backend"),
                "placement": runtime.get("placement"),
                "experimental": bool(runtime.get("experimental")),
                "pinned": bool(runtime.get("pinned")),
                "idle_seconds": round((now - last_used).total_seconds(), 1) if last_used else None,
                "uptime_seconds": round((now - launched_at).total_seconds(), 1) if launched_at else None,
                "estimated_ram_gib": _bytes_to_gib(runtime.get("estimated_ram_bytes")),
                "estimated_vram_gib": _bytes_to_gib(runtime.get("estimated_vram_bytes")),
                "endpoint_url": runtime.get("endpoint_url"),
            }
        )
    return {
        "summary": {
            "loaded_runtime_count": len(runtimes),
            "max_loaded_instances": settings.policy.max_loaded_instances,
            "system_ram_free_gib": _bytes_to_gib(inventory.get("system_ram_free_bytes")),
            "system_ram_total_gib": _bytes_to_gib(inventory.get("system_ram_total_bytes")),
            "available_backends": inventory.get("backends_available", []),
        },
        "runtimes": diagnostic_rows,
    }


def _benchmark_summary_payload(alias_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize benchmark quality so operators can spot weak evidence quickly."""
    best_verified: dict[str, dict[str, Any]] = {}
    unverified_count = 0
    for record in records:
        metadata = dict(record.get("metadata", {}))
        placement = str(record.get("placement"))
        score = float(record.get("prompt_tps", 0.0)) + float(record.get("generation_tps", 0.0))
        if not metadata.get("verified", True):
            unverified_count += 1
            continue
        current = best_verified.get(placement)
        if current is None or score > current["combined_tps"]:
            best_verified[placement] = {
                "backend": record.get("backend"),
                "placement": placement,
                "prompt_tps": record.get("prompt_tps"),
                "generation_tps": record.get("generation_tps"),
                "combined_tps": score,
                "collected_at": record.get("collected_at"),
            }
    return {
        "alias_id": alias_id,
        "total_records": len(records),
        "unverified_records": unverified_count,
        "best_verified_by_placement": list(best_verified.values()),
    }


def _bytes_to_gib(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 3)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
