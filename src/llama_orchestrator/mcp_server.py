"""MCP control-plane server."""

from __future__ import annotations

from typing import Any

from .catalog import CatalogStore
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

    @mcp.tool(name="llama_list_models", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_models() -> dict[str, Any]:
        return {"models": [model.model_dump(mode="json") for model in catalog.list_models()]}

    @mcp.tool(name="llama_import_model", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def import_model(params: dict[str, Any]) -> dict[str, Any]:
        model = BaseModelDefinition.model_validate(params)
        catalog.upsert_model(model)
        return {"ok": True, "model": model.model_dump(mode="json")}

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

    @mcp.tool(name="llama_pin_alias", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
    async def pin_alias(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        pinned = bool(params.get("pinned", True))
        await runtime_manager.mark_pinned(alias_id, pinned)
        alias = catalog.get_alias(alias_id)
        alias.pinned = pinned
        catalog.upsert_alias(alias)
        return {"ok": True, "alias_id": alias_id, "pinned": pinned}

    @mcp.tool(name="llama_list_benchmarks", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def list_benchmarks(params: dict[str, Any] | None = None) -> dict[str, Any]:
        alias_id = str(params["alias_id"]) if params and "alias_id" in params else None
        records = state.list_benchmarks(alias_id)
        return {"benchmarks": [record.model_dump(mode="json") for record in records]}

    @mcp.tool(name="llama_route_explain", annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
    async def route_explain(params: dict[str, Any]) -> dict[str, Any]:
        alias_id = str(params["alias_id"])
        alias, model, profile, _ = catalog.resolve_alias(alias_id)
        decision = runtime_manager.router.choose_placement(
            alias=alias,
            profile=profile,
            model_ram_bytes=model.estimated_ram_bytes or model.size_bytes or 4 * 1024**3,
            model_vram_bytes=model.estimated_vram_bytes or 2 * 1024**3,
            context=RouteContext(
                inventory=hardware_probe.collect(),
                warm_runtimes=runtime_manager.list_runtimes(),
                benchmarks=state.list_benchmarks(alias_id),
                requested_backend_preference=alias.backend_preference or profile.backend_preference,
            ),
        )
        return decision.model_dump(mode="json")

    return mcp
