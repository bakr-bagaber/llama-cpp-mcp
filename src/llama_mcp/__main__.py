"""CLI entry points."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
import re
import sys

import uvicorn

from .catalog import CatalogStore
from .hardware import HardwareProbe
from .http_api import create_app
from .mcp_server import create_mcp_server
from .models import AliasDefinition, BackendPreference, BaseModelDefinition, Capability, CatalogDocument, GenerationPreset, LoadProfile, ReasoningMode, SourceType
from .runtime import RuntimeManager
from .settings import AppSettings
from .state import StateStore
from .router import Router


def validate_startup_config(settings: AppSettings, catalog: CatalogStore) -> None:
    """Fail fast when the persisted configuration is not internally valid."""
    errors = catalog.validate()
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"Catalog validation failed:\n{joined}")


def initialize_catalog(catalog_path: Path) -> Path:
    """Write a starter catalog when the target path does not exist yet."""
    if catalog_path.exists():
        return catalog_path
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    template = files("llama_mcp").joinpath("templates/default_catalog.yaml").read_text(encoding="utf-8")
    catalog_path.write_text(template, encoding="utf-8")
    return catalog_path


def initialize_catalog_auto(settings: AppSettings, catalog_path: Path) -> Path:
    """Build a starter catalog from models discovered in the models directory."""
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    document = build_auto_catalog(settings)
    store = CatalogStore(catalog_path)
    store.save(document)
    return catalog_path


def build_auto_catalog(settings: AppSettings) -> CatalogDocument:
    models = discover_models(settings.models_dir)
    profiles = default_profiles(settings)
    presets = default_presets()
    aliases = build_aliases(models, profiles, presets)
    return CatalogDocument(models=models, profiles=profiles, presets=presets, aliases=aliases)


def discover_models(models_dir: Path) -> list[BaseModelDefinition]:
    if not models_dir.exists():
        return []
    models: list[BaseModelDefinition] = []
    for path in sorted(models_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".gguf":
            models.append(infer_model_definition(path))
    return models


def infer_model_definition(path: Path) -> BaseModelDefinition:
    stem = path.stem
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower() or "model"
    family = infer_family_from_name(stem)
    capabilities = infer_capabilities_from_name(stem)
    estimated_ram_bytes = estimate_model_ram(path)
    estimated_vram_bytes = max(estimated_ram_bytes // 2, 0)
    return BaseModelDefinition(
        id=normalized,
        display_name=stem.replace("-", " ").replace("_", " ").strip().title(),
        source=SourceType.LOCAL,
        local_path=path,
        family=family,
        capabilities=capabilities,
        estimated_ram_bytes=estimated_ram_bytes,
        estimated_vram_bytes=estimated_vram_bytes,
        size_bytes=path.stat().st_size if path.exists() else None,
    )


def infer_family_from_name(stem: str) -> str | None:
    lower = stem.lower()
    for family in ("qwen3.5", "qwen3", "qwen", "llama", "mistral", "gemma", "phi", "deepseek", "yi", "mixtral"):
        if family in lower:
            return family
    return None


def infer_capabilities_from_name(stem: str) -> list[Capability]:
    lower = stem.lower()
    if any(token in lower for token in ("embed", "embedding")):
        return [Capability.EMBEDDING]
    capabilities = [Capability.CHAT, Capability.COMPLETION, Capability.TOOLS]
    if any(token in lower for token in ("vision", "vl", "multimodal", "image")):
        capabilities.append(Capability.VISION)
    if any(token in lower for token in ("rerank", "rank")):
        capabilities.append(Capability.RERANK)
    return capabilities


def estimate_model_ram(path: Path) -> int:
    size = path.stat().st_size
    if size <= 0:
        return 2 * 1024**3
    return min(max(size * 3, 2 * 1024**3), 256 * 1024**3)


def default_profiles(settings: AppSettings) -> list[LoadProfile]:
    return [
        LoadProfile(id="cpu-safe", description="Low-risk CPU profile for reliability and reserve headroom.", context_size=8192, backend_preference=BackendPreference.PREFER_CPU, gpu_layers=0, idle_unload_seconds=1800),
        LoadProfile(id="cpu-full", description="Higher-throughput CPU profile for larger local machines.", context_size=32768, backend_preference=BackendPreference.FORCE_CPU, gpu_layers=0, batch_size=512, ubatch_size=128, idle_unload_seconds=900),
        LoadProfile(id="gpu-only", description="GPU-first profile for models that should stay off the CPU path.", context_size=32768, backend_preference=BackendPreference.FORCE_DGPU, gpu_layers=99, idle_unload_seconds=900),
        LoadProfile(id="gpu-priority", description="Prefer GPU but keep a safe CPU fallback if needed.", context_size=65536, backend_preference=BackendPreference.PREFER_DGPU, gpu_layers=99, idle_unload_seconds=900),
        LoadProfile(id="auto", description="Automatic profile selection based on hardware and model fit.", context_size=65536, backend_preference=BackendPreference.AUTO, gpu_layers=99, idle_unload_seconds=settings.default_idle_unload_seconds),
    ]


def default_presets() -> list[GenerationPreset]:
    presets: list[GenerationPreset] = []
    specs = [
        ("deterministic-think-off", "Deterministic, factual, and precise without reasoning prompts.", 0.0, 0.85, 1024, ReasoningMode.OFF),
        ("deterministic-think-on", "Deterministic with explicit reasoning enabled for harder tasks.", 0.0, 0.85, 1024, ReasoningMode.DEEP),
        ("balanced-think-off", "Instruction-following balanced preset without reasoning prompts.", 0.4, 0.92, 1536, ReasoningMode.OFF),
        ("balanced-think-on", "Instruction-following balanced preset with reasoning enabled.", 0.4, 0.92, 1536, ReasoningMode.LIGHT),
        ("creative-think-off", "Narrative and roleplay friendly preset with softer sampling.", 0.9, 0.97, 2048, ReasoningMode.OFF),
        ("creative-think-on", "Creative preset with explicit reasoning available when helpful.", 0.9, 0.97, 2048, ReasoningMode.DEEP),
        ("chaos-think-off", "Experimental, weird, and highly exploratory generation.", 1.3, 1.0, 2048, ReasoningMode.OFF),
        ("chaos-think-on", "Experimental generation with reasoning enabled.", 1.3, 1.0, 2048, ReasoningMode.DEEP),
        ("coherent-long-form-think-off", "Long-form writing with stable tone and low drift.", 0.35, 0.95, 4096, ReasoningMode.OFF),
        ("coherent-long-form-think-on", "Long-form writing with reasoning enabled for structure and planning.", 0.35, 0.95, 4096, ReasoningMode.DEEP),
    ]
    for preset_id, description, temperature, top_p, max_tokens, reasoning_mode in specs:
        presets.append(
            GenerationPreset(
                id=preset_id,
                description=description,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_mode=reasoning_mode,
            )
        )
    return presets


def build_aliases(models: list[BaseModelDefinition], profiles: list[LoadProfile], presets: list[GenerationPreset]) -> list[AliasDefinition]:
    alias_specs = [
        ("coding", {Capability.CHAT, Capability.COMPLETION}),
        ("art", {Capability.CHAT, Capability.COMPLETION}),
        ("chat", {Capability.CHAT, Capability.COMPLETION}),
        ("embed", {Capability.EMBEDDING}),
        ("rerank", {Capability.RERANK}),
    ]
    aliases: list[AliasDefinition] = []
    for model in models:
        for profile in profiles:
            for preset in presets:
                suffix = preset_bucket(preset.id)
                model_caps = set(model.capabilities)
                for label, required in alias_specs:
                    if not required.issubset(model_caps):
                        continue
                    alias_id = f"{model.id}/{profile.id}-{label}-{suffix}"
                    aliases.append(
                        AliasDefinition(
                            id=alias_id,
                            base_model_id=model.id,
                            load_profile_id=profile.id,
                            preset_id=preset.id,
                            capabilities=list(sorted(model_caps, key=lambda item: item.value)),
                            backend_preference=profile.backend_preference,
                            metadata={"auto_generated": True, "alias_family": label},
                        )
                    )
    return aliases


def preset_bucket(preset_id: str) -> str:
    for bucket in ("deterministic", "balanced", "creative", "chaos", "coherent-long-form"):
        if bucket in preset_id:
            return bucket
    return preset_id




def build_services():
    settings = AppSettings.load()
    settings.ensure_directories()
    catalog = CatalogStore(settings.catalog_path)
    catalog.load()
    validate_startup_config(settings, catalog)
    state = StateStore(settings.state_path)
    hardware_probe = HardwareProbe(settings)
    router = Router(settings)
    runtime_manager = RuntimeManager(settings=settings, catalog=catalog, state=state, router=router)
    return settings, catalog, state, hardware_probe, runtime_manager


def main_http() -> None:
    settings, catalog, state, hardware_probe, runtime_manager = build_services()
    app = create_app(settings, catalog, hardware_probe, runtime_manager, state)
    print(f"Starting HTTP server on http://{settings.host}:{settings.port}", file=sys.stderr)
    uvicorn.run(app, host=settings.host, port=settings.port)


def main_mcp() -> None:
    settings, catalog, state, hardware_probe, runtime_manager = build_services()
    mcp = create_mcp_server(settings, catalog, state, hardware_probe, runtime_manager)
    print("Starting MCP server on stdio", file=sys.stderr)
    mcp.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the llama.cpp MCP server")
    parser.add_argument("mode", choices=["http", "mcp", "init-config", "validate-config"], nargs="?", default="http")
    args = parser.parse_args()
    if args.mode == "init-config":
        settings = AppSettings.load()
        settings.ensure_directories()
        path = initialize_catalog_auto(settings, settings.catalog_path)
        print(f"Initialized auto-generated catalog at {path}")
        return
    if args.mode == "validate-config":
        settings = AppSettings.load()
        settings.ensure_directories()
        catalog = CatalogStore(settings.catalog_path)
        catalog.load()
        validate_startup_config(settings, catalog)
        print("Configuration is valid.")
        return
    if args.mode == "mcp":
        main_mcp()
        return
    main_http()


if __name__ == "__main__":
    main()
