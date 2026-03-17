"""CLI entry points."""

from __future__ import annotations

import argparse

import uvicorn

from .catalog import CatalogStore
from .hardware import HardwareProbe
from .http_api import create_app
from .mcp_server import create_mcp_server
from .router import Router
from .runtime import RuntimeManager
from .settings import AppSettings
from .state import StateStore


def validate_startup_config(settings: AppSettings, catalog: CatalogStore) -> None:
    """Fail fast when the persisted configuration is not internally valid."""
    errors = catalog.validate()
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"Catalog validation failed:\n{joined}")


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
    uvicorn.run(app, host=settings.host, port=settings.port)


def main_mcp() -> None:
    settings, catalog, state, hardware_probe, runtime_manager = build_services()
    mcp = create_mcp_server(settings, catalog, state, hardware_probe, runtime_manager)
    mcp.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the llama.cpp orchestrator")
    parser.add_argument("mode", choices=["http", "mcp"], nargs="?", default="http")
    args = parser.parse_args()
    if args.mode == "mcp":
        main_mcp()
        return
    main_http()


if __name__ == "__main__":
    main()
