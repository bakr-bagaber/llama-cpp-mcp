from __future__ import annotations

from pathlib import Path

import pytest

from llama_orchestrator.__main__ import validate_startup_config
from llama_orchestrator.catalog import CatalogStore
from llama_orchestrator.models import AliasDefinition, BaseModelDefinition, GenerationPreset, LoadProfile
from llama_orchestrator.settings import AppSettings


def test_validate_startup_config_rejects_missing_local_model_path(sandbox_path: Path) -> None:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "orchestrator.db",
    )
    settings.ensure_directories()
    catalog = CatalogStore(settings.catalog_path)
    catalog.load()
    catalog.upsert_model(
        BaseModelDefinition(
            id="missing-model",
            display_name="Missing",
            local_path=sandbox_path / "missing.gguf",
        )
    )
    catalog.upsert_profile(LoadProfile(id="balanced"))
    catalog.upsert_preset(GenerationPreset(id="default"))
    catalog.upsert_alias(
        AliasDefinition(
            id="missing/alias",
            base_model_id="missing-model",
            load_profile_id="balanced",
            preset_id="default",
        )
    )

    with pytest.raises(RuntimeError):
        validate_startup_config(settings, catalog)
