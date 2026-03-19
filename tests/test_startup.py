from __future__ import annotations

from pathlib import Path

import pytest

from llama_mcp.__main__ import build_auto_catalog, initialize_catalog, validate_startup_config
from llama_mcp.catalog import CatalogStore
from llama_mcp.models import AliasDefinition, BaseModelDefinition, GenerationPreset, LoadProfile
from llama_mcp.settings import AppSettings


def test_validate_startup_config_rejects_missing_local_model_path(sandbox_path: Path) -> None:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "mcp.db",
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


def test_initialize_catalog_writes_starter_template(sandbox_path: Path) -> None:
    target = sandbox_path / "catalog" / "catalog.yaml"

    initialize_catalog(target)

    text = target.read_text(encoding="utf-8")
    assert "profiles:" in text
    assert "presets:" in text
    assert "balanced" in text
    assert 'reasoning_mode: "off"' in text


def test_build_auto_catalog_discovers_models_and_generates_aliases(sandbox_path: Path) -> None:
    models_dir = sandbox_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "qwen-coder-7b.gguf"
    model_path.write_bytes(b"0" * 1024)
    settings = AppSettings(catalog_path=sandbox_path / "catalog.yaml", state_path=sandbox_path / "mcp.db", models_dir=models_dir)

    document = build_auto_catalog(settings)

    assert any(model.id == "qwen-coder-7b" for model in document.models)
    assert any(profile.id == "gpu-only" for profile in document.profiles)
    assert any(preset.id == "balanced-think-on" for preset in document.presets)
    assert any(alias.base_model_id == "qwen-coder-7b" for alias in document.aliases)
    assert any("coding" in alias.id for alias in document.aliases)


def test_appsettings_loads_dotenv_values(sandbox_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = sandbox_path / ".env"
    dotenv.write_text(
        "LLAMA_MCP_HOST=0.0.0.0\nLLAMA_MCP_MODELS_DIR=./models\nLLAMA_MCP_PORT=9090\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(sandbox_path)

    settings = AppSettings.load()

    assert settings.host == "0.0.0.0"
    assert settings.port == 9090
    assert settings.models_dir == Path("./models")
