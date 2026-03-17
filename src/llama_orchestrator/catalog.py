"""Catalog loading and lookup helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import AliasDefinition, BaseModelDefinition, CatalogDocument, GenerationPreset, LoadProfile


class CatalogError(RuntimeError):
    """Raised when catalog operations fail."""


class CatalogStore:
    """Persistent YAML-backed catalog."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.document = CatalogDocument()

    def load(self) -> CatalogDocument:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.save(CatalogDocument())
        with self.path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        self.document = CatalogDocument.model_validate(raw)
        return self.document

    def save(self, document: CatalogDocument | None = None) -> None:
        self.document = document or self.document
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.document.model_dump(mode="json"), handle, sort_keys=False)

    def list_models(self) -> list[BaseModelDefinition]:
        return self.document.models

    def list_profiles(self) -> list[LoadProfile]:
        return self.document.profiles

    def list_presets(self) -> list[GenerationPreset]:
        return self.document.presets

    def list_aliases(self) -> list[AliasDefinition]:
        return self.document.aliases

    def get_model(self, model_id: str) -> BaseModelDefinition:
        for item in self.document.models:
            if item.id == model_id:
                return item
        raise CatalogError(f"Model '{model_id}' was not found.")

    def get_profile(self, profile_id: str) -> LoadProfile:
        for item in self.document.profiles:
            if item.id == profile_id:
                return item
        raise CatalogError(f"Profile '{profile_id}' was not found.")

    def get_preset(self, preset_id: str) -> GenerationPreset:
        for item in self.document.presets:
            if item.id == preset_id:
                return item
        raise CatalogError(f"Preset '{preset_id}' was not found.")

    def get_alias(self, alias_id: str) -> AliasDefinition:
        for item in self.document.aliases:
            if item.id == alias_id:
                return item
        raise CatalogError(f"Alias '{alias_id}' was not found.")

    def resolve_alias(self, alias_id: str) -> tuple[AliasDefinition, BaseModelDefinition, LoadProfile, GenerationPreset]:
        alias = self.get_alias(alias_id)
        return alias, self.get_model(alias.base_model_id), self.get_profile(alias.load_profile_id), self.get_preset(alias.preset_id)

    def upsert_model(self, model: BaseModelDefinition) -> None:
        self._upsert_item(self.document.models, model)
        self.save()

    def upsert_profile(self, profile: LoadProfile) -> None:
        self._upsert_item(self.document.profiles, profile)
        self.save()

    def upsert_preset(self, preset: GenerationPreset) -> None:
        self._upsert_item(self.document.presets, preset)
        self.save()

    def upsert_alias(self, alias: AliasDefinition) -> None:
        self._upsert_item(self.document.aliases, alias)
        self.save()

    def delete_alias(self, alias_id: str) -> None:
        self.document.aliases = [item for item in self.document.aliases if item.id != alias_id]
        self.save()

    def delete_profile(self, profile_id: str) -> None:
        self.document.profiles = [item for item in self.document.profiles if item.id != profile_id]
        self.save()

    def delete_preset(self, preset_id: str) -> None:
        self.document.presets = [item for item in self.document.presets if item.id != preset_id]
        self.save()

    @staticmethod
    def _upsert_item(items: list, new_item: object) -> None:
        for index, item in enumerate(items):
            if getattr(item, "id") == getattr(new_item, "id"):
                items[index] = new_item
                return
        items.append(new_item)
