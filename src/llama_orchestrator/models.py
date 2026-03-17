"""Core data models for the orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Shared strict Pydantic configuration."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class Backend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    VULKAN = "vulkan"
    SYCL = "sycl"


class PlacementKind(str, Enum):
    CPU_ONLY = "cpu_only"
    DGPU_ONLY = "dgpu_only"
    IGPU_ONLY = "igpu_only"
    CPU_DGPU_HYBRID = "cpu_dgpu_hybrid"
    CPU_IGPU_HYBRID = "cpu_igpu_hybrid"
    DGPU_IGPU_MIXED = "dgpu_igpu_mixed"
    SAME_BACKEND_MULTI_GPU = "same_backend_multi_gpu"


class SupportLevel(str, Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"


class Capability(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    TOOLS = "tools"
    VISION = "vision"
    RERANK = "rerank"


class ReasoningMode(str, Enum):
    OFF = "off"
    LIGHT = "light"
    DEEP = "deep"
    MODEL_NATIVE = "model_native"


class BackendPreference(str, Enum):
    AUTO = "auto"
    PREFER_CPU = "prefer_cpu"
    PREFER_DGPU = "prefer_dgpu"
    PREFER_IGPU = "prefer_igpu"
    FORCE_CPU = "force_cpu"
    FORCE_DGPU = "force_dgpu"
    FORCE_IGPU = "force_igpu"


class SourceType(str, Enum):
    LOCAL = "local"
    HUGGING_FACE = "hugging_face"
    URL = "url"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaseModelDefinition(StrictModel):
    id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    source: SourceType = Field(default=SourceType.LOCAL)
    local_path: Path | None = None
    hf_repo: str | None = None
    hf_filename: str | None = None
    family: str | None = None
    quantization: str | None = None
    capabilities: list[Capability] = Field(default_factory=lambda: [Capability.CHAT, Capability.COMPLETION])
    metadata: dict[str, Any] = Field(default_factory=dict)
    size_bytes: int | None = Field(default=None, ge=0)
    estimated_ram_bytes: int | None = Field(default=None, ge=0)
    estimated_vram_bytes: int | None = Field(default=None, ge=0)

    @field_validator("local_path")
    @classmethod
    def _expand_local_path(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value else None


class LoadProfile(StrictModel):
    id: str = Field(..., min_length=1)
    description: str | None = None
    context_size: int = Field(default=8192, ge=256)
    threads: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    backend_preference: BackendPreference = Field(default=BackendPreference.AUTO)
    gpu_layers: int | None = Field(default=None, ge=0)
    tensor_split: list[float] = Field(default_factory=list)
    flash_attention: bool = False
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    embedding_mode: bool = False
    reranking_mode: bool = False
    extra_args: list[str] = Field(default_factory=list)
    idle_unload_seconds: int | None = Field(default=None, ge=1)


class GenerationPreset(StrictModel):
    id: str = Field(..., min_length=1)
    description: str | None = None
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repeat_penalty: float | None = Field(default=None, ge=0.0)
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] = Field(default_factory=list)
    json_mode: bool = False
    grammar: str | None = None
    reasoning_mode: ReasoningMode = Field(default=ReasoningMode.OFF)
    reasoning_effort: str | None = None
    request_overrides: dict[str, Any] = Field(default_factory=dict)


class AliasDefinition(StrictModel):
    id: str = Field(..., min_length=1)
    base_model_id: str = Field(..., min_length=1)
    load_profile_id: str = Field(..., min_length=1)
    preset_id: str = Field(..., min_length=1)
    capabilities: list[Capability] = Field(default_factory=list)
    backend_preference: BackendPreference | None = None
    experimental: bool = False
    priority: int = Field(default=0, ge=-100, le=100)
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CatalogDocument(StrictModel):
    models: list[BaseModelDefinition] = Field(default_factory=list)
    profiles: list[LoadProfile] = Field(default_factory=list)
    presets: list[GenerationPreset] = Field(default_factory=list)
    aliases: list[AliasDefinition] = Field(default_factory=list)


class HardwareDevice(StrictModel):
    id: str
    name: str
    backend_candidates: list[Backend] = Field(default_factory=list)
    kind: str = Field(description="cpu, dgpu, or igpu")
    total_memory_bytes: int | None = Field(default=None, ge=0)
    free_memory_bytes: int | None = Field(default=None, ge=0)
    driver: str | None = None
    experimental: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class HardwareInventory(StrictModel):
    collected_at: datetime = Field(default_factory=utc_now)
    cpu_count: int = Field(default=1, ge=1)
    system_ram_total_bytes: int = Field(default=0, ge=0)
    system_ram_free_bytes: int = Field(default=0, ge=0)
    backends_available: list[Backend] = Field(default_factory=list)
    devices: list[HardwareDevice] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def devices_by_kind(self, kind: str) -> list[HardwareDevice]:
        return [device for device in self.devices if device.kind == kind]


class BenchmarkRecord(StrictModel):
    alias_id: str
    backend: Backend
    placement: PlacementKind
    prompt_tps: float = Field(default=0.0, ge=0.0)
    generation_tps: float = Field(default=0.0, ge=0.0)
    load_seconds: float = Field(default=0.0, ge=0.0)
    peak_ram_bytes: int | None = Field(default=None, ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    collected_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlacementEstimate(StrictModel):
    ram_bytes: int = Field(default=0, ge=0)
    vram_bytes: int = Field(default=0, ge=0)


class CandidatePlacement(StrictModel):
    backend: Backend
    placement: PlacementKind
    support_level: SupportLevel
    devices: list[str] = Field(default_factory=list)
    feasible: bool = True
    reason: str = ""
    estimated: PlacementEstimate = Field(default_factory=PlacementEstimate)
    score: float = 0.0


class RoutingDecision(StrictModel):
    alias_id: str
    selected: CandidatePlacement | None = None
    candidates: list[CandidatePlacement] = Field(default_factory=list)
    reason_summary: str = ""
    reused_runtime_key: str | None = None


class RuntimeStatus(str, Enum):
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class RuntimeRecord(StrictModel):
    runtime_key: str
    alias_id: str
    model_id: str
    profile_id: str
    backend: Backend
    placement: PlacementKind
    endpoint_url: str
    support_level: SupportLevel
    status: RuntimeStatus = Field(default=RuntimeStatus.STARTING)
    pinned: bool = False
    experimental: bool = False
    process_id: int | None = None
    command: list[str] = Field(default_factory=list)
    estimated_ram_bytes: int = Field(default=0, ge=0)
    estimated_vram_bytes: int = Field(default=0, ge=0)
    launched_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime = Field(default_factory=utc_now)
    failure_reason: str | None = None


class MemoryPolicy(StrictModel):
    min_free_system_ram_bytes: int = Field(default=4 * 1024**3, ge=0)
    min_free_dgpu_vram_bytes: int = Field(default=1 * 1024**3, ge=0)
    min_free_igpu_shared_ram_bytes: int = Field(default=2 * 1024**3, ge=0)
    max_loaded_instances: int = Field(default=4, ge=1)
    max_concurrent_requests_per_runtime: int = Field(default=4, ge=1)
    allow_experimental_igpu: bool = False
    allow_experimental_mixed_gpu: bool = False
