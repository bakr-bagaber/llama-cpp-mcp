# Llama.cpp Orchestrator

Local orchestration layer for `llama.cpp` with:

- OpenAI-compatible HTTP endpoints
- Anthropic-compatible HTTP endpoints
- an MCP control plane for model and runtime management

Inference goes through HTTP only. MCP is used for management, configuration, routing diagnostics, and runtime control.

## Current Status

The project currently includes:

- typed catalog, profile, preset, and alias models
- YAML-backed catalog storage
- SQLite-backed benchmark and route-event state
- hardware probing for CPU, NVIDIA dGPU, Windows video controllers, and Vulkan device selectors
- placement routing with reserve buffers and experimental iGPU gating
- lazy runtime launch, device-targeted Vulkan selection, and idle-unload support
- OpenAI and Anthropic request translation helpers
- FastMCP control-plane scaffolding
- pytest coverage for catalog, routing, and protocol translation paths

## Quick Start

1. Sync dependencies:

```powershell
uv sync
```

2. Start the HTTP compatibility server:

```powershell
uv run llama-orchestrator-http
```

3. Start the MCP control plane:

```powershell
uv run llama-orchestrator-mcp
```

## Catalog

The default catalog file is:

- `catalog/catalog.yaml`

Aliases are the client-facing model names used by OpenAI-style and Anthropic-style callers.

## Environment Variables

Optional settings:

- `LLAMA_ORCH_HOST`
- `LLAMA_ORCH_PORT`
- `LLAMA_ORCH_API_KEY`
- `LLAMA_ORCH_CATALOG_PATH`
- `LLAMA_ORCH_STATE_PATH`
- `LLAMA_ORCH_MIN_FREE_RAM`
- `LLAMA_ORCH_MIN_FREE_DGPU_VRAM`
- `LLAMA_ORCH_MIN_FREE_IGPU_RAM`
- `LLAMA_ORCH_MAX_LOADED`
- `LLAMA_ORCH_ALLOW_EXPERIMENTAL_IGPU`
- `LLAMA_ORCH_ALLOW_EXPERIMENTAL_MIXED`
- `LLAMA_SERVER_CPU`
- `LLAMA_SERVER_CUDA`
- `LLAMA_SERVER_VULKAN`
- `LLAMA_SERVER_SYCL`

## Validation

Run the test suite with:

```powershell
uv run pytest -q
```

## Notes

- `POST /v1/chat/completions`, `POST /v1/responses`, and `POST /v1/messages` are the tool-capable generation endpoints.
- `POST /v1/completions` stays text-only by design.
- iGPU and mixed dGPU+iGPU routing are intentionally marked experimental for now.
- Hardware inventory uses backend-agnostic ids such as `dgpu0` and `igpu0`, while also exposing backend selectors such as `cuda0` and `vulkan1` for explicit targeting and benchmarking.
- On this machine, experimental `Vulkan0` iGPU routing has been live-validated with the local `Qwen3.5-0.8B-UD-Q8_K_XL.gguf` model.
