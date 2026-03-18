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

1. Install dependencies:

```powershell
uv sync
```

2. Bootstrap a starter catalog if this is a fresh machine:

```powershell
uv run llama-orchestrator init-config
```

3. Validate the configuration before serving traffic:

```powershell
uv run llama-orchestrator validate-config
```

4. Start the HTTP compatibility server:

```powershell
uv run llama-orchestrator-http
```

5. Start the MCP control plane:

```powershell
uv run llama-orchestrator-mcp
```

You can also use the unified CLI:

```powershell
uv run llama-orchestrator http
uv run llama-orchestrator mcp
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
- `LLAMA_BENCH_CPU`
- `LLAMA_BENCH_CUDA`
- `LLAMA_BENCH_VULKAN`
- `LLAMA_BENCH_SYCL`

## Deployment Notes

- The package now includes a starter catalog template, so a clean install can bootstrap itself with `init-config`.
- `validate-config` fails fast on broken alias references and missing local model files, which is useful in startup scripts and service managers.
- The default executable discovery still looks for common Windows `llama.cpp` install paths, but explicit environment variables are preferred for portable deployments.

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
- Anthropic streaming requests are translated into Anthropic-style SSE events instead of leaking raw OpenAI-style chunks.
- OpenAI `POST /v1/responses` streaming is translated into `response.*` SSE events.
- Mixed Vulkan benchmark attempts are recorded conservatively: if `llama-bench` reports separate per-device rows instead of a combined run, the result is stored as unverified and does not bias routing.
- The MCP control plane now supports benchmark verification/deletion, route history inspection, and safe model deletion that refuses to remove models still used by aliases.
- The MCP control plane now also includes direct `get_*` lookup tools plus runtime and benchmark summary diagnostics for quicker operator troubleshooting.
- Profile and preset deletion are now dependency-aware like model deletion, and alias deletion unloads any live runtimes first.
- Startup now fails fast on broken catalog references or missing local model files, and MCP includes clone helpers for deriving new profiles and presets from existing ones.
- The compatibility layer now handles richer `POST /v1/responses` input shapes, including typed message items, `instructions`, and safe placeholders for non-text input blocks.
- On this machine, experimental `Vulkan0` iGPU routing has been live-validated with the local `Qwen3.5-0.8B-UD-Q8_K_XL.gguf` model.
