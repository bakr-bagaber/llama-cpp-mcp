# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -q

# Run a single test file
uv run pytest tests/test_catalog.py -q

# Auto-generate catalog from models directory
uv run llama-mcp init-config

# Validate catalog
uv run llama-mcp validate-config

# Start HTTP server (OpenAI/Anthropic-compatible API)
uv run llama-mcp http

# Start MCP server (stdio, for Claude Desktop etc.)
uv run llama-mcp mcp
```

## Architecture

This is a local inference orchestration layer with two cooperating server processes:

- **HTTP data plane** (`http_api.py`): OpenAI-compatible and Anthropic-compatible inference endpoints on `:8080`. Clients send requests using standard model aliases; the server resolves, routes, and proxies to the appropriate `llama-server` subprocess.
- **MCP control plane** (`mcp_server.py`): FastMCP server on stdio with 30+ tools for catalog management, routing inspection, benchmarking, and policy control.

### Core pipeline

**Request flow (HTTP → inference):**
1. Client sends request with an alias name (e.g. `qwen-fast`)
2. `CatalogStore` resolves alias → base model + load profile + generation preset
3. `HardwareProbe` snapshots current CPU/RAM/GPU state
4. `Router` scores candidate placements (CPU-only, dGPU-only, hybrid, iGPU) and selects best
5. `RuntimeManager` returns a warm matching subprocess or launches a new `llama-server` instance
6. Request is proxied to the subprocess's local port; response is normalized to OpenAI or Anthropic format

### Key modules

| Module | Responsibility |
|---|---|
| `models.py` | All Pydantic data models: `BaseModelDefinition`, `LoadProfile`, `GenerationPreset`, `AliasDefinition`, `CatalogDocument`, `RoutingDecision`, etc. |
| `settings.py` | `AppSettings` loaded from `.env` / environment. Defines paths to backend `llama-server` executables (CPU, CUDA, Vulkan, SYCL). |
| `catalog.py` | `CatalogStore` — YAML persistence, validation, CRUD with dependency checks, alias resolution |
| `router.py` | `Router` — scores placement candidates using hardware feasibility, warm-reuse bonus (+100), and benchmark evidence |
| `runtime.py` | `RuntimeManager` — async subprocess lifecycle: lazy load, warm reuse, idle TTL unload, LRU eviction, pinning |
| `hardware.py` | `HardwareProbe` — CPU/RAM via psutil, NVIDIA via `nvidia-smi`, iGPU via WMI, backend binary detection |
| `state.py` | `StateStore` — SQLite for machine-managed state: benchmark records and route events |
| `http_api.py` | FastAPI app: `/v1/chat/completions`, `/v1/messages`, `/v1/embeddings`, `/v1/models`, etc. |
| `mcp_server.py` | FastMCP tools: catalog ops, runtime ops, route explain/simulate, benchmarks, downloads |
| `__main__.py` | CLI entry points; `init-config` auto-scan logic; startup validation |

### State separation

- **`catalog/catalog.yaml`** — human-editable; defines models, profiles, presets, aliases. Managed by `CatalogStore`.
- **`state/mcp.db`** — machine-managed SQLite; benchmarks and routing history. Never hand-edit.

### Catalog object model

Aliases are the only thing clients ever reference. They compose:
- **BaseModelDefinition** — path to `.gguf`, family, capabilities, memory estimates
- **LoadProfile** — context size, `gpu_layers`, `backend_preference`, idle timeout
- **GenerationPreset** — temperature, top_p, max_tokens, reasoning_mode

### Placement types

`PlacementKind` enum: `CPU_ONLY`, `DGPU_ONLY`, `IGPU_ONLY` (experimental), `HYBRID_DGPU_PRIMARY`, `HYBRID_IGPU_PRIMARY`. iGPU paths are gated by `SUPPORT_LEVEL = EXPERIMENTAL` and disabled by default.

### Testing

Tests use `pytest-asyncio` with `asyncio` backend only. A `sandbox_path` fixture (from `conftest.py`) provides isolated temporary directories under `test-workspace/` (not `/tmp`, to avoid Windows path issues).
