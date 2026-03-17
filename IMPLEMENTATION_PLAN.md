# Implementation Plan

## Delivery Approach

Build the project as a Python 3.12 service with two cooperating servers in one process:

- an MCP control plane using FastMCP
- a compatibility HTTP plane using FastAPI for OpenAI-style and Anthropic-style clients

The plan favors a strong foundation first: catalog, routing, lifecycle, and compatibility layers before aggressive optimization.

## Scope

- In:
  - MCP management tools
  - OpenAI-compatible inference gateway
  - Anthropic-compatible inference gateway
  - multi-model runtime management
  - configurable reserve buffers
  - CPU and dGPU stable routing
  - iGPU and mixed GPU experimental routing from the first version
  - benchmark-aware placement
- Out:
  - custom inference backend
  - model conversion pipeline
  - distributed multi-host scheduling
  - non-local-first deployment concerns in v1

## Milestone 0: Project Skeleton

Deliverables:

- `pyproject.toml`
- package layout
- configuration loading
- logging setup
- typed settings model

Acceptance:

- service boots with empty catalog
- MCP server and HTTP server start cleanly in local mode

## Milestone 1: Catalog and Alias Model

Deliverables:

- YAML catalog schema
- model, load profile, preset, and alias models
- validation rules
- CRUD logic used by MCP tools

Acceptance:

- aliases resolve deterministically
- invalid profiles fail with actionable messages

## Milestone 2: Hardware Inventory and Policy Engine

Deliverables:

- CPU and RAM probe
- dGPU probe
- iGPU probe
- backend executable discovery
- reserve buffer policy engine

Acceptance:

- hardware inventory returns stable structured data
- policy engine can simulate fit for a given alias and placement

Notes:

- dGPU and iGPU paths exist from the beginning
- iGPU and mixed placements are labeled experimental in outputs

## Milestone 3: Runtime Lifecycle Manager

Deliverables:

- `llama-server` process launcher
- health checks
- port allocation
- warm runtime registry
- TTL unload
- pin and prewarm support
- eviction engine

Acceptance:

- first request launches a runtime
- subsequent compatible requests reuse it
- idle runtime unloads on schedule
- new load honors reserve buffers

## Milestone 4: Compatibility Gateway

Deliverables:

- `/v1/models`
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `GET /v1/models/{model_id}`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- Anthropic tool-use request and response translation
- OpenAI chat tool-call request and response translation
- explicit compatibility errors for tool fields on unsupported endpoints
- streaming support
- response normalization
- error normalization

Acceptance:

- client sees aliases as models
- requests route through the selected runtime
- errors are stable and easy to act on
- Anthropic-style clients can send `messages` requests successfully
- Anthropic-style tool definitions can be accepted and mapped into local-model prompting or structured decoding paths
- OpenAI-style chat clients can send tools and receive tool-call-compatible responses
- legacy `/v1/completions` rejects tool-bearing requests with a clear documented error
- token counting works with local tokenizer metadata or a documented approximation path

## Milestone 5: MCP Management Surface

Deliverables:

- model download and import tools
- profile and preset CRUD tools
- alias CRUD tools
- memory policy tools
- runtime status tools
- route explain and simulate tools

Acceptance:

- no inference is required through MCP
- all catalog and runtime operations are available through MCP

## Milestone 6: Benchmarking and Smarter Routing

Deliverables:

- `llama-bench` integration
- benchmark state storage
- routing score updates based on measured performance
- route fallback logic

Acceptance:

- routing can operate without benchmarks
- routing improves when benchmark data exists
- benchmark freshness is tracked

## Milestone 7: Experimental GPU Paths

Deliverables:

- iGPU-only placement path
- CPU+iGPU hybrid placement path
- mixed dGPU+iGPU placement path behind explicit experimental gating
- same-backend multi-GPU exploration hooks

Acceptance:

- experimental paths are discoverable
- experimental paths are never chosen silently when disabled
- route explanations clearly state the experimental status

## Milestone 8: Validation and Hardening

Deliverables:

- unit tests for catalog validation
- unit tests for routing decisions
- unit tests for reserve buffer behavior
- integration tests with mocked subprocesses
- documentation for setup and client integration

Acceptance:

- stable paths covered by automated tests
- documentation covers MCP setup plus OpenAI-compatible and Anthropic-compatible usage

## Recommended Initial File Layout

- `src/llama_orchestrator/__init__.py`
- `src/llama_orchestrator/settings.py`
- `src/llama_orchestrator/catalog.py`
- `src/llama_orchestrator/models.py`
- `src/llama_orchestrator/state.py`
- `src/llama_orchestrator/router.py`
- `src/llama_orchestrator/policy.py`
- `src/llama_orchestrator/runtime.py`
- `src/llama_orchestrator/backends/`
- `src/llama_orchestrator/http_api.py`
- `src/llama_orchestrator/mcp_server.py`
- `src/llama_orchestrator/benchmarks.py`
- `catalog/catalog.yaml`
- `tests/`

## Validation Matrix

Test at minimum:

- no GPU available
- dGPU available
- iGPU available
- dGPU and iGPU both available
- low-memory environment
- multiple warm models
- pinned runtime under pressure
- benchmark-aware route selection

## Risks

- upstream `llama-server` endpoint differences may require compatibility translation
- Anthropic streaming and tool-use semantics may require orchestrator-side translation rather than simple proxying
- Windows iGPU telemetry may vary by vendor and driver
- heterogeneous mixed GPU routing may be technically possible but not consistently beneficial
- model metadata may be incomplete, requiring heuristic estimates before first load

## Exit Criteria for First Usable Release

- an OpenAI-compatible client can list models and chat using alias names
- an Anthropic-compatible client can list models and call `POST /v1/messages` using alias names
- MCP can download a model, create a preset, create an alias, and inspect runtime state
- at least two models can be kept warm simultaneously under configured reserve buffers
- idle eviction works
- routing chooses between CPU and dGPU on a real machine
- iGPU and mixed placements are visible and explicitly experimental
