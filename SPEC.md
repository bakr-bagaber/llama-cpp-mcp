# Llama.cpp MCP Server Spec

Status: Draft v1

## Summary

This project will provide a local orchestration layer for `llama.cpp` with two separate surfaces:

- Control plane: an MCP server used only for model, profile, runtime, benchmark, and policy management
- Data plane: OpenAI-compatible and Anthropic-compatible HTTP APIs used for all inference traffic

The system will not expose general text generation through MCP. All chat, completion, embedding, and future inference requests will flow through the HTTP compatibility layer.

## Goals

- Route requests to local `llama.cpp` runtimes using named model aliases
- Support multiple loaded models at once within configurable RAM and VRAM reserve policies
- Load models lazily on first use and unload them after idle time or memory pressure
- Manage local models through MCP, including download, import, aliasing, profile creation, pinning, and benchmarking
- Support CPU, dGPU, and iGPU aware routing from the beginning
- Mark iGPU-only and mixed dGPU+iGPU placements as experimental until validated on real hardware
- Preserve compatibility with OpenAI-style and Anthropic-style clients by exposing stable model names and standard endpoints

## Non-Goals

- Reimplement `llama.cpp` inference internals
- Replace `llama-server` with a custom model server
- Promise universal heterogeneous multi-GPU behavior across all backends on day one
- Convert arbitrary model formats in v1
- Fine-tune or train models

## Product Surfaces

### 1. MCP Control Plane

The MCP server is the administration and observability interface. It manages local assets and runtime policy but does not serve inference.

Planned tool families:

- Model library management
- Profile and alias management
- Runtime management
- Policy management
- Benchmark and routing diagnostics
- Observability and health

### 2. Compatibility Data Plane

The HTTP API is the only inference surface exposed to clients. It should present both OpenAI-compatible and Anthropic-compatible surfaces over the same MCP server.

Initial OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

Planned compatibility shim:

- `POST /v1/responses`

The `/v1/responses` route should be implemented by the MCP server itself if specific clients need it. This should not depend on upstream `llama-server` supporting that path.

Initial Anthropic-compatible endpoints:

- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

Anthropic-compatible support is included because tools such as Claude Code expect Anthropic-style APIs. Streaming event normalization for `POST /v1/messages` is part of the compatibility layer.

For Claude-style coding clients, Anthropic tool-use compatibility should be treated as a primary requirement, not an optional extra.

## Tool Use Requirement

Tool use is a first-class requirement for all compatible generation endpoints.

Required tool-use support:

- OpenAI `POST /v1/chat/completions`
- OpenAI `POST /v1/responses` when enabled
- Anthropic `POST /v1/messages`

Non-tool endpoints:

- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/embeddings`

Legacy caveat:

- OpenAI `POST /v1/completions` is a legacy text completion endpoint and does not natively define tool use in the current official API. The MCP server should keep this endpoint text-only and return a clear compatibility error if tool-related fields are supplied.

The server must never silently ignore requested tool use on an endpoint that does not support it.

## High-Level Architecture

### Supervisor

A Python MCP server process owns:

- the MCP server
- the compatibility HTTP server
- the model catalog
- hardware inventory and resource probes
- routing logic
- `llama-server` subprocess lifecycle
- benchmark history and runtime state

### Backend Adapters

Each backend adapter describes how to run `llama-server` for a placement.

Expected adapter families:

- CPU
- CUDA
- Vulkan
- SYCL

Adapters define:

- executable discovery
- capability checks
- launch flags
- device targeting behavior
- telemetry collection
- failure classification

### Runtime Instances

Each loaded instance represents one running `llama-server` process with:

- one base model artifact
- one load profile
- one backend placement
- one port assignment
- one health state

Generation presets are request-time defaults and must not create separate loaded instances unless they imply load-time settings.

## Catalog Model

The catalog separates model identity from runtime tuning.

### Base Model

Represents the underlying GGUF artifact and static metadata.

Fields:

- `id`
- `display_name`
- `source`
- `local_path`
- `hf_repo`
- `hf_filename`
- `family`
- `quantization`
- `capabilities`
- `metadata`
- `size_bytes`
- `estimated_ram_bytes`
- `estimated_vram_bytes`

### Load Profile

Represents settings that can require a separate loaded runtime.

Examples:

- `context_size`
- `threads`
- `batch_size`
- `ubatch_size`
- `backend_policy`
- `gpu_layers`
- `tensor_split`
- `flash_attention`
- `cache_settings`
- `speculative_draft_model`
- `embedding_mode`
- `reranking_mode`

### Generation Preset

Represents request-time defaults.

Examples:

- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repeat_penalty`
- `presence_penalty`
- `frequency_penalty`
- `max_tokens`
- `stop`
- `json_mode`
- `grammar`
- `reasoning_mode`
- `reasoning_effort`

Reasoning or "thinking" is modeled as preset metadata, alias behavior, or model-specific request shaping. It is not assumed to be a universal engine flag across all models.

### Alias

Represents the user-facing `model` string used by OpenAI-compatible clients.

An alias resolves to:

- one base model
- one default load profile
- one default generation preset
- optional routing hints
- optional capability tags

Examples:

- `qwen-coder/precise`
- `qwen-coder/fast`
- `qwen-coder/long-context`
- `embedding/qwen3-8b`

## State Model

User-managed configuration should live in editable YAML.

Suggested file:

- `catalog/catalog.yaml`

Machine-managed state should live in a local SQLite database.

Suggested file:

- `state/mcp.db`

Database domains:

- downloaded artifacts
- resolved metadata
- benchmark runs
- runtime history
- route outcomes
- failure records
- pin state
- idle eviction history

## MCP Tool Surface

Planned MCP tools:

- `llama_list_models`
- `llama_download_model`
- `llama_import_model`
- `llama_remove_model`
- `llama_list_profiles`
- `llama_create_profile`
- `llama_update_profile`
- `llama_delete_profile`
- `llama_list_presets`
- `llama_create_preset`
- `llama_update_preset`
- `llama_delete_preset`
- `llama_list_aliases`
- `llama_create_alias`
- `llama_update_alias`
- `llama_delete_alias`
- `llama_get_hardware`
- `llama_get_runtime_status`
- `llama_load_alias`
- `llama_unload_alias`
- `llama_pin_alias`
- `llama_unpin_alias`
- `llama_set_memory_policy`
- `llama_get_memory_policy`
- `llama_run_benchmark`
- `llama_list_benchmarks`
- `llama_route_explain`
- `llama_route_simulate`
- `llama_list_logs`

MCP tools should return concise operational data and human-readable explanations. They should not return generated model content except for optional smoke-test diagnostics if explicitly added later.

## HTTP API Contract

### Model Naming

OpenAI-compatible and Anthropic-compatible clients choose a `model` string. That string maps to an alias in the local catalog.

### Request Flow

1. Client sends request to the MCP server
2. The MCP server resolves alias to base model, load profile, and generation preset
3. Router chooses placement and either reuses or launches a runtime
4. The MCP server translates the request into the selected upstream `llama-server` format
5. The MCP server normalizes the response back to the client

### Compatibility Strategy

The MCP server must normalize differences between client expectations and upstream `llama-server`.

This includes:

- streaming format normalization
- endpoint translation when needed
- alias metadata in `/v1/models`
- optional `/v1/responses` translation
- Anthropic `messages` payload normalization
- Anthropic streaming event normalization
- Anthropic-style error payloads and headers
- consistent error payloads

### OpenAI-Compatible Surface

Supported routes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- optional `POST /v1/responses`

Tool-use behavior:

- `POST /v1/chat/completions`: supported
- `POST /v1/responses`: supported when the route is enabled
- `POST /v1/completions`: not supported by protocol, return explicit error if tool fields are present
- `POST /v1/embeddings`: not applicable

### Anthropic-Compatible Surface

Supported routes:

- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

Tool-use behavior:

- `POST /v1/messages`: supported
- `POST /v1/messages/count_tokens`: counts tokens for tool-bearing message requests as well as plain text requests

Required request compatibility:

- `x-api-key`
- `anthropic-version`
- optional `anthropic-beta`

The server should accept Anthropic-style requests and translate them to the selected local runtime. If a feature is unsupported by the selected local model, the server should return an explicit Anthropic-style error with a clear explanation.

### Anthropic Feature Mapping

The compatibility layer should plan for:

- `system` prompt translation
- content block flattening or preservation depending on backend support
- tool schema translation and best-effort tool-use compatibility
- tool result turn-shaping for Anthropic-style conversations
- function-call and tool-call normalization for OpenAI-style conversations
- stop reason mapping
- usage token reporting
- streaming event translation

For v1, the compatibility target is strong support for Claude-style coding workflows used by local coding clients. That means text messages, token counting, streaming, and tool-use-compatible request and response shaping should all be considered in scope. Multimodal blocks, advanced prompt caching semantics, and full batch APIs can follow later.

## Routing and Placement

### Placement Types

Planned placements:

- `cpu_only`
- `dgpu_only`
- `igpu_only`
- `cpu_dgpu_hybrid`
- `cpu_igpu_hybrid`
- `dgpu_igpu_mixed`
- `same_backend_multi_gpu`

### Support Levels

- Stable:
  - `cpu_only`
  - `dgpu_only`
  - `cpu_dgpu_hybrid`
- Experimental:
  - `igpu_only`
  - `cpu_igpu_hybrid`
  - `dgpu_igpu_mixed`
  - `same_backend_multi_gpu`

The experimental label is required from the beginning. The router may surface these placements, but only after compatibility checks and with clear explanations.

### Routing Inputs

- alias
- request size and estimated context
- live free system RAM
- live per-device free memory
- configured reserve buffers
- backend availability
- model artifact size and metadata
- benchmark history
- warm runtime reuse opportunity
- concurrency and queue depth
- user policy overrides

### Routing Outputs

Each routing decision should produce:

- selected runtime or launch plan
- selected backend and placement
- reason summary
- fit estimate
- expected load or queue behavior
- fallback path if placement fails

### Routing Policy

The router should use a weighted score with hard feasibility filters.

Hard filters:

- missing executable or backend
- model does not fit after reserves
- requested capability unsupported
- placement explicitly disabled

Soft preferences:

- warm instance reuse
- better benchmark score
- lower expected launch latency
- lower pressure on shared memory
- user-selected backend preference

## Memory Policy

### Reserve Buffers

The system must preserve resources for other applications.

Core settings:

- `min_free_system_ram_bytes`
- `min_free_dgpu_vram_bytes`
- `min_free_igpu_shared_ram_bytes`
- `max_loaded_instances`
- `max_concurrent_requests_per_runtime`

### Pressure Levels

- Healthy: all reserves satisfied
- Soft pressure: new loads may trigger idle eviction
- Hard pressure: new loads rejected or queued unless a pinned runtime can be displaced by policy

### Eviction Policy

By default:

- pinned runtimes are protected
- active runtimes are protected
- idle runtimes compete by priority
- least recently used and lowest-value runtimes are evicted first

Eviction score inputs:

- idle duration
- pin state
- alias priority
- benchmark value
- load cost
- recent usage frequency

## Runtime Lifecycle

### Lazy Load

Models are loaded only when first requested or explicitly prewarmed.

### Warm Reuse

Compatible requests should reuse an existing runtime whenever possible.

### Idle Unload

Each runtime tracks `last_used_at`.

Settings:

- `idle_unload_seconds`
- `idle_scan_interval_seconds`

### Explicit Prewarm

Administrators can pin or prewarm aliases through MCP.

### Failure Recovery

If a runtime becomes unhealthy:

- stop routing new traffic to it
- capture reason and log context
- mark failure in state
- attempt restart within configured retry policy

## Benchmarking

The system should support both heuristic routing and benchmark-informed routing.

### Sources

- `llama-bench` results
- MCP-observed first token latency
- tokens per second
- peak memory during load
- peak memory during steady-state inference
- failure rate by placement

### Benchmark Keys

- base model
- quantization
- load profile
- backend
- placement
- driver or adapter signature
- executable signature

### Benchmark Usage

If benchmark data exists and is fresh, it should influence routing.

If benchmark data does not exist, the router should fall back to heuristic fit and warm reuse.

## Download and Import Management

Supported flows:

- import local GGUF
- download from Hugging Face by repo and filename
- remove local artifact
- re-scan model metadata

V1 should support GGUF acquisition and registration. Format conversion is out of scope.

## Observability

Planned observability data:

- loaded runtimes
- per-runtime health
- last-used timestamps
- queue depth
- route decisions
- load duration
- tokens per second
- eviction events
- benchmark history
- recent failures

The MCP server should expose compact diagnostics. The HTTP plane should expose conventional health endpoints if needed later.

## Security and Locality

Defaults:

- bind HTTP to `127.0.0.1`
- do not expose MCP over the network by default
- optional API key for HTTP
- no remote download side effects outside configured model directories

## Assumptions

- The MCP server is a thin manager around upstream `llama.cpp`
- Upstream `llama-server` is the underlying runtime
- `/v1/responses` compatibility may need to be implemented in the MCP layer
- Mixed dGPU+iGPU execution should be treated as experimental until validated

## Source Notes

This spec is informed by the current official `llama.cpp` documentation for:

- OpenAI-compatible `llama-server`
- supported backends including CUDA, Vulkan, and SYCL
- CPU+GPU hybrid inference
- `llama-bench`
- embeddings and reranking server modes

Relevant sources:

- https://github.com/ggml-org/llama.cpp
- https://github.com/ggml-org/llama.cpp/issues/14702
- https://docs.anthropic.com/en/api/messages
- https://docs.anthropic.com/en/docs/build-with-claude/token-counting
- https://docs.anthropic.com/en/api/models-list


