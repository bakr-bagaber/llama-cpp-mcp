"""FastAPI compatibility layer for OpenAI and Anthropic-style clients."""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .catalog import CatalogError, CatalogStore
from .hardware import HardwareProbe
from .models import ReasoningMode, RuntimeRecord
from .runtime import RuntimeLaunchError, RuntimeManager
from .settings import AppSettings
from .state import StateStore


def create_app(
    settings: AppSettings,
    catalog: CatalogStore,
    hardware_probe: HardwareProbe,
    runtime_manager: RuntimeManager,
    state: StateStore,
) -> FastAPI:
    """Create the FastAPI app."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await runtime_manager.start_idle_janitor()
        yield
        await runtime_manager.stop_idle_janitor()
        for runtime in list(runtime_manager.list_runtimes()):
            await runtime_manager.unload_runtime(runtime.runtime_key)

    app = FastAPI(title="Llama Orchestrator", lifespan=lifespan)
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.hardware_probe = hardware_probe
    app.state.runtime_manager = runtime_manager
    app.state.state_store = state

    async def require_api_key(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> None:
        if not settings.api_key:
            return
        bearer = authorization.removeprefix("Bearer ").strip() if authorization else None
        if x_api_key != settings.api_key and bearer != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key.")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "loaded_runtimes": len(runtime_manager.list_runtimes())}

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def list_models() -> dict[str, Any]:
        aliases = catalog.list_aliases()
        return {
            "object": "list",
            "data": [
                {
                    "id": alias.id,
                    "object": "model",
                    "owned_by": "llama-orchestrator",
                    "capabilities": [cap.value for cap in alias.capabilities],
                    "experimental": alias.experimental,
                }
                for alias in aliases
            ],
        }

    @app.get("/v1/models/{model_id}", dependencies=[Depends(require_api_key)])
    async def get_model(model_id: str) -> dict[str, Any]:
        alias, model, profile, preset = catalog.resolve_alias(model_id)
        return {
            "id": alias.id,
            "display_name": model.display_name,
            "type": "model",
            "capabilities": [cap.value for cap in alias.capabilities or model.capabilities],
            "backend_preference": (alias.backend_preference or profile.backend_preference).value,
            "experimental": alias.experimental,
            "preset": preset.id,
            "profile": profile.id,
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(payload: dict[str, Any]) -> Any:
        runtime, proxied_payload = await _prepare_openai_chat(payload)
        return await _proxy_openai(runtime, "/v1/chat/completions", proxied_payload, bool(payload.get("stream")), payload["model"])

    @app.post("/v1/completions", dependencies=[Depends(require_api_key)])
    async def completions(payload: dict[str, Any]) -> Any:
        if any(key in payload for key in ("tools", "tool_choice", "parallel_tool_calls")):
            raise HTTPException(status_code=400, detail="The legacy /v1/completions endpoint does not support tool use.")
        runtime = await _ensure_runtime(payload["model"])
        proxied_payload = _apply_preset_defaults(catalog, payload["model"], dict(payload))
        return await _proxy_openai(runtime, "/v1/completions", proxied_payload, bool(payload.get("stream")), payload["model"])

    @app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
    async def embeddings(payload: dict[str, Any]) -> Any:
        runtime = await _ensure_runtime(payload["model"])
        proxied_payload = _apply_preset_defaults(catalog, payload["model"], dict(payload))
        return await _proxy_openai(runtime, "/v1/embeddings", proxied_payload, False, payload["model"])

    @app.post("/v1/responses", dependencies=[Depends(require_api_key)])
    async def responses(payload: dict[str, Any]) -> Any:
        runtime, chat_payload = await _prepare_openai_response(payload)
        if payload.get("stream"):
            return _proxy_responses_stream(runtime, chat_payload, payload)
        response = await _proxy_openai(runtime, "/v1/chat/completions", chat_payload, False, payload["model"])
        if isinstance(response, JSONResponse):
            body = json.loads(response.body)
            return JSONResponse(_chat_completion_to_response(body))
        return response

    @app.post("/v1/messages", dependencies=[Depends(require_api_key)])
    async def anthropic_messages(
        payload: dict[str, Any],
        anthropic_version: str | None = Header(default=None),
        anthropic_beta: str | None = Header(default=None),
    ) -> Any:
        if not anthropic_version:
            raise HTTPException(status_code=400, detail="Anthropic requests must include the anthropic-version header.")
        runtime, chat_payload = await _prepare_anthropic_messages(payload, anthropic_beta)
        if payload.get("stream"):
            return _proxy_anthropic_stream(runtime, chat_payload, payload)
        response = await _proxy_openai(runtime, "/v1/chat/completions", chat_payload, False, payload["model"])
        body = json.loads(response.body) if isinstance(response, JSONResponse) else response
        return JSONResponse(_chat_completion_to_anthropic(body, payload))

    @app.post("/v1/messages/count_tokens", dependencies=[Depends(require_api_key)])
    async def count_tokens(payload: dict[str, Any]) -> dict[str, Any]:
        # This is an approximation for now. Once llama.cpp tokenizer metadata
        # is wired in, we can make this model-aware.
        serialized = json.dumps(
            {
                "system": payload.get("system"),
                "messages": payload.get("messages", []),
                "tools": payload.get("tools", []),
            },
            ensure_ascii=False,
        )
        return {"input_tokens": math.ceil(len(serialized) / 4)}

    async def _ensure_runtime(alias_id: str) -> RuntimeRecord:
        inventory = hardware_probe.collect()
        try:
            return await runtime_manager.ensure_runtime(alias_id, inventory=inventory)
        except CatalogError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeLaunchError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def _prepare_openai_chat(payload: dict[str, Any]) -> tuple[RuntimeRecord, dict[str, Any]]:
        if "model" not in payload:
            raise HTTPException(status_code=400, detail="The 'model' field is required.")
        runtime = await _ensure_runtime(payload["model"])
        return runtime, _apply_preset_defaults(catalog, payload["model"], dict(payload))

    async def _prepare_openai_response(payload: dict[str, Any]) -> tuple[RuntimeRecord, dict[str, Any]]:
        model = payload.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="The 'model' field is required.")
        runtime = await _ensure_runtime(model)
        messages = _responses_input_to_messages(payload.get("input", ""))
        instructions = payload.get("instructions")
        if instructions:
            messages.insert(0, {"role": "system", "content": _flatten_content(instructions)})
        chat_payload = {
            "model": model,
            "messages": messages,
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_output_tokens", payload.get("max_completion_tokens")),
            "tools": payload.get("tools"),
            "tool_choice": payload.get("tool_choice"),
            "stream": bool(payload.get("stream")),
        }
        chat_payload = _apply_preset_defaults(catalog, model, chat_payload)
        _compact_none(chat_payload)
        return runtime, chat_payload

    async def _prepare_anthropic_messages(payload: dict[str, Any], anthropic_beta: str | None) -> tuple[RuntimeRecord, dict[str, Any]]:
        model = payload.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="The 'model' field is required.")
        runtime = await _ensure_runtime(model)
        messages = _anthropic_messages_to_openai_messages(payload.get("messages", []))
        if payload.get("system"):
            messages.insert(0, {"role": "system", "content": _flatten_content(payload["system"])})
        chat_payload = {
            "model": model,
            "messages": messages,
            "tools": [_anthropic_tool_to_openai(tool) for tool in payload.get("tools", [])] or None,
            "tool_choice": _anthropic_tool_choice_to_openai(payload.get("tool_choice")),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "stream": bool(payload.get("stream")),
            "metadata": {"anthropic_beta": anthropic_beta} if anthropic_beta else None,
        }
        chat_payload = _apply_preset_defaults(catalog, model, chat_payload)
        _compact_none(chat_payload)
        return runtime, chat_payload

    async def _proxy_openai(runtime: RuntimeRecord, path: str, payload: dict[str, Any], stream: bool, requested_model: str) -> Any:
        try:
            if stream:
                async def iterator():
                    async with runtime_manager.stream_json(runtime, path, payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line:
                                yield f"{line}\n"

                return StreamingResponse(iterator(), media_type="text/event-stream")

            response = await runtime_manager.post_json(runtime, path, payload)
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            body = response.json()
            body["model"] = requested_model
            return JSONResponse(body)
        except httpx.HTTPError as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            raise HTTPException(status_code=502, detail=detail) from exc

    def _proxy_anthropic_stream(runtime: RuntimeRecord, payload: dict[str, Any], request_payload: dict[str, Any]) -> StreamingResponse:
        async def iterator() -> AsyncIterator[str]:
            try:
                async with runtime_manager.stream_json(runtime, "/v1/chat/completions", payload) as response:
                    response.raise_for_status()
                    async for event in _openai_stream_to_anthropic_events(response.aiter_lines(), request_payload):
                        yield event
            except httpx.HTTPError as exc:
                error_event = _anthropic_sse_event(
                    "error",
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": str(exc).strip() or exc.__class__.__name__},
                    },
                )
                yield error_event

        return StreamingResponse(iterator(), media_type="text/event-stream")

    def _proxy_responses_stream(runtime: RuntimeRecord, payload: dict[str, Any], request_payload: dict[str, Any]) -> StreamingResponse:
        async def iterator() -> AsyncIterator[str]:
            try:
                async with runtime_manager.stream_json(runtime, "/v1/chat/completions", payload) as response:
                    response.raise_for_status()
                    async for event in _openai_stream_to_responses_events(response.aiter_lines(), request_payload):
                        yield event
            except httpx.HTTPError as exc:
                yield _responses_sse_event(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "error": {"type": "server_error", "message": str(exc).strip() or exc.__class__.__name__},
                    },
                )

        return StreamingResponse(iterator(), media_type="text/event-stream")

    return app


def _chat_completion_to_response(body: dict[str, Any]) -> dict[str, Any]:
    message = ((body.get("choices") or [{}])[0]).get("message", {})
    output_blocks: list[dict[str, Any]] = []
    if message.get("content"):
        output_blocks.append({"type": "output_text", "text": message["content"]})
    for tool_call in message.get("tool_calls", []):
        output_blocks.append(
            {
                "type": "tool_call",
                "id": tool_call.get("id"),
                "name": tool_call.get("function", {}).get("name"),
                "arguments": tool_call.get("function", {}).get("arguments"),
            }
        )
    return {
        "id": body.get("id", "resp_local"),
        "object": "response",
        "status": "completed",
        "output": output_blocks,
        "model": body.get("model"),
        "usage": body.get("usage", {}),
    }


def _apply_preset_defaults(catalog: CatalogStore, model_alias: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Merge generation preset defaults without overriding explicit user input."""
    try:
        _alias, model, _profile, preset = catalog.resolve_alias(model_alias)
    except Exception:
        return payload

    field_mapping = {
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "min_p": preset.min_p,
        "repeat_penalty": preset.repeat_penalty,
        "presence_penalty": preset.presence_penalty,
        "frequency_penalty": preset.frequency_penalty,
        "max_tokens": preset.max_tokens,
        "stop": preset.stop or None,
        "grammar": preset.grammar,
    }
    for key, value in field_mapping.items():
        if key not in payload and value is not None:
            payload[key] = value
    for key, value in preset.request_overrides.items():
        payload.setdefault(key, value)
    _apply_reasoning_hint(model.family, preset.reasoning_mode, payload)
    return payload


def _apply_reasoning_hint(model_family: str | None, reasoning_mode: ReasoningMode, payload: dict[str, Any]) -> None:
    """Inject model-family-specific thinking controls.

    Qwen documents `/no_think` and `/think` for earlier Qwen3-family models.
    We use those hints as a best-effort bridge so aliases can express
    thinking policy even when the upstream API does not expose a direct flag.
    """
    family = (model_family or "").lower()
    if not family.startswith("qwen"):
        return

    # Qwen3.5 documents a hard `enable_thinking` switch rather than the
    # Qwen3 soft `/think` and `/no_think` controls. We attach the explicit
    # request fields first and only use prompt-level hints for older Qwen
    # families that support them.
    if family.startswith("qwen3.5") or family == "qwen3.5":
        if reasoning_mode is ReasoningMode.OFF:
            payload.setdefault("chat_template_kwargs", {})
            if isinstance(payload["chat_template_kwargs"], dict):
                payload["chat_template_kwargs"].setdefault("enable_thinking", False)
            payload.setdefault("enable_thinking", False)
        elif reasoning_mode in {ReasoningMode.LIGHT, ReasoningMode.DEEP, ReasoningMode.MODEL_NATIVE}:
            payload.setdefault("chat_template_kwargs", {})
            if isinstance(payload["chat_template_kwargs"], dict):
                payload["chat_template_kwargs"].setdefault("enable_thinking", True)
            payload.setdefault("enable_thinking", True)
        return

    if reasoning_mode is ReasoningMode.OFF:
        directive = "/no_think"
    elif reasoning_mode in {ReasoningMode.LIGHT, ReasoningMode.DEEP, ReasoningMode.MODEL_NATIVE}:
        directive = "/think"
    else:
        return

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return

    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        existing = str(messages[0].get("content", ""))
        if directive not in existing:
            messages[0]["content"] = f"{directive}\n{existing}".strip()
        return

    messages.insert(0, {"role": "system", "content": directive})


def _compact_none(payload: dict[str, Any]) -> None:
    for key in list(payload):
        if payload[key] is None:
            payload.pop(key)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                block_type = item.get("type")
                if block_type in {"text", "input_text"}:
                    parts.append(str(item.get("text", "")))
                elif block_type == "tool_result":
                    parts.append(str(item.get("content", "")))
                elif block_type in {"image", "input_image"}:
                    parts.append("[image omitted]")
                elif block_type in {"file", "input_file"}:
                    parts.append("[file omitted]")
                elif block_type == "output_text":
                    parts.append(str(item.get("text", "")))
                elif block_type == "tool_use":
                    parts.append(f"[tool_use {item.get('name', 'tool')}]")
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _responses_input_to_messages(input_items: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI Responses-style input into chat-completions messages."""
    if isinstance(input_items, str):
        return [{"role": "user", "content": input_items}]
    if not isinstance(input_items, list):
        raise HTTPException(status_code=400, detail="Unsupported 'input' payload for /v1/responses.")

    messages: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue
        if item.get("type") == "message":
            role = str(item.get("role", "user"))
            messages.append({"role": role, "content": _flatten_content(item.get("content", ""))})
            continue
        if item.get("role"):
            messages.append({"role": str(item["role"]), "content": _flatten_content(item.get("content", ""))})
            continue
        block_type = item.get("type")
        if block_type in {"input_text", "text"}:
            messages.append({"role": "user", "content": str(item.get("text", ""))})
            continue
        if block_type in {"input_image", "image", "input_file", "file"}:
            messages.append({"role": "user", "content": _flatten_content([item])})
            continue
        raise HTTPException(status_code=400, detail=f"Unsupported input item type '{block_type}' for /v1/responses.")
    return messages


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if choice in (None, "auto"):
        return choice
    if choice == "any":
        return "required"
    if choice == "none":
        return "none"
    if isinstance(choice, dict) and choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice.get("name")}}
    return choice


def _anthropic_messages_to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    openai_messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                pending_tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(block)

        if pending_tool_calls and role == "assistant":
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(part for part in text_parts if part) or None,
                    "tool_calls": pending_tool_calls,
                }
            )
            pending_tool_calls = []
        elif text_parts:
            openai_messages.append({"role": role, "content": "\n".join(text_parts)})

        for tool_result in tool_results:
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result.get("tool_use_id"),
                    "content": _flatten_content(tool_result.get("content", "")),
                }
            )
    return openai_messages


def _chat_completion_to_anthropic(body: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content_blocks: list[dict[str, Any]] = []
    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})
    for tool_call in message.get("tool_calls", []):
        arguments = tool_call.get("function", {}).get("arguments")
        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", "tool_call"),
                "name": tool_call.get("function", {}).get("name", "tool"),
                "input": parsed_arguments,
            }
        )
    finish_reason = choice.get("finish_reason")
    if any(block["type"] == "tool_use" for block in content_blocks):
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    usage = body.get("usage", {})
    return {
        "id": body.get("id", "msg_local"),
        "type": "message",
        "role": "assistant",
        "model": request_payload.get("model"),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def _openai_stream_to_anthropic_events(
    lines: AsyncIterator[str],
    request_payload: dict[str, Any],
) -> AsyncIterator[str]:
    """Translate OpenAI-style SSE frames into Anthropic SSE events."""
    message_started = False
    message_id = "msg_local"
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}
    text_block_index: int | None = None
    next_block_index = 0
    open_block_indices: set[int] = set()
    tool_block_indices: dict[int, int] = {}
    tool_state: dict[int, dict[str, Any]] = {}

    async for raw_line in lines:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if not message_started:
            message_started = True
            message_id = chunk.get("id", message_id)
            initial_usage = chunk.get("usage", {})
            usage["input_tokens"] = int(initial_usage.get("prompt_tokens", 0) or 0)
            yield _anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": request_payload.get("model"),
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage.copy(),
                    },
                },
            )

        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        chunk_usage = chunk.get("usage", {})
        if chunk_usage:
            usage["input_tokens"] = int(chunk_usage.get("prompt_tokens", usage["input_tokens"]) or 0)
            usage["output_tokens"] = int(chunk_usage.get("completion_tokens", usage["output_tokens"]) or 0)

        content_delta = delta.get("content")
        if content_delta:
            if text_block_index is None:
                text_block_index = next_block_index
                next_block_index += 1
                open_block_indices.add(text_block_index)
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": text_block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            yield _anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": content_delta},
                },
            )

        for tool_call in delta.get("tool_calls", []):
            openai_index = int(tool_call.get("index", len(tool_block_indices)))
            if openai_index not in tool_block_indices:
                block_index = next_block_index
                next_block_index += 1
                tool_block_indices[openai_index] = block_index
                open_block_indices.add(block_index)
                tool_state[openai_index] = {"id": None, "name": "", "arguments": ""}
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "tool_use", "id": "", "name": "", "input": {}},
                    },
                )

            state = tool_state[openai_index]
            if tool_call.get("id"):
                state["id"] = tool_call["id"]
            function = tool_call.get("function", {})
            if function.get("name"):
                state["name"] = function["name"]
            if function.get("arguments"):
                arguments_fragment = str(function["arguments"])
                state["arguments"] += arguments_fragment
                yield _anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tool_block_indices[openai_index],
                        "delta": {"type": "input_json_delta", "partial_json": arguments_fragment},
                    },
                )

        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason:
            stop_reason = "end_turn"

    if not message_started:
        yield _anthropic_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": request_payload.get("model"),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage.copy(),
                },
            },
        )

    for block_index in sorted(open_block_indices):
        yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})

    yield _anthropic_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]},
        },
    )
    yield _anthropic_sse_event("message_stop", {"type": "message_stop"})


def _anthropic_sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _openai_stream_to_responses_events(
    lines: AsyncIterator[str],
    request_payload: dict[str, Any],
) -> AsyncIterator[str]:
    """Translate chat-completions SSE into Responses-style SSE."""
    response_id = "resp_local"
    output_index = 0
    text_started = False
    text_accumulator = ""
    tool_calls: dict[int, dict[str, Any]] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    created = False

    async for raw_line in lines:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        response_id = chunk.get("id", response_id)
        chunk_usage = chunk.get("usage", {})
        if chunk_usage:
            usage["prompt_tokens"] = int(chunk_usage.get("prompt_tokens", usage["prompt_tokens"]) or 0)
            usage["completion_tokens"] = int(chunk_usage.get("completion_tokens", usage["completion_tokens"]) or 0)
            usage["total_tokens"] = int(chunk_usage.get("total_tokens", usage["total_tokens"]) or 0)
        if not created:
            created = True
            yield _responses_sse_event(
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "in_progress",
                        "model": request_payload.get("model"),
                        "output": [],
                    },
                },
            )

        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        content_delta = delta.get("content")
        if content_delta:
            if not text_started:
                text_started = True
                yield _responses_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {"type": "message", "role": "assistant", "content": []},
                    },
                )
                yield _responses_sse_event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    },
                )
            text_accumulator += content_delta
            yield _responses_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": content_delta,
                },
            )

        for tool_call in delta.get("tool_calls", []):
            tool_index = int(tool_call.get("index", len(tool_calls)))
            state = tool_calls.setdefault(
                tool_index,
                {"id": tool_call.get("id"), "name": "", "arguments": "", "output_index": output_index + len(tool_calls)},
            )
            function = tool_call.get("function", {})
            if function.get("name"):
                state["name"] = function["name"]
            if tool_call.get("id"):
                state["id"] = tool_call["id"]
            if "started" not in state:
                state["started"] = True
                yield _responses_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": state["output_index"],
                        "item": {
                            "type": "tool_call",
                            "id": state["id"] or f"tool_{tool_index}",
                            "name": state["name"],
                            "arguments": "",
                        },
                    },
                )
            if function.get("arguments"):
                fragment = str(function["arguments"])
                state["arguments"] += fragment
                yield _responses_sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": state["output_index"],
                        "delta": fragment,
                    },
                )

        if finish_reason:
            if text_started:
                yield _responses_sse_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text_accumulator,
                    },
                )
            for state in tool_calls.values():
                yield _responses_sse_event(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": state["output_index"],
                        "arguments": state["arguments"],
                    },
                )

    yield _responses_sse_event(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "completed",
                "model": request_payload.get("model"),
                "usage": usage,
            },
        },
    )


def _responses_sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
