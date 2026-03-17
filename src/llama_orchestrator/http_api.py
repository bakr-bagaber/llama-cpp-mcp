"""FastAPI compatibility layer for OpenAI and Anthropic-style clients."""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .catalog import CatalogError, CatalogStore
from .hardware import HardwareProbe
from .models import RuntimeRecord
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
        yield
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
        return await _proxy_openai(runtime, "/v1/chat/completions", proxied_payload, bool(payload.get("stream")))

    @app.post("/v1/completions", dependencies=[Depends(require_api_key)])
    async def completions(payload: dict[str, Any]) -> Any:
        if any(key in payload for key in ("tools", "tool_choice", "parallel_tool_calls")):
            raise HTTPException(status_code=400, detail="The legacy /v1/completions endpoint does not support tool use.")
        runtime = await _ensure_runtime(payload["model"])
        return await _proxy_openai(runtime, "/v1/completions", payload, bool(payload.get("stream")))

    @app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
    async def embeddings(payload: dict[str, Any]) -> Any:
        runtime = await _ensure_runtime(payload["model"])
        return await _proxy_openai(runtime, "/v1/embeddings", payload, False)

    @app.post("/v1/responses", dependencies=[Depends(require_api_key)])
    async def responses(payload: dict[str, Any]) -> Any:
        runtime, chat_payload = await _prepare_openai_response(payload)
        response = await _proxy_openai(runtime, "/v1/chat/completions", chat_payload, False)
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
        response = await _proxy_openai(runtime, "/v1/chat/completions", chat_payload, bool(payload.get("stream")))
        if isinstance(response, JSONResponse):
            body = json.loads(response.body)
            return JSONResponse(_chat_completion_to_anthropic(body, payload))
        return response

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
        return runtime, dict(payload)

    async def _prepare_openai_response(payload: dict[str, Any]) -> tuple[RuntimeRecord, dict[str, Any]]:
        model = payload.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="The 'model' field is required.")
        runtime = await _ensure_runtime(model)
        input_items = payload.get("input", "")
        messages: list[dict[str, Any]] = []
        if isinstance(input_items, str):
            messages = [{"role": "user", "content": input_items}]
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, dict) and item.get("role"):
                    messages.append({"role": item["role"], "content": _flatten_content(item.get("content", ""))})
        else:
            raise HTTPException(status_code=400, detail="Unsupported 'input' payload for /v1/responses.")
        chat_payload = {
            "model": model,
            "messages": messages,
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_output_tokens"),
            "tools": payload.get("tools"),
            "tool_choice": payload.get("tool_choice"),
            "stream": bool(payload.get("stream")),
        }
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
        _compact_none(chat_payload)
        return runtime, chat_payload

    async def _proxy_openai(runtime: RuntimeRecord, path: str, payload: dict[str, Any], stream: bool) -> Any:
        try:
            if stream:
                stream_context = await runtime_manager.proxy_json(runtime, path, payload, stream=True)

                async def iterator():
                    async with stream_context as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line:
                                yield f"{line}\n"

                return StreamingResponse(iterator(), media_type="text/event-stream")

            response = await runtime_manager.proxy_json(runtime, path, payload, stream=False)
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return JSONResponse(response.json())
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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
            "output": output_blocks,
            "model": body.get("model"),
            "usage": body.get("usage", {}),
        }

    return app


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
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


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
    if choice in (None, "auto", "any"):
        return choice
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
