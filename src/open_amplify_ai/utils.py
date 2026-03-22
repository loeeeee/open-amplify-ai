"""Helper functions for Amplify API integration."""
import json
import logging
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import HTTPException

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.types import (
    AmplifyChatRequest,
    AmplifyFilesQueryData,
    AmplifyFilesQueryRequest,
)


def handle_upstream_error(logger: logging.Logger, e: Exception, context_msg: str) -> HTTPException:
    """Consistently log and wrap upstream HTTP errors as a 500 HTTPException."""
    error_detail = ""
    if hasattr(e, "response") and e.response is not None:
        try:
            error_detail = e.response.text
        except Exception:
            pass

    logger.error("Error during %s: %s - Response: %s", context_msg, e, error_detail)
    try:
        if hasattr(e, "request") and e.request:
            body = getattr(e.request, "content", None) or getattr(e.request, "body", None)
            if body:
                logger.error("Request body sent: %s", body)
    except Exception:
        pass

    return HTTPException(
        status_code=500,
        detail=f"Error communicating with Amplify AI: {e} - {error_detail}",
    )


def not_implemented(feature: str) -> HTTPException:
    """Return a 501 HTTPException for unimplemented features."""
    return HTTPException(
        status_code=501,
        detail=f"{feature} is not supported by the Amplify AI backend.",
    )


def _estimate_bytes(item: Dict[str, Any]) -> int:
    """Estimate bytes from totalTokens which can be an int or a dict."""
    tokens = item.get("totalTokens", 0)
    if isinstance(tokens, dict):
        tokens = tokens.get("gpt", next(iter(tokens.values())) if tokens else 0)
    elif not isinstance(tokens, (int, float)):
        tokens = 0
    return int(tokens) * 4


def amplify_item_to_openai_file(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map an Amplify file record to an OpenAI File object shape."""
    import datetime

    created_at = 0
    try:
        dt = datetime.datetime.fromisoformat(item.get("createdAt", ""))
        created_at = int(dt.timestamp())
    except Exception:
        pass
    return {
        "id": item.get("id", ""),
        "object": "file",
        "bytes": _estimate_bytes(item),
        "created_at": created_at,
        "filename": item.get("name", ""),
        "purpose": "assistants",
    }


def amplify_assistant_to_openai(assistant: Dict[str, Any]) -> Dict[str, Any]:
    """Map an Amplify assistant record to an OpenAI Assistant object shape."""
    import datetime

    created_at = 0
    try:
        dt = datetime.datetime.fromisoformat(assistant.get("createdAt", ""))
        created_at = int(dt.timestamp())
    except Exception:
        pass
    return {
        "id": assistant.get("assistantId", assistant.get("id", "")),
        "object": "assistant",
        "created_at": created_at,
        "name": assistant.get("name", ""),
        "description": assistant.get("description", None),
        "model": "amplify",
        "instructions": assistant.get("instructions", None),
        "tools": [],
        "file_ids": [ds.get("id", "") for ds in assistant.get("dataSources", [])],
        "metadata": {},
    }


async def query_amplify_files(
    headers: Dict[str, str],
    page_size: int = 100,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Query all Amplify files, optionally filtered by tags. Paginates automatically."""
    items: List[Dict[str, Any]] = []
    page_key: Optional[Dict[str, Any]] = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            query_data: AmplifyFilesQueryData = {
                "pageSize": page_size,
                "forwardScan": False,
                "sortIndex": "createdAt",
            }
            if tags:
                query_data["tags"] = tags
            if page_key:
                query_data["pageKey"] = page_key

            payload: AmplifyFilesQueryRequest = {"data": query_data}
            resp = await client.post(
                f"{AMPLIFY_BASE_URL}/files/query",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            batch = data.get("items", [])
            items.extend(batch)
            page_key = data.get("pageKey")
            if not page_key or len(batch) < page_size:
                break

    return items


async def stream_amplify_chat(
    amplify_request: AmplifyChatRequest,
    headers: Dict[str, str],
    model: str,
    completion_id: str,
    created: int,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """
    Stream an Amplify /chat response and yield OpenAI-format SSE chunks.

    Amplify sends newline-delimited lines. If a line starts with 'data: ',
    its content is the assistant text delta. A final [DONE] marker is emitted.
    Uses httpx async streaming so the event loop is not blocked during the
    upstream request.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{AMPLIFY_BASE_URL}/chat",
            headers=headers,
            json=amplify_request,
        ) as resp:
            resp.raise_for_status()
            async for line_str in resp.aiter_lines():
                if not line_str:
                    continue

                content_delta = ""
                if line_str.startswith("data: "):
                    payload_str = line_str[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        parsed = json.loads(payload_str)
                        content_delta = (
                            parsed.get("data", "")
                            or parsed.get("content", "")
                            or parsed.get("message", "")
                        )
                    except json.JSONDecodeError:
                        content_delta = payload_str
                else:
                    content_delta = line_str

                if not content_delta:
                    continue

                tool_calls = None
                parsed_content_delta = None

                if isinstance(content_delta, str) and "[Tool Call:" in content_delta:
                    try:
                        match = re.search(
                            r"\[Tool Call:\s*([^\]]+)\]\s*\n?Parameters:\s*(.+)",
                            content_delta,
                            re.DOTALL,
                        )
                        if match:
                            name = match.group(1).strip()
                            params_str = match.group(2).strip()
                            try:
                                params = json.loads(params_str, strict=False)
                            except json.JSONDecodeError:
                                params = {}
                            tool_calls = [
                                {
                                    "index": 0,
                                    "id": f"call_{uuid.uuid4().hex[:12]}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(params),
                                    },
                                }
                            ]
                            parsed_content_delta = None
                    except Exception:
                        pass

                if tool_calls is None and isinstance(content_delta, str) and (
                    '"tool"' in content_delta or '"command"' in content_delta
                ):
                    try:
                        match = re.search(r"(\{[\s\S]*\})", content_delta)
                        json_str = match.group(1) if match else content_delta
                        parsed_content = json.loads(json_str, strict=False)
                        name = parsed_content.get("tool") or parsed_content.get("command")
                        if name:
                            tool_calls = [
                                {
                                    "index": 0,
                                    "id": f"call_{uuid.uuid4().hex[:12]}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(
                                            parsed_content.get("parameters", {})
                                        ),
                                    },
                                }
                            ]
                            parsed_content_delta = None
                        else:
                            parsed_content_delta = content_delta
                    except Exception:
                        parsed_content_delta = content_delta

                if tool_calls is None and parsed_content_delta is None:
                    parsed_content_delta = content_delta

                delta_obj: Dict[str, Any] = {
                    "role": "assistant",
                    "content": parsed_content_delta,
                }
                if tool_calls is not None:
                    delta_obj["tool_calls"] = tool_calls

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": "",
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_obj,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"

    if include_usage:
        usage_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "",
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"
