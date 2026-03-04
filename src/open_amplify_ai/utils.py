"""Helper functions for Amplify API integration."""
import json
import uuid
from typing import Any, Dict, Iterator, List, Optional

import requests
from fastapi import HTTPException

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.types import (
    AmplifyChatRequest,
    AmplifyFilesQueryData,
    AmplifyFilesQueryRequest,
)

def not_implemented(feature: str) -> HTTPException:
    """Return a 501 HTTPException for unimplemented features."""
    return HTTPException(status_code=501, detail=f"{feature} is not supported by the Amplify AI backend.")


def _estimate_bytes(item: Dict[str, Any]) -> int:
    """Estimate bytes from totalTokens which can be an int or a dict."""
    tokens = item.get("totalTokens", 0)
    if isinstance(tokens, dict):
        # average or max them, just pick gpt or first value
        tokens = tokens.get("gpt", next(iter(tokens.values())) if tokens else 0)
    elif not isinstance(tokens, (int, float)):
        tokens = 0
    return int(tokens) * 4


def amplify_item_to_openai_file(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map an Amplify file record to an OpenAI File object shape."""
    created_at = 0
    try:
        import datetime
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
    created_at = 0
    try:
        import datetime
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


def query_amplify_files(
    headers: Dict[str, str],
    page_size: int = 100,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Query all Amplify files, optionally filtered by tags. Paginates automatically."""
    items: List[Dict[str, Any]] = []
    page_key: Optional[Dict[str, Any]] = None

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
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/query",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        batch = data.get("items", [])
        items.extend(batch)
        page_key = data.get("pageKey")
        if not page_key or len(batch) < page_size:
            break

    return items


def stream_amplify_chat(
    amplify_request: AmplifyChatRequest,
    headers: Dict[str, str],
    model: str,
    completion_id: str,
    created: int,
    include_usage: bool = False,
) -> Iterator[str]:
    """
    Stream an Amplify /chat response and yield OpenAI-format SSE chunks.

    Amplify sends newline-delimited lines. If a line starts with 'data: ',
    its content is the assistant text delta. A final [DONE] marker is emitted.
    """
    with requests.post(
        f"{AMPLIFY_BASE_URL}/chat",
        headers=headers,
        json=amplify_request,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line_str = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            # Amplify may send plain text lines or data: prefixed lines
            content_delta = ""
            if line_str.startswith("data: "):
                payload_str = line_str[6:]
                if payload_str == "[DONE]":
                    break
                try:
                    parsed = json.loads(payload_str)
                    # Try common content paths
                    content_delta = (
                        parsed.get("data", "")
                        or parsed.get("content", "")
                        or parsed.get("message", "")
                    )
                except json.JSONDecodeError:
                    content_delta = payload_str
            else:
                content_delta = line_str

            if content_delta:
                # Try to parse content_delta as a tool call (kilo formatting)
                tool_calls = None
                parsed_content_delta = None
                
                # Check if it looks like a complete JSON tool call
                if isinstance(content_delta, str) and content_delta.strip().startswith('{"command"'):
                    try:
                        parsed_content = json.loads(content_delta)
                        tool_calls = [
                            {
                                "index": 0,
                                "id": f"call_{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": {
                                    "name": parsed_content.get("command"),
                                    "arguments": json.dumps(parsed_content.get("parameters", {}))
                                }
                            }
                        ]
                        parsed_content_delta = None
                    except json.JSONDecodeError:
                        parsed_content_delta = content_delta
                else:
                    parsed_content_delta = content_delta

                delta_obj = {"role": "assistant"}
                if parsed_content_delta is not None:
                    delta_obj["content"] = parsed_content_delta
                elif tool_calls is not None:
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

    # Final chunk with finish_reason
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
