from open_amplify_ai.utils import handle_upstream_error
from open_amplify_ai.middleware import ErrorLoggingMiddleware, DebugLoggingMiddleware
import base64
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, TypedDict

import requests
import dotenv
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

os.makedirs("logs", exist_ok=True)

# Setup basic logging per project rules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

AMPLIFY_BASE_URL = "https://prod-api.vanderbilt.ai"

app = FastAPI(title="Amplify AI OpenAI Compatible API")

app.add_middleware(DebugLoggingMiddleware)
app.add_middleware(ErrorLoggingMiddleware)

if os.getenv("AMPLIFY_DEBUG", "0").lower() in ("1", "true", "yes"):
    logger.setLevel(logging.DEBUG)
    logging.getLogger("open_amplify_ai.middleware").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "amplify-ai"


@dataclass
class ChatMessage:
    """Single chat message with role and content."""

    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    """Parsed OpenAI chat completion request body."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4000
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None


# Amplify request/response typed dicts ----------------------------------------


class AmplifyModelOption(TypedDict):
    """Model selector object for Amplify chat options."""

    id: str


class AmplifyChatOptions(TypedDict, total=False):
    """Options block inside an Amplify chat request."""

    model: AmplifyModelOption
    assistantId: str
    prompt: str


class AmplifyChatMessage(TypedDict):
    """Single message in an Amplify chat request."""

    role: str
    content: str


class AmplifyChatData(TypedDict, total=False):
    """Data block for Amplify chat request."""

    temperature: Optional[float]
    max_tokens: Optional[int]
    dataSources: List[str]
    messages: List[AmplifyChatMessage]
    options: AmplifyChatOptions


class AmplifyChatRequest(TypedDict):
    """Top-level Amplify chat request payload."""

    data: AmplifyChatData


class AmplifyFileUploadData(TypedDict, total=False):
    """Data block for Amplify file upload request."""

    type: str
    name: str
    knowledgeBase: str
    tags: List[str]
    data: Dict[str, Any]
    actions: List[Dict[str, Any]]


class AmplifyFileUploadRequest(TypedDict):
    """Top-level Amplify file upload request payload."""

    data: AmplifyFileUploadData


class AmplifyFilesQueryData(TypedDict, total=False):
    """Data block for Amplify files/query request."""

    pageSize: int
    forwardScan: bool
    sortIndex: str
    tags: List[str]
    pageKey: Optional[Dict[str, Any]]


class AmplifyFilesQueryRequest(TypedDict):
    """Top-level Amplify files/query request payload."""

    data: AmplifyFilesQueryData


class AmplifyAssistantCreateData(TypedDict, total=False):
    """Data block for Amplify assistant create/update request."""

    name: str
    description: str
    assistantId: str
    tags: List[str]
    instructions: str
    disclaimer: str
    dataSources: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]


class AmplifyAssistantCreateRequest(TypedDict):
    """Top-level Amplify assistant create request payload."""

    data: AmplifyAssistantCreateData


class AmplifyKeyData(TypedDict):
    """Generic {key} data block used for delete and download endpoints."""

    key: str


class AmplifyKeyRequest(TypedDict):
    """Top-level wrapper for key-based Amplify requests."""

    data: AmplifyKeyData


class AmplifyTagsData(TypedDict, total=False):
    """Data block for Amplify tag operations."""

    tags: List[str]
    tag: str
    id: str


class AmplifyTagsRequest(TypedDict):
    """Top-level wrapper for tag operation requests."""

    data: AmplifyTagsData


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def get_amplify_token() -> str:
    """Retrieve the API token from the environment."""
    token = os.getenv("AMPLIFY_AI_TOKEN")
    if not token:
        logger.error("AMPLIFY_AI_TOKEN not found in environment.")
        raise HTTPException(status_code=401, detail="Amplify AI token not configured")
    return token


def get_amplify_headers(token: str = Depends(get_amplify_token)) -> Dict[str, str]:
    """Generate headers with the authorization token required by Amplify API."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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
                if isinstance(content_delta, str) and (content_delta.strip().startswith('{"command"') or content_delta.strip().startswith('{"tool"')):
                    try:
                        parsed_content = json.loads(content_delta)
                        name = parsed_content.get("tool") or parsed_content.get("command")
                        if name:
                            tool_calls = [
                                {
                                    "index": 0,
                                    "id": f"call_{uuid.uuid4().hex[:12]}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(parsed_content.get("parameters", {}))
                                    }
                                }
                            ]
                            parsed_content_delta = None
                        else:
                            parsed_content_delta = content_delta
                    except json.JSONDecodeError:
                        parsed_content_delta = content_delta
                else:
                    parsed_content_delta = content_delta

                delta_obj = {"role": "assistant", "content": parsed_content_delta}
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


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from open_amplify_ai.routers import (
    models,
    chat,
    files,
    assistants,
    threads,
    vector_stores,
    stubs
)

app.include_router(models.router)
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(assistants.router)
app.include_router(threads.router)
app.include_router(vector_stores.router)
app.include_router(stubs.router)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(port: Optional[int] = None, debug: bool = False) -> None:
    """Start the Uvicorn server.

    Bind address and port are read from environment variables so that the
    NixOS systemd unit can configure them without requiring code changes:
      AMPLIFY_SERVER_HOST  - defaults to 0.0.0.0
      AMPLIFY_SERVER_PORT  - defaults to 8080
      
    CLI argument for port overrides the environment variable.
    """
    if debug:
        os.environ["AMPLIFY_DEBUG"] = "1"
        logger.setLevel(logging.DEBUG)
        logging.getLogger("open_amplify_ai.middleware").setLevel(logging.DEBUG)

    host = os.getenv("AMPLIFY_SERVER_HOST", "0.0.0.0")
    port = port or int(os.getenv("AMPLIFY_SERVER_PORT", "8080"))
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run("open_amplify_ai.server:app", host=host, port=port, reload=False)
