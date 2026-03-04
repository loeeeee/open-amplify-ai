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


# ---------------------------------------------------------------------------
# Models endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models(headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """Convert Amplify GET /available_models to OpenAI GET /v1/models."""
    logger.info("Listing available models")
    try:
        response = requests.get(f"{AMPLIFY_BASE_URL}/available_models", headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise HTTPException(status_code=500, detail="Failed to fetch models from Amplify AI")

        amplify_models = data.get("data", {}).get("models", [])
        models = [ModelInfo(id=m.get("id")) for m in amplify_models]

        return {
            "object": "list",
            "data": [
                {
                    "id": m.id,
                    "object": m.object,
                    "created": m.created,
                    "owned_by": m.owned_by,
                }
                for m in models
            ],
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching models: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.get("/v1/models/{model}")
async def retrieve_model(model: str, headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    Retrieve a single model by ID.

    Amplify has no per-model endpoint, so this fetches the full list and filters.
    """
    logger.info("Retrieving model: %s", model)
    try:
        response = requests.get(f"{AMPLIFY_BASE_URL}/available_models", headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise HTTPException(status_code=500, detail="Failed to fetch models from Amplify AI")

        amplify_models = data.get("data", {}).get("models", [])
        match = next((m for m in amplify_models if m.get("id") == model), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

        info = ModelInfo(id=match.get("id"))
        return {
            "id": info.id,
            "object": info.object,
            "created": info.created,
            "owned_by": info.owned_by,
        }
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching model %s: %s", model, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.delete("/v1/models/{model}")
async def delete_model(model: str) -> Dict[str, Any]:
    """
    Amplify does not support model deletion.

    Returns 405 Method Not Allowed per the mapping document.
    """
    logger.info("Attempted deletion of model %s (not supported)", model)
    raise HTTPException(status_code=405, detail="Model deletion is not supported by Amplify AI.")


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request, headers: dict = Depends(get_amplify_headers)
) -> Any:
    """
    Convert OpenAI POST /v1/chat/completions to Amplify POST /chat.

    Supports both streaming (SSE) and non-streaming responses.
    """
    try:
        req_json = await request.json()
        
        parsed_messages = []
        for m in req_json.get("messages", []):
            role = m.get("role", "user")
            content_raw = m.get("content", "")
            if isinstance(content_raw, list):
                content_text = ""
                for part in content_raw:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_text += part.get("text", "")
                    elif isinstance(part, str):
                        content_text += part
                content = content_text
            else:
                content = str(content_raw)
            parsed_messages.append(ChatMessage(role=role, content=content))
            
        chat_request = ChatCompletionRequest(
            model=req_json.get("model", ""),
            messages=parsed_messages,
            temperature=req_json.get("temperature", 0.7),
            max_tokens=req_json.get("max_tokens", 4000),
            stream=req_json.get("stream", False),
            stream_options=req_json.get("stream_options"),
        )
    except Exception as e:
        logger.error("Invalid request format: %s", e)
        raise HTTPException(status_code=400, detail="Invalid request format")

    logger.info("Creating chat completion with model %s (stream=%s)", chat_request.model, chat_request.stream)

    amplify_request: AmplifyChatRequest = {
        "data": {
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
            "dataSources": [],
            "messages": [{"role": m.role, "content": m.content} for m in chat_request.messages],
            "options": {
                "model": {"id": chat_request.model},
            },
        }
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if chat_request.stream:
        logger.info("Streaming response requested for model %s", chat_request.model)
        
        include_usage = False
        if chat_request.stream_options and chat_request.stream_options.get("include_usage"):
            include_usage = True
            
        return StreamingResponse(
            stream_amplify_chat(
                amplify_request=amplify_request,
                headers=headers,
                model=chat_request.model,
                completion_id=completion_id,
                created=created,
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
        )

    try:
        response = requests.post(
            f"{AMPLIFY_BASE_URL}/chat",
            headers=headers,
            json=amplify_request,
            timeout=120,
        )
        response.raise_for_status()
        try:
            data = response.json()
            content = data.get("data", "")
        except Exception:
            content = response.text
            
        # Try to parse content as a tool call (kilo formatting)
        tool_calls = None
        if isinstance(content, str) and content.strip().startswith('{"command"'):
            try:
                parsed_content = json.loads(content)
                tool_calls = [
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": parsed_content.get("command"),
                            "arguments": json.dumps(parsed_content.get("parameters", {}))
                        }
                    }
                ]
                content = None
            except json.JSONDecodeError:
                pass

        message_obj = {"role": "assistant"}
        if content is not None:
            message_obj["content"] = content
        if tool_calls is not None:
            message_obj["tool_calls"] = tool_calls

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": chat_request.model,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error during chat completion: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


# ---------------------------------------------------------------------------
# Files endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/files")
async def list_files(headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    List all files via Amplify POST /files/query.

    Maps Amplify file records to the OpenAI File object shape.
    """
    logger.info("Listing files")
    try:
        items = query_amplify_files(headers)
        return {
            "object": "list",
            "data": [amplify_item_to_openai_file(item) for item in items],
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error listing files: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
    headers: dict = Depends(get_amplify_headers),
) -> Dict[str, Any]:
    """
    Upload a file using Amplify's two-step process.

    Step 1: POST /files/upload to get a pre-signed S3 URL.
    Step 2: PUT the file binary to that URL.
    Returns an OpenAI File object using the Amplify key as the ID.
    """
    logger.info("Uploading file: %s (purpose=%s)", file.filename, purpose)
    content_type = file.content_type or "application/octet-stream"
    file_bytes = await file.read()

    upload_payload: AmplifyFileUploadRequest = {
        "data": {
            "type": content_type,
            "name": file.filename or "upload",
            "knowledgeBase": "default",
            "tags": [],
            "data": {},
        }
    }

    try:
        init_resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/upload",
            headers=headers,
            json=upload_payload,
            timeout=30,
        )
        init_resp.raise_for_status()
        init_data = init_resp.json()

        if not init_data.get("success"):
            raise HTTPException(status_code=500, detail="Amplify file upload init failed")

        upload_url: str = init_data["uploadUrl"]
        file_key: str = init_data["key"]

        # PUT the binary directly to S3 (no auth header, pre-signed URL)
        s3_resp = requests.put(
            upload_url,
            data=file_bytes,
            headers={"Content-Type": content_type},
            timeout=60,
        )
        s3_resp.raise_for_status()

        return {
            "id": file_key,
            "object": "file",
            "bytes": len(file_bytes),
            "created_at": int(time.time()),
            "filename": file.filename or "upload",
            "purpose": purpose,
        }
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error uploading file: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.get("/v1/files/{file_id:path}/content")
async def retrieve_file_content(
    file_id: str, headers: dict = Depends(get_amplify_headers)
) -> Any:
    """
    Download file content via Amplify POST /assistant/files/download/codeinterpreter.

    Only supported for Code Interpreter files. Returns a redirect or raw bytes.
    This route MUST be registered before the bare GET /v1/files/{file_id:path} route
    to prevent the greedy :path wildcard from swallowing the /content suffix.
    """
    logger.info("Retrieving file content: %s", file_id)
    payload: AmplifyKeyRequest = {"data": {"key": file_id}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/files/download/codeinterpreter",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        download_url = data.get("downloadUrl", "")
        if not download_url:
            raise HTTPException(status_code=404, detail="File content not available or file not found")
        # Proxy the actual binary content
        content_resp = requests.get(download_url, timeout=60)
        content_resp.raise_for_status()
        return StreamingResponse(
            iter([content_resp.content]),
            media_type=content_resp.headers.get("Content-Type", "application/octet-stream"),
        )
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error retrieving file content %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.get("/v1/files/{file_id:path}")
async def retrieve_file(file_id: str, headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    Retrieve a single file record by ID.

    Amplify has no per-file endpoint. Fetches the full list and filters by id.
    """
    logger.info("Retrieving file: %s", file_id)
    try:
        items = query_amplify_files(headers)
        match = next((item for item in items if item.get("id") == file_id), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
        return amplify_item_to_openai_file(match)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error retrieving file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")



@app.delete("/v1/files/{file_id:path}")
async def delete_file(file_id: str, headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    Translate OpenAI DELETE /v1/files/{file_id} to Amplify POST /files op=/delete.

    The Amplify file delete uses a generic action dispatch pattern observed in the web UI:
      POST /files
      body: {
        "data": base64({"key": file_id}),
        "method": "POST",
        "op": "/delete",
        "path": "/files",
        "service": "file"
      }

    The file_id must be the Amplify key returned by POST /files/upload,
    e.g. "email@vanderbilt.edu/2024-07-15/uuid.json".
    """
    logger.info("Deleting file: %s", file_id)
    encoded_data = base64.b64encode(json.dumps({"key": file_id}).encode()).decode()
    amplify_body = {
        "data": encoded_data,
        "method": "POST",
        "op": "/delete",
        "path": "/files",
        "service": "file",
    }
    try:
        response = requests.post(
            f"{AMPLIFY_BASE_URL}/files",
            headers=headers,
            json=amplify_body,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        deleted = bool(data.get("success", False))
        return {
            "id": file_id,
            "object": "file",
            "deleted": deleted,
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error deleting file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


# ---------------------------------------------------------------------------
# Assistants endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/assistants")
async def list_assistants(headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """List all assistants via Amplify GET /assistant/list."""
    logger.info("Listing assistants")
    try:
        resp = requests.get(f"{AMPLIFY_BASE_URL}/assistant/list", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        assistants = data.get("data", [])
        openai_assistants = [amplify_assistant_to_openai(a) for a in assistants]
        return {
            "object": "list",
            "data": openai_assistants,
            "first_id": openai_assistants[0]["id"] if openai_assistants else None,
            "last_id": openai_assistants[-1]["id"] if openai_assistants else None,
            "has_more": False,
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error listing assistants: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.post("/v1/assistants")
async def create_assistant(
    request: Request, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Create a new assistant via Amplify POST /assistant/create.

    Maps OpenAI assistant fields to Amplify assistant fields.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    logger.info("Creating assistant: %s", body.get("name", "<unnamed>"))

    amplify_payload: AmplifyAssistantCreateRequest = {
        "data": {
            "name": body.get("name", ""),
            "description": body.get("description", ""),
            "tags": body.get("metadata", {}).get("tags", []) if body.get("metadata") else [],
            "instructions": body.get("instructions", ""),
            "dataSources": [],
            "tools": body.get("tools", []),
        }
    }

    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/create",
            headers=headers,
            json=amplify_payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("data", {})
        return {
            "id": result.get("assistantId", result.get("id", "")),
            "object": "assistant",
            "created_at": int(time.time()),
            "name": body.get("name", ""),
            "description": body.get("description", None),
            "model": body.get("model", "amplify"),
            "instructions": body.get("instructions", None),
            "tools": body.get("tools", []),
            "file_ids": [],
            "metadata": body.get("metadata", {}),
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error creating assistant: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.get("/v1/assistants/{assistant_id:path}")
async def retrieve_assistant(
    assistant_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Retrieve a single assistant by ID.

    Amplify has no per-assistant endpoint; fetches the full list and filters.
    """
    logger.info("Retrieving assistant: %s", assistant_id)
    try:
        resp = requests.get(f"{AMPLIFY_BASE_URL}/assistant/list", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        assistants = data.get("data", [])
        match = next(
            (a for a in assistants if a.get("assistantId") == assistant_id or a.get("id") == assistant_id),
            None,
        )
        if not match:
            raise HTTPException(status_code=404, detail=f"Assistant '{assistant_id}' not found")
        return amplify_assistant_to_openai(match)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error retrieving assistant %s: %s", assistant_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.post("/v1/assistants/{assistant_id:path}")
async def modify_assistant(
    assistant_id: str,
    request: Request,
    headers: dict = Depends(get_amplify_headers),
) -> Dict[str, Any]:
    """
    Modify an existing assistant via Amplify POST /assistant/create (upsert).

    Passing assistantId in the body triggers an update.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    logger.info("Modifying assistant: %s", assistant_id)

    amplify_payload: AmplifyAssistantCreateRequest = {
        "data": {
            "name": body.get("name", ""),
            "description": body.get("description", ""),
            "assistantId": assistant_id,
            "tags": body.get("metadata", {}).get("tags", []) if body.get("metadata") else [],
            "instructions": body.get("instructions", ""),
            "dataSources": [],
            "tools": body.get("tools", []),
        }
    }

    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/create",
            headers=headers,
            json=amplify_payload,
            timeout=30,
        )
        resp.raise_for_status()
        return {
            "id": assistant_id,
            "object": "assistant",
            "created_at": int(time.time()),
            "name": body.get("name", ""),
            "description": body.get("description", None),
            "model": body.get("model", "amplify"),
            "instructions": body.get("instructions", None),
            "tools": body.get("tools", []),
            "file_ids": [],
            "metadata": body.get("metadata", {}),
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error modifying assistant %s: %s", assistant_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.delete("/v1/assistants/{assistant_id:path}")
async def delete_assistant(
    assistant_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """Delete an assistant via Amplify POST /assistant/delete."""
    logger.info("Deleting assistant: %s", assistant_id)
    payload: AmplifyAssistantCreateRequest = {"data": {"assistantId": assistant_id}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/delete",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": assistant_id,
            "object": "assistant.deleted",
            "deleted": bool(data.get("success", False)),
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error deleting assistant %s: %s", assistant_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


# ---------------------------------------------------------------------------
# Threads endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/threads")
async def create_thread(request: Request) -> None:
    """Amplify has no standalone thread creation endpoint."""
    raise not_implemented("Thread creation")


@app.get("/v1/threads/{thread_id}")
async def retrieve_thread(thread_id: str) -> None:
    """Amplify has no thread retrieval endpoint."""
    raise not_implemented("Thread retrieval")


@app.post("/v1/threads/{thread_id}")
async def modify_thread(thread_id: str, request: Request) -> None:
    """Amplify has no thread modification endpoint."""
    raise not_implemented("Thread modification")


@app.delete("/v1/threads/{thread_id:path}")
async def delete_thread(
    thread_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Delete an Amplify thread via DELETE /assistant/openai/thread/delete.

    The thread_id is passed as a query parameter to the Amplify endpoint.
    """
    logger.info("Deleting thread: %s", thread_id)
    try:
        resp = requests.delete(
            f"{AMPLIFY_BASE_URL}/assistant/openai/thread/delete",
            headers=headers,
            params={"threadId": thread_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": thread_id,
            "object": "thread.deleted",
            "deleted": bool(data.get("success", False)),
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error deleting thread %s: %s", thread_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


# ---------------------------------------------------------------------------
# Messages endpoints (501 stubs)
# ---------------------------------------------------------------------------


@app.post("/v1/threads/{thread_id}/messages")
async def create_message(thread_id: str, request: Request) -> None:
    """Amplify has no standalone message creation endpoint."""
    raise not_implemented("Message creation")


@app.get("/v1/threads/{thread_id}/messages")
async def list_messages(thread_id: str) -> None:
    """Amplify has no thread message history endpoint."""
    raise not_implemented("Message listing")


@app.get("/v1/threads/{thread_id}/messages/{message_id}")
async def retrieve_message(thread_id: str, message_id: str) -> None:
    """Amplify has no single message retrieval endpoint."""
    raise not_implemented("Message retrieval")


# ---------------------------------------------------------------------------
# Runs endpoints (501 stubs)
# ---------------------------------------------------------------------------


@app.post("/v1/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request) -> None:
    """Amplify run model is synchronous and does not match the OpenAI async run API."""
    raise not_implemented("Run creation via threads")


@app.get("/v1/threads/{thread_id}/runs/{run_id}")
async def retrieve_run(thread_id: str, run_id: str) -> None:
    """Amplify has no async run status endpoint."""
    raise not_implemented("Run retrieval")


@app.post("/v1/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> None:
    """Amplify is synchronous; run cancellation is not supported."""
    raise not_implemented("Run cancellation")


@app.get("/v1/threads/{thread_id}/runs")
async def list_runs(thread_id: str) -> None:
    """Amplify has no run history endpoint."""
    raise not_implemented("Run listing")


@app.post("/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs")
async def submit_tool_outputs(thread_id: str, run_id: str, request: Request) -> None:
    """Amplify does not support tool-calling pause-and-resume run semantics."""
    raise not_implemented("Tool output submission")


@app.post("/v1/threads/runs")
async def create_thread_and_run(request: Request) -> None:
    """Amplify has no combined create-thread-and-run endpoint."""
    raise not_implemented("Create thread and run")


# ---------------------------------------------------------------------------
# Run steps endpoints (501 stubs)
# ---------------------------------------------------------------------------


@app.get("/v1/threads/{thread_id}/runs/{run_id}/steps")
async def list_run_steps(thread_id: str, run_id: str) -> None:
    """Amplify does not expose run step details."""
    raise not_implemented("Run steps listing")


@app.get("/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}")
async def retrieve_run_step(thread_id: str, run_id: str, step_id: str) -> None:
    """Amplify does not expose individual run step details."""
    raise not_implemented("Run step retrieval")


# ---------------------------------------------------------------------------
# Embeddings (501 stub)
# ---------------------------------------------------------------------------


@app.post("/v1/embeddings")
async def create_embedding(request: Request) -> None:
    """
    Amplify has no raw embedding vector endpoint.

    POST /embedding-dual-retrieval returns ranked document snippets, not float vectors.
    """
    raise not_implemented("Embedding vector generation")


# ---------------------------------------------------------------------------
# Audio endpoints (501 stubs)
# ---------------------------------------------------------------------------


@app.post("/v1/audio/speech")
async def create_speech(request: Request) -> None:
    """Amplify has no text-to-speech capability."""
    raise not_implemented("Audio speech synthesis")


@app.post("/v1/audio/transcriptions")
async def create_transcription(request: Request) -> None:
    """Amplify has no audio transcription capability."""
    raise not_implemented("Audio transcription")


@app.post("/v1/audio/translations")
async def create_translation(request: Request) -> None:
    """Amplify has no audio translation capability."""
    raise not_implemented("Audio translation")


# ---------------------------------------------------------------------------
# Images endpoints (501 stubs)
# ---------------------------------------------------------------------------


@app.post("/v1/images/generations")
async def create_image(request: Request) -> None:
    """Amplify has no image generation capability."""
    raise not_implemented("Image generation")


@app.post("/v1/images/edits")
async def create_image_edit(request: Request) -> None:
    """Amplify has no image editing capability."""
    raise not_implemented("Image editing")


@app.post("/v1/images/variations")
async def create_image_variation(request: Request) -> None:
    """Amplify has no image variation capability."""
    raise not_implemented("Image variation")


# ---------------------------------------------------------------------------
# Fine-tuning endpoints (501 stubs)
# ---------------------------------------------------------------------------


@app.post("/v1/fine_tuning/jobs")
async def create_fine_tuning_job(request: Request) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job creation")


@app.get("/v1/fine_tuning/jobs")
async def list_fine_tuning_jobs() -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job listing")


@app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}")
async def retrieve_fine_tuning_job(fine_tuning_job_id: str) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job retrieval")


@app.post("/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel")
async def cancel_fine_tuning_job(fine_tuning_job_id: str) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job cancellation")


@app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}/events")
async def list_fine_tuning_events(fine_tuning_job_id: str) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning event listing")


# ---------------------------------------------------------------------------
# Moderations (501 stub)
# ---------------------------------------------------------------------------


@app.post("/v1/moderations")
async def create_moderation(request: Request) -> None:
    """Amplify has no content moderation endpoint."""
    raise not_implemented("Content moderation")


# ---------------------------------------------------------------------------
# Batch endpoints (501 stubs)
# ---------------------------------------------------------------------------


@app.post("/v1/batches")
async def create_batch(request: Request) -> None:
    """Amplify has no batch processing endpoint."""
    raise not_implemented("Batch creation")


@app.get("/v1/batches")
async def list_batches() -> None:
    """Amplify has no batch listing endpoint."""
    raise not_implemented("Batch listing")


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str) -> None:
    """Amplify has no batch status endpoint."""
    raise not_implemented("Batch retrieval")


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str) -> None:
    """Amplify has no batch cancellation endpoint."""
    raise not_implemented("Batch cancellation")


# ---------------------------------------------------------------------------
# Vector stores endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/vector_stores")
async def create_vector_store(
    request: Request, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Create a virtual vector store backed by an Amplify tag.

    Amplify knowledge base + tagging approximates OpenAI's vector stores.
    A unique tag is created to represent the store.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    name = body.get("name", f"vs-{uuid.uuid4().hex[:8]}")
    tag_id = name
    logger.info("Creating vector store (tag): %s", tag_id)

    payload: AmplifyTagsRequest = {"data": {"tags": [tag_id]}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/tags/create",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return {
            "id": tag_id,
            "object": "vector_store",
            "created_at": int(time.time()),
            "name": name,
            "usage_bytes": 0,
            "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "status": "completed",
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error creating vector store %s: %s", tag_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.get("/v1/vector_stores/{vector_store_id}")
async def retrieve_vector_store(
    vector_store_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Retrieve a vector store by ID.

    Confirms the backing tag exists and counts the associated files.
    """
    logger.info("Retrieving vector store: %s", vector_store_id)
    try:
        tags_resp = requests.get(
            f"{AMPLIFY_BASE_URL}/files/tags/list",
            headers=headers,
            timeout=30,
        )
        tags_resp.raise_for_status()
        tags = tags_resp.json().get("data", {}).get("tags", [])
        if vector_store_id not in tags:
            raise HTTPException(status_code=404, detail=f"Vector store '{vector_store_id}' not found")

        items = query_amplify_files(headers, tags=[vector_store_id])
        file_count = len(items)
        return {
            "id": vector_store_id,
            "object": "vector_store",
            "created_at": int(time.time()),
            "name": vector_store_id,
            "usage_bytes": sum(item.get("totalTokens", 0) * 4 for item in items),
            "file_counts": {
                "in_progress": 0,
                "completed": file_count,
                "failed": 0,
                "cancelled": 0,
                "total": file_count,
            },
            "status": "completed",
        }
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error retrieving vector store %s: %s", vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.post("/v1/vector_stores/{vector_store_id}")
async def modify_vector_store(vector_store_id: str, request: Request) -> None:
    """Amplify has no tag rename endpoint; vector store modification is not supported."""
    raise not_implemented("Vector store modification")


@app.delete("/v1/vector_stores/{vector_store_id}")
async def delete_vector_store(
    vector_store_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Delete a vector store by removing its backing tag via Amplify POST /files/tags/delete.

    Note: this removes the tag only; underlying files remain in the knowledge base.
    """
    logger.info("Deleting vector store (tag): %s", vector_store_id)
    payload: AmplifyTagsRequest = {"data": {"tag": vector_store_id}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/tags/delete",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": vector_store_id,
            "object": "vector_store.deleted",
            "deleted": bool(data.get("success", False)),
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error deleting vector store %s: %s", vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.get("/v1/vector_stores/{vector_store_id}/files")
async def list_vector_store_files(
    vector_store_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """List all files in a vector store by querying Amplify files filtered by tag."""
    logger.info("Listing files in vector store: %s", vector_store_id)
    try:
        items = query_amplify_files(headers, tags=[vector_store_id])
        data = [
            {
                "id": item.get("id", ""),
                "object": "vector_store.file",
                "usage_bytes": item.get("totalTokens", 0) * 4,
                "created_at": 0,
                "vector_store_id": vector_store_id,
                "status": "completed",
            }
            for item in items
        ]
        return {"object": "list", "data": data}
    except requests.exceptions.RequestException as e:
        logger.error("Error listing vector store files for %s: %s", vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.post("/v1/vector_stores/{vector_store_id}/files")
async def create_vector_store_file(
    vector_store_id: str,
    request: Request,
    headers: dict = Depends(get_amplify_headers),
) -> Dict[str, Any]:
    """
    Associate a file with a vector store by adding the store tag to the file.

    Uses Amplify POST /files/set_tags with the vector_store_id as a tag.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    file_id = body.get("file_id", "")
    logger.info("Adding file %s to vector store %s", file_id, vector_store_id)

    payload: AmplifyTagsRequest = {"data": {"id": file_id, "tags": [vector_store_id]}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/set_tags",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return {
            "id": file_id,
            "object": "vector_store.file",
            "usage_bytes": 0,
            "created_at": int(time.time()),
            "vector_store_id": vector_store_id,
            "status": "completed",
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error adding file %s to vector store %s: %s", file_id, vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@app.delete("/v1/vector_stores/{vector_store_id}/files/{file_id:path}")
async def delete_vector_store_file(vector_store_id: str, file_id: str) -> None:
    """
    Amplify has no endpoint to remove a single tag from a single file.

    POST /files/set_tags replaces the entire tag list; partial removal via that
    endpoint requires first fetching the current tags, which is not exposed.
    """
    raise not_implemented("Vector store file removal")


# Vector store file batch endpoints (501 stubs) --------------------------------


@app.post("/v1/vector_stores/{vector_store_id}/file_batches")
async def create_vector_store_file_batch(vector_store_id: str, request: Request) -> None:
    """Amplify has no async batch file ingestion endpoint."""
    raise not_implemented("Vector store file batch creation")


@app.get("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}")
async def retrieve_vector_store_file_batch(vector_store_id: str, batch_id: str) -> None:
    """Amplify has no async batch file status endpoint."""
    raise not_implemented("Vector store file batch retrieval")


@app.post("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel")
async def cancel_vector_store_file_batch(vector_store_id: str, batch_id: str) -> None:
    """Amplify has no async batch file cancellation endpoint."""
    raise not_implemented("Vector store file batch cancellation")


@app.get("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files")
async def list_files_in_vector_store_batch(vector_store_id: str, batch_id: str) -> None:
    """Amplify has no async batch file listing endpoint."""
    raise not_implemented("Vector store file batch file listing")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Start the Uvicorn server.

    Bind address and port are read from environment variables so that the
    NixOS systemd unit can configure them without requiring code changes:
      AMPLIFY_SERVER_HOST  - defaults to 0.0.0.0
      AMPLIFY_SERVER_PORT  - defaults to 8000
    """
    host = os.getenv("AMPLIFY_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("AMPLIFY_SERVER_PORT", "8080"))
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run("open_amplify_ai.server:app", host=host, port=port, reload=False)
