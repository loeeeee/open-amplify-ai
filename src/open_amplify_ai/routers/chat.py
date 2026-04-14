"""Chat completions endpoints mapped to the Amplify API.

Refactored to use a 5-stage pipeline:
1. Strict OpenAI request validation
2. Normalize to internal IR
3. Render to Amplify format
4. Parse response to internal IR
5. Render to OpenAI format
"""
import json
import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.error_handling import normalize_upstream_error, create_validation_error
from open_amplify_ai.streaming import stream_amplify_response
from open_amplify_ai.tool_parsing import parse_tool_calls
from open_amplify_ai.transformation import (
    internal_request_to_amplify,
    validate_tool_output,
)
from open_amplify_ai.types import InternalRequest, InternalResponse
from open_amplify_ai.utils import get_model_metadata
from open_amplify_ai.validation import validate_and_parse_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["Chat"])


@router.post("/completions")
async def create_chat_completion(
    request: Request, headers: dict = Depends(get_amplify_headers)
) -> Any:
    """
    Convert OpenAI POST /v1/chat/completions to Amplify POST /chat.

    Supports both streaming (SSE) and non-streaming responses.
    
    Five-stage pipeline:
    1. Strict validation of incoming OpenAI request
    2. Normalize to internal IR (preserves semantics)
    3. Render to Amplify format (capability-aware)
    4. Parse Amplify response to internal IR
    5. Render to OpenAI format
    """
    # Generate request ID for tracking
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    
    try:
        req_json = await request.json()
    except Exception as e:
        logger.error("Failed to parse request JSON: %s", e)
        raise create_validation_error("Invalid JSON in request body")
    
    # Stage 1: Strict validation and parse to internal IR
    try:
        internal_req = validate_and_parse_request(req_json)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Validation error: %s", e)
        raise create_validation_error(f"Request validation failed: {e}")

    # Validate max_tokens against model limits
    model_metadata = await get_model_metadata(chat_request.model, headers)
    if model_metadata:
        output_limit = model_metadata.get("outputTokenLimit")
        if output_limit and chat_request.max_tokens:
            if chat_request.max_tokens > output_limit:
                logger.warning(
                    "Requested max_tokens (%d) exceeds model '%s' limit (%d)",
                    chat_request.max_tokens,
                    chat_request.model,
                    output_limit,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Requested max_tokens ({chat_request.max_tokens}) exceeds "
                        f"model '{chat_request.model}' output token limit ({output_limit}). "
                        f"Please reduce max_tokens to {output_limit} or less."
                    ),
                )

    logger.info(
        "Creating chat completion with model %s (stream=%s)",
        chat_request.model,
        chat_request.stream,
    )

    amplify_request: AmplifyChatRequest = {
        "data": {
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
            "dataSources": [],
            "messages": [
                {"role": m.role, "content": m.content} for m in chat_request.messages
            ],
            "options": {
                "model": {"id": chat_request.model},
            },
        }
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if chat_request.stream:
        logger.info("Streaming response requested for model %s", chat_request.model)

        include_usage = bool(
            chat_request.stream_options
            and chat_request.stream_options.get("include_usage")
        )

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
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{AMPLIFY_BASE_URL}/chat",
                headers=headers,
                json=amplify_request,
            )
            response.raise_for_status()

        try:
            data = response.json()
            content = data.get("data", "")
        except Exception:
            content = response.text

        tool_calls = None

        if isinstance(content, str) and "[Tool Call:" in content:
            try:
                match = re.search(
                    r"\[Tool Call:\s*([^\]]+)\]\s*\n?Parameters:\s*(.+)",
                    content,
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
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(params),
                            },
                        }
                    ]
                    content = None
            except Exception as parse_err:
                logger.debug("Exception parsing legacy tool call format: %s", parse_err)

        if tool_calls is None and isinstance(content, str) and (
            '"tool"' in content or '"command"' in content
        ):
            try:
                match = re.search(r"(\{[\s\S]*\})", content)
                json_str = match.group(1) if match else content
                parsed_content = json.loads(json_str, strict=False)
                name = parsed_content.get("tool") or parsed_content.get("command")
                if name:
                    tool_calls = [
                        {
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
                    content = None
            except Exception as parse_err:
                logger.debug("Exception parsing JSON tool call: %s", parse_err)

        # Parse XML format tool calls (for Opus model)
        if tool_calls is None and isinstance(content, str) and (
            "<tool_call>" in content or "<tool_use>" in content
        ):
            try:
                # Try to extract XML block
                xml_match = re.search(
                    r"<tool_(?:call|use)>([\s\S]*?)</tool_(?:call|use)>",
                    content,
                    re.DOTALL,
                )
                if xml_match:
                    xml_content = f"<root>{xml_match.group(0)}</root>"
                    root = ET.fromstring(xml_content)
                    tool_elem = root.find(".//tool_call")
                    if tool_elem is None:
                        tool_elem = root.find(".//tool_use")
                    if tool_elem is not None:
                        tool_name_elem = tool_elem.find("tool_name")
                        params_elem = tool_elem.find("parameters")
                        if tool_name_elem is not None:
                            name = tool_name_elem.text or ""
                            params = {}
                            if params_elem is not None:
                                for child in params_elem:
                                    # Convert "false"/"true" strings to boolean
                                    value = child.text or ""
                                    if value.lower() == "false":
                                        params[child.tag] = False
                                    elif value.lower() == "true":
                                        params[child.tag] = True
                                    else:
                                        params[child.tag] = value
                            tool_calls = [
                                {
                                    "id": f"call_{uuid.uuid4().hex[:12]}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(params),
                                    },
                                }
                            ]
                            content = None
            except Exception as parse_err:
                logger.debug("Exception parsing XML tool call: %s", parse_err)

        message_obj: dict[str, Any] = {"role": "assistant", "content": content}
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
    except httpx.HTTPError as e:
        raise handle_upstream_error(logger, e, "chat completion")
