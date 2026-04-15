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
    model_metadata = await get_model_metadata(internal_req.model, headers)
    if model_metadata:
        output_limit = model_metadata.get("outputTokenLimit")
        context_limit = model_metadata.get("inputContextWindow")
        
        if output_limit and internal_req.max_tokens:
            if internal_req.max_tokens > output_limit:
                logger.warning(
                    "Requested max_tokens (%d) exceeds model '%s' limit (%d)",
                    internal_req.max_tokens,
                    internal_req.model,
                    output_limit,
                )
                raise create_validation_error(
                    f"Requested max_tokens ({internal_req.max_tokens}) exceeds "
                    f"model '{internal_req.model}' output token limit ({output_limit}). "
                    f"Please reduce max_tokens to {output_limit} or less.",
                    param="max_tokens",
                )
    
    logger.info(
        "Creating chat completion with model %s (stream=%s, request_id=%s)",
        internal_req.model,
        internal_req.stream,
        request_id,
    )
    
    # Stage 2 & 3: Transform internal IR to Amplify format
    try:
        amplify_request = internal_request_to_amplify(internal_req)
    except Exception as e:
        logger.error("Failed to transform request to Amplify format: %s", e)
        raise create_validation_error(f"Request transformation failed: {e}")
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    
    # Handle streaming
    if internal_req.stream:
        logger.info("Streaming response requested for model %s", internal_req.model)
        
        include_usage = bool(
            internal_req.stream_options
            and internal_req.stream_options.get("include_usage")
        )
        
        try:
            return StreamingResponse(
                stream_amplify_response(
                    amplify_request=amplify_request,
                    headers=headers,
                    model=internal_req.model,
                    completion_id=completion_id,
                    created=created,
                    tools=internal_req.tools,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )
        except httpx.HTTPError as e:
            raise normalize_upstream_error(e, "streaming chat completion", request_id)
    
    # Handle non-streaming
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0, read=120.0)
        ) as client:
            response = await client.post(
                f"{AMPLIFY_BASE_URL}/chat",
                headers=headers,
                json=amplify_request,
            )
            response.raise_for_status()
        
        # Stage 4: Parse Amplify response
        try:
            data = response.json()
            content = data.get("data", "")
        except Exception:
            content = response.text
        
        # Stage 5: Parse tool calls using deterministic parser
        parse_result = parse_tool_calls(content, internal_req.tools)
        
        if parse_result.is_tool_call:
            # Validate tool calls if tools were provided
            if internal_req.tools:
                for tool_call in parse_result.tool_calls:
                    try:
                        args = json.loads(tool_call.function_arguments)
                        is_valid = validate_tool_output(
                            tool_call.function_name,
                            args,
                            internal_req.tools,
                        )
                        if not is_valid:
                            logger.warning(
                                "Tool call validation failed for '%s'",
                                tool_call.function_name,
                            )
                    except json.JSONDecodeError:
                        logger.warning(
                            "Invalid JSON in tool call arguments for '%s'",
                            tool_call.function_name,
                        )
            
            # Build tool_calls response
            tool_calls_json = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function_name,
                        "arguments": tc.function_arguments,
                    },
                }
                for tc in parse_result.tool_calls
            ]
            
            # Support mixed content: include both remaining_content and tool_calls
            message_obj = {
                "role": "assistant",
                "content": parse_result.remaining_content,
                "tool_calls": tool_calls_json,
            }
            finish_reason = "tool_calls"
        else:
            # Normal content response
            message_obj = {
                "role": "assistant",
                "content": content if isinstance(content, str) else str(content),
            }
            finish_reason = "stop"
        
        # Build OpenAI response
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": internal_req.model,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    
    except httpx.HTTPError as e:
        raise normalize_upstream_error(e, "chat completion", request_id)
    except Exception as e:
        logger.error("Unexpected error in chat completion: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Internal server error: {e}", "type": "api_error"}},
        )
