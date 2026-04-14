"""Request validation and transformation to internal IR."""
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from open_amplify_ai.types import (
    ContentPart,
    ContentPartType,
    ErrorResponse,
    ErrorType,
    InternalMessage,
    InternalRequest,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Supported OpenAI parameters
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
}

# Parameters that are silently ignored (documented)
IGNORED_PARAMS = {
    "user",  # Not used by Amplify
    "logprobs",  # Not supported by Amplify
    "logit_bias",  # Not supported by Amplify
    "parallel_tool_calls",  # Always true when tools provided
}

# Parameters that cause rejection (behavior-changing)
REJECTED_PARAMS = {
    "n",  # Multiple completions not supported
    "top_p",  # Not supported by Amplify
    "presence_penalty",  # Not supported by Amplify
    "frequency_penalty",  # Not supported by Amplify
    "seed",  # Determinism not supported
    "response_format",  # Structured output not fully supported
    "stop",  # Stop sequences not supported
}


def create_error_response(
    message: str,
    error_type: ErrorType,
    status_code: int,
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> HTTPException:
    """Create a structured error response matching OpenAI format."""
    error_obj = {
        "error": {
            "message": message,
            "type": error_type.value,
            "param": param,
            "code": code,
        }
    }
    return HTTPException(status_code=status_code, detail=error_obj)


def validate_and_parse_request(req_json: Dict[str, Any]) -> InternalRequest:
    """
    Validate incoming OpenAI request and parse to internal IR.
    
    Raises HTTPException with structured error for invalid requests.
    """
    # Check for required fields
    if "model" not in req_json:
        raise create_error_response(
            "Missing required parameter: 'model'",
            ErrorType.INVALID_REQUEST_ERROR,
            400,
            param="model",
        )
    
    if "messages" not in req_json or not isinstance(req_json["messages"], list):
        raise create_error_response(
            "Missing or invalid required parameter: 'messages'",
            ErrorType.INVALID_REQUEST_ERROR,
            400,
            param="messages",
        )
    
    # Check for unsupported parameters
    unsupported_found = {}
    for param in req_json.keys():
        if param in REJECTED_PARAMS:
            raise create_error_response(
                f"Parameter '{param}' is not supported by the Amplify AI backend",
                ErrorType.INVALID_REQUEST_ERROR,
                400,
                param=param,
            )
        elif param not in SUPPORTED_PARAMS and param not in IGNORED_PARAMS:
            logger.warning("Unknown parameter '%s' in request", param)
            unsupported_found[param] = req_json[param]
    
    # Parse messages
    messages = []
    for i, msg in enumerate(req_json["messages"]):
        try:
            internal_msg = parse_message(msg)
            messages.append(internal_msg)
        except ValueError as e:
            raise create_error_response(
                f"Invalid message at index {i}: {e}",
                ErrorType.INVALID_REQUEST_ERROR,
                400,
                param=f"messages[{i}]",
            )
    
    # Validate at least one message
    if not messages:
        raise create_error_response(
            "At least one message is required",
            ErrorType.INVALID_REQUEST_ERROR,
            400,
            param="messages",
        )
    
    # Parse tools if present
    tools = None
    if "tools" in req_json and req_json["tools"]:
        tools = []
        for i, tool in enumerate(req_json["tools"]):
            try:
                tools.append(ToolDefinition(type=tool.get("type", "function"), function=tool.get("function", {})))
            except Exception as e:
                raise create_error_response(
                    f"Invalid tool definition at index {i}: {e}",
                    ErrorType.INVALID_REQUEST_ERROR,
                    400,
                    param=f"tools[{i}]",
                )
    
    # Validate tool_choice if present
    tool_choice = req_json.get("tool_choice")
    if tool_choice is not None and tools is None:
        raise create_error_response(
            "'tool_choice' requires 'tools' to be specified",
            ErrorType.INVALID_REQUEST_ERROR,
            400,
            param="tool_choice",
        )
    
    # Build internal request
    return InternalRequest(
        model=req_json["model"],
        messages=messages,
        temperature=req_json.get("temperature", 0.7),
        max_tokens=req_json.get("max_tokens", 10000),
        stream=req_json.get("stream", False),
        stream_options=req_json.get("stream_options"),
        tools=tools,
        tool_choice=tool_choice,
        unsupported_params=unsupported_found,
    )


def parse_message(msg: Dict[str, Any]) -> InternalMessage:
    """
    Parse a single OpenAI message into internal representation.
    
    Raises ValueError for invalid messages.
    """
    if "role" not in msg:
        raise ValueError("Message missing required 'role' field")
    
    role_str = msg["role"]
    
    # Map role to internal enum
    try:
        if role_str == "developer":
            role = MessageRole.DEVELOPER
        elif role_str == "system":
            role = MessageRole.SYSTEM
        elif role_str == "user":
            role = MessageRole.USER
        elif role_str == "assistant":
            role = MessageRole.ASSISTANT
        elif role_str == "tool":
            role = MessageRole.TOOL
        else:
            raise ValueError(f"Unknown role: {role_str}")
    except Exception:
        raise ValueError(f"Invalid role: {role_str}")
    
    # Parse content
    content_parts = []
    content_raw = msg.get("content")
    
    if content_raw is not None:
        if isinstance(content_raw, str):
            # Simple string content
            content_parts.append(ContentPart(type=ContentPartType.TEXT, text=content_raw))
        elif isinstance(content_raw, list):
            # Array content - validate each part
            for part in content_raw:
                if isinstance(part, str):
                    content_parts.append(ContentPart(type=ContentPartType.TEXT, text=part))
                elif isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        content_parts.append(
                            ContentPart(type=ContentPartType.TEXT, text=part.get("text", ""))
                        )
                    elif part_type == "image_url":
                        # Reject unsupported content type
                        raise ValueError(
                            "Content type 'image_url' is not supported by Amplify AI backend"
                        )
                    elif part_type == "image_file":
                        raise ValueError(
                            "Content type 'image_file' is not supported by Amplify AI backend"
                        )
                    elif part_type == "audio":
                        raise ValueError(
                            "Content type 'audio' is not supported by Amplify AI backend"
                        )
                    else:
                        raise ValueError(f"Unknown content part type: {part_type}")
                else:
                    raise ValueError(f"Invalid content part: {part}")
    
    # Parse tool_calls if present (assistant messages)
    tool_calls = None
    if "tool_calls" in msg and msg["tool_calls"]:
        tool_calls = []
        for tc in msg["tool_calls"]:
            if tc.get("type") != "function":
                raise ValueError(f"Unsupported tool call type: {tc.get('type')}")
            
            func = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", f"call_{tc.get('function', {}).get('name', 'unknown')}"),
                    type="function",
                    function_name=func.get("name", ""),
                    function_arguments=func.get("arguments", "{}"),
                )
            )
    
    # Parse tool result if this is a tool message
    tool_result = None
    if role == MessageRole.TOOL:
        tool_call_id = msg.get("tool_call_id")
        if not tool_call_id:
            raise ValueError("Tool message missing required 'tool_call_id' field")
        
        # Extract content as string
        result_content = ""
        for part in content_parts:
            if part.type == ContentPartType.TEXT and part.text:
                result_content += part.text
        
        tool_result = ToolResult(
            tool_call_id=tool_call_id,
            tool_name=msg.get("name", "unknown"),
            content=result_content,
        )
    
    return InternalMessage(
        role=role,
        content_parts=content_parts,
        tool_calls=tool_calls,
        tool_result=tool_result,
        name=msg.get("name"),
    )
