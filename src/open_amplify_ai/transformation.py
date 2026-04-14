"""Transformation between internal IR and Amplify API format."""
import json
import logging
from typing import Any, Dict, List

from open_amplify_ai.types import (
    AmplifyChatData,
    AmplifyChatMessage,
    AmplifyChatOptions,
    AmplifyChatRequest,
    AmplifyModelOption,
    ContentPartType,
    InternalMessage,
    InternalRequest,
    MessageRole,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

# Tool protocol version for tracking
TOOL_PROTOCOL_VERSION = "v1"


def internal_request_to_amplify(
    internal_req: InternalRequest,
) -> AmplifyChatRequest:
    """
    Transform internal request IR to Amplify API format.
    
    This is Stage 3 of the pipeline: capability-aware rendering.
    """
    # Transform messages with role preservation
    amplify_messages: List[AmplifyChatMessage] = []
    
    # Check if we need to inject tool definitions
    tool_injection_needed = bool(internal_req.tools)
    
    for msg in internal_req.messages:
        amplify_msg = transform_message_to_amplify(msg)
        amplify_messages.append(amplify_msg)
    
    # Inject tool definitions into system message if tools provided
    if tool_injection_needed:
        tool_instruction = create_tool_instruction(internal_req.tools)
        
        # Find existing system message or create one
        system_msg_idx = None
        for i, msg in enumerate(amplify_messages):
            if msg["role"] == "system":
                system_msg_idx = i
                break
        
        if system_msg_idx is not None:
            # Append to existing system message
            amplify_messages[system_msg_idx]["content"] += "\n\n" + tool_instruction
        else:
            # Insert new system message at the beginning
            amplify_messages.insert(
                0,
                {"role": "system", "content": tool_instruction}
            )
    
    # Build Amplify request
    chat_data: AmplifyChatData = {
        "temperature": internal_req.temperature,
        "max_tokens": internal_req.max_tokens,
        "dataSources": [],
        "messages": amplify_messages,
        "options": {
            "model": {"id": internal_req.model},
        },
    }
    
    return {"data": chat_data}


def transform_message_to_amplify(msg: InternalMessage) -> AmplifyChatMessage:
    """
    Transform a single internal message to Amplify format.
    
    Key transformations:
    - DEVELOPER -> system (Amplify doesn't have developer role)
    - TOOL -> user (with structured tool result wrapper)
    - Assistant messages with tool_calls -> append tool call JSON
    - Content parts -> concatenated text (only text supported)
    """
    # Map role for Amplify
    if msg.role == MessageRole.DEVELOPER:
        amplify_role = "system"
    elif msg.role == MessageRole.TOOL:
        amplify_role = "user"  # Tools results become user messages
    else:
        amplify_role = msg.role.value
    
    # Build content
    content = ""
    
    # Extract text from content parts
    text_content = msg.get_text_content()
    
    # Handle tool results with explicit wrapper
    if msg.role == MessageRole.TOOL and msg.tool_result:
        # Wrap tool result as structured JSON for better parsing
        # Mark it clearly as untrusted tool output
        tool_result_wrapper = {
            "_tool_result": True,
            "tool_call_id": msg.tool_result.tool_call_id,
            "tool_name": msg.tool_result.tool_name,
            "content": msg.tool_result.content,
            "is_error": msg.tool_result.is_error,
        }
        content = "<TOOL_RESULT>\n" + json.dumps(tool_result_wrapper, indent=2) + "\n</TOOL_RESULT>"
    else:
        content = text_content
    
    # Append tool calls if present (assistant messages)
    if msg.tool_calls:
        for tc in msg.tool_calls:
            # Use canonical tool call format
            tool_call_obj = {
                "_tool_call": True,
                "id": tc.id,
                "tool": tc.function_name,
                "parameters": json.loads(tc.function_arguments),
            }
            content += "\n" + json.dumps(tool_call_obj) + "\n"
    
    return {"role": amplify_role, "content": content}


def create_tool_instruction(tools: List[ToolDefinition]) -> str:
    """
    Create deterministic tool instruction block.
    
    This is isolated in a clearly delimited block to avoid
    mixing prose instructions and tool protocol.
    """
    tools_json = []
    for tool in tools:
        tools_json.append({
            "type": tool.type,
            "function": tool.function,
        })
    
    instruction = (
        f"=== TOOL PROTOCOL {TOOL_PROTOCOL_VERSION} ===\n"
        "You have access to the following tools:\n\n"
        + json.dumps(tools_json, indent=2)
        + "\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. To use a tool, output EXACTLY ONE JSON object with this format:\n"
        '   {"_tool_call": true, "id": "call_xxxxx", "tool": "tool_name", "parameters": {"arg1": "value1"}}\n'
        "2. The JSON MUST include '_tool_call': true as the first field\n"
        "3. Generate a unique ID starting with 'call_'\n"
        "4. DO NOT wrap in markdown code blocks\n"
        "5. DO NOT output any other text before or after the JSON\n"
        "6. Only one tool call per response\n"
        "7. If you want to respond without using a tool, do NOT output any JSON\n"
        f"=== END TOOL PROTOCOL {TOOL_PROTOCOL_VERSION} ==="
    )
    
    return instruction


def validate_tool_output(
    tool_name: str,
    arguments: Dict[str, Any],
    tools: List[ToolDefinition],
) -> bool:
    """
    Validate tool output against declared schema.
    
    Returns True if valid, False otherwise.
    """
    # Find tool definition
    tool_def = None
    for tool in tools:
        if tool.get_name() == tool_name:
            tool_def = tool
            break
    
    if tool_def is None:
        logger.warning("Tool '%s' not found in tool definitions", tool_name)
        return False
    
    # Get parameter schema
    schema = tool_def.get_schema()
    
    # Basic validation: check required parameters
    required = schema.get("required", [])
    for param in required:
        if param not in arguments:
            logger.warning(
                "Required parameter '%s' missing from tool '%s' call",
                param,
                tool_name,
            )
            return False
    
    # Type validation would go here (if schema has type info)
    # For now, just check that we don't have extra top-level fields
    properties = schema.get("properties", {})
    for arg_name in arguments:
        if arg_name not in properties:
            logger.warning(
                "Unknown parameter '%s' in tool '%s' call",
                arg_name,
                tool_name,
            )
            # Don't fail for extra params, just log
    
    return True
