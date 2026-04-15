"""Deterministic tool call parsing from Amplify responses."""
import json
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from open_amplify_ai.types import InternalResponse, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class ToolParseResult:
    """Result of attempting to parse a tool call."""
    
    def __init__(
        self,
        is_tool_call: bool,
        tool_calls: Optional[List[ToolCall]] = None,
        remaining_content: Optional[str] = None,
        parser_used: Optional[str] = None,
    ):
        self.is_tool_call = is_tool_call
        self.tool_calls = tool_calls or []
        self.remaining_content = remaining_content
        self.parser_used = parser_used


class MixedOutputResult:
    """Result of handling mixed output (text + tool calls)."""
    
    def __init__(
        self,
        has_tool_calls: bool,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        validation_passed: bool = True,
        fallback_reason: Optional[str] = None,
    ):
        self.has_tool_calls = has_tool_calls
        self.content = content
        self.tool_calls = tool_calls or []
        self.validation_passed = validation_passed
        self.fallback_reason = fallback_reason


def parse_tool_calls(
    content: str,
    tools: Optional[List[ToolDefinition]] = None,
) -> ToolParseResult:
    """
    Parse tool calls from Amplify response content with strict anchoring.
    
    Strategy:
    1. Try canonical format first (strict)
    2. Only if tools were enabled, try compatibility parsers
    3. Compatibility parsers require strong anchors
    4. Validate tool name exists in allowed tools
    5. Validate arguments are valid JSON
    
    Returns ToolParseResult indicating whether tool call was found.
    """
    if not content:
        return ToolParseResult(is_tool_call=False, remaining_content=content)
    
    # Try canonical format first (PROTOCOL_V1)
    result = try_canonical_format(content, tools)
    if result.is_tool_call:
        logger.info("Parsed tool call using canonical format")
        return result
    
    # Only try compatibility parsers if tools were enabled
    if tools is None:
        return ToolParseResult(is_tool_call=False, remaining_content=content)
    
    # Try legacy format with strong anchor
    result = try_legacy_format(content, tools)
    if result.is_tool_call:
        logger.info("Parsed tool call using legacy format")
        return result
    
    # Try JSON format with strong anchor
    result = try_json_format(content, tools)
    if result.is_tool_call:
        logger.info("Parsed tool call using JSON format")
        return result
    
    # Try XML format with strong anchor
    result = try_xml_format(content, tools)
    if result.is_tool_call:
        logger.info("Parsed tool call using XML format")
        return result
    
    # Not a tool call
    return ToolParseResult(is_tool_call=False, remaining_content=content)


def handle_mixed_output(
    content: str,
    tools: Optional[List[ToolDefinition]] = None,
) -> MixedOutputResult:
    """
    Handle mixed output from Amplify, splitting text commentary and tool calls.
    
    This implements the plan's requirements:
    1. Treat structured tool calls as authoritative
    2. Only promote to tool call if parsing succeeds and tool is declared
    3. Keep commentary separate from tool calls
    4. Fall back to plain text if ambiguous
    
    Returns:
        MixedOutputResult with separated content and tool calls, or fallback to plain text
    """
    if not content:
        return MixedOutputResult(
            has_tool_calls=False,
            content="",
            tool_calls=[],
        )
    
    # Check if the content is a partially parsed JSON response
    # e.g. {"success": true, "message": "...", "data": "..."}
    try:
        parsed_content = json.loads(content)
        if isinstance(parsed_content, dict) and "data" in parsed_content and "success" in parsed_content:
            # Extract the data field which contains the actual content
            data_content = parsed_content["data"]
            if isinstance(data_content, str):
                content = data_content
    except json.JSONDecodeError:
        pass
    
    # Try to parse tool calls
    parse_result = parse_tool_calls(content, tools)
    
    if not parse_result.is_tool_call:
        # No tool call detected, return as plain text
        return MixedOutputResult(
            has_tool_calls=False,
            content=content,
            tool_calls=[],
        )
    
    # Tool call detected - validate before promoting
    if not tools:
        # If no tools were declared, fall back to plain text
        logger.warning("Tool call detected but no tools were declared, falling back to text")
        return MixedOutputResult(
            has_tool_calls=False,
            content=content,
            tool_calls=[],
            validation_passed=False,
            fallback_reason="no_tools_declared",
        )
    
    # Validate each tool call
    valid_tool_calls = []
    invalid_tools = []
    
    for tool_call in parse_result.tool_calls:
        # Check if tool exists in declared tools
        tool_exists = any(t.get_name() == tool_call.function_name for t in tools)
        if not tool_exists:
            logger.warning(
                "Tool '%s' not in declared tools, will exclude from response",
                tool_call.function_name,
            )
            invalid_tools.append(tool_call.function_name)
            continue
        
        # Validate JSON arguments
        try:
            args = json.loads(tool_call.function_arguments)
            if not isinstance(args, dict):
                logger.warning(
                    "Tool '%s' has non-dict arguments, will exclude from response",
                    tool_call.function_name,
                )
                invalid_tools.append(tool_call.function_name)
                continue
        except json.JSONDecodeError as e:
            logger.warning(
                "Tool '%s' has invalid JSON arguments: %s, will exclude from response",
                tool_call.function_name,
                e,
            )
            invalid_tools.append(tool_call.function_name)
            continue
        
        # Tool call is valid
        valid_tool_calls.append(tool_call)
    
    # If no valid tool calls, fall back to plain text
    if not valid_tool_calls:
        logger.warning(
            "All tool calls failed validation, falling back to plain text. Invalid tools: %s",
            invalid_tools,
        )
        return MixedOutputResult(
            has_tool_calls=False,
            content=content,
            tool_calls=[],
            validation_passed=False,
            fallback_reason="validation_failed",
        )
    
    # Return mixed output with separated content and valid tool calls
    return MixedOutputResult(
        has_tool_calls=True,
        content=parse_result.remaining_content,
        tool_calls=valid_tool_calls,
        validation_passed=True,
    )


def extract_json_objects(content: str) -> List[Tuple[str, int, int]]:
    """
    Extract all potential JSON objects from a string by counting braces.
    Returns a list of tuples: (json_string, start_index, end_index).
    """
    results = []
    brace_level = 0
    in_string = False
    escape_next = False
    start_idx = -1
    
    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and start_idx != -1:
                    json_str = content[start_idx:i+1]
                    results.append((json_str, start_idx, i+1))
                    start_idx = -1
                elif brace_level < 0:
                    # Malformed, reset
                    brace_level = 0
                    start_idx = -1
                    
    return results


def try_canonical_format(
    content: str,
    tools: Optional[List[ToolDefinition]] = None,
) -> ToolParseResult:
    """
    Try to parse canonical PROTOCOL_V1 format.
    
    Format: {"_tool_call": true, "id": "call_xxx", "tool": "name", "parameters": {...}}
    
    Strong anchor: Must have "_tool_call": true at the beginning.
    Supports mixed content: extracts text before/after tool call JSON.
    """
    # Strong anchor: look for the _tool_call marker
    if '"_tool_call"' not in content and "'_tool_call'" not in content:
        return ToolParseResult(is_tool_call=False)
    
    try:
        # Try to parse the entire content as JSON first
        parsed = json.loads(content.strip())
        
        if not isinstance(parsed, dict):
            return ToolParseResult(is_tool_call=False)
        
        if not parsed.get("_tool_call"):
            return ToolParseResult(is_tool_call=False)
        
        tool_name = parsed.get("tool")
        if not tool_name:
            logger.warning("Canonical format missing 'tool' field")
            return ToolParseResult(is_tool_call=False)
        
        # Validate tool exists if tools provided
        if tools and not any(t.get_name() == tool_name for t in tools):
            logger.warning("Tool '%s' not in allowed tools", tool_name)
            return ToolParseResult(is_tool_call=False)
        
        tool_call = ToolCall(
            id=parsed.get("id", f"call_{uuid.uuid4().hex[:12]}"),
            type="function",
            function_name=tool_name,
            function_arguments=json.dumps(parsed.get("parameters", {})),
        )
        
        return ToolParseResult(
            is_tool_call=True,
            tool_calls=[tool_call],
            remaining_content=None,
            parser_used="canonical_v1",
        )
    
    except json.JSONDecodeError:
        # Maybe embedded in text, try to extract with surrounding content
        # Use brace counting to handle nested JSON objects
        for json_str, start_idx, end_idx in extract_json_objects(content):
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and parsed.get("_tool_call") and parsed.get("tool"):
                    tool_name = parsed["tool"]
                    
                    # Validate tool exists
                    if tools and not any(t.get_name() == tool_name for t in tools):
                        logger.warning("Tool '%s' not in allowed tools", tool_name)
                        continue
                    
                    tool_call = ToolCall(
                        id=parsed.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        type="function",
                        function_name=tool_name,
                        function_arguments=json.dumps(parsed.get("parameters", {})),
                    )
                    
                    # Extract remaining content (text before and after tool call)
                    remaining_parts = []
                    text_before = content[:start_idx].strip()
                    text_after = content[end_idx:].strip()
                    
                    if text_before:
                        remaining_parts.append(text_before)
                    if text_after:
                        remaining_parts.append(text_after)
                    
                    remaining_text = "\n".join(remaining_parts) if remaining_parts else None
                    
                    return ToolParseResult(
                        is_tool_call=True,
                        tool_calls=[tool_call],
                        remaining_content=remaining_text,
                        parser_used="canonical_v1_embedded",
                    )
            except json.JSONDecodeError:
                continue
    
    return ToolParseResult(is_tool_call=False)


def try_legacy_format(
    content: str,
    tools: List[ToolDefinition],
) -> ToolParseResult:
    """
    Try legacy format: [Tool Call: name]\nParameters: {...}
    
    Strong anchor: Must start with [Tool Call:
    Supports mixed content: extracts text before/after tool call.
    """
    if not content.strip().startswith("[Tool Call:"):
        return ToolParseResult(is_tool_call=False)
    
    try:
        match = re.match(
            r"\[Tool Call:\s*([^\]]+)\]\s*\n?Parameters:\s*(.+?)(?:\n\n|$)",
            content,
            re.DOTALL,
        )
        if not match:
            return ToolParseResult(is_tool_call=False)
        
        name = match.group(1).strip()
        params_str = match.group(2).strip()
        
        # Validate tool exists
        if not any(t.get_name() == name for t in tools):
            logger.warning("Tool '%s' not in allowed tools", name)
            return ToolParseResult(is_tool_call=False)
        
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in legacy format parameters")
            return ToolParseResult(is_tool_call=False)
        
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function_name=name,
            function_arguments=json.dumps(params),
        )
        
        # Extract remaining content (text after tool call)
        remaining_text = None
        if match.end() < len(content):
            text_after = content[match.end():].strip()
            if text_after:
                remaining_text = text_after
        
        return ToolParseResult(
            is_tool_call=True,
            tool_calls=[tool_call],
            remaining_content=remaining_text,
            parser_used="legacy",
        )
    
    except Exception as e:
        logger.debug("Legacy format parsing failed: %s", e)
        return ToolParseResult(is_tool_call=False)


def try_json_format(
    content: str,
    tools: List[ToolDefinition],
) -> ToolParseResult:
    """
    Try JSON format with strong anchors: {"tool": ...} or {"command": ...}
    
    Strong anchor: Must have "tool" or "command" field.
    Supports mixed content: extracts JSON and returns remaining text.
    """
    # Strong anchor: check for tool/command field anywhere in content
    if '"tool"' not in content and '"command"' not in content:
        return ToolParseResult(is_tool_call=False)
    
    try:
        # Try parsing entire content first
        parsed = json.loads(content.strip())
        
        if not isinstance(parsed, dict):
            return ToolParseResult(is_tool_call=False)
        
        name = parsed.get("tool") or parsed.get("command")
        if not name:
            return ToolParseResult(is_tool_call=False)
        
        # Validate tool exists
        if not any(t.get_name() == name for t in tools):
            logger.warning("Tool '%s' not in allowed tools", name)
            return ToolParseResult(is_tool_call=False)
        
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function_name=name,
            function_arguments=json.dumps(parsed.get("parameters", {})),
        )
        
        return ToolParseResult(
            is_tool_call=True,
            tool_calls=[tool_call],
            remaining_content=None,
            parser_used="json",
        )
    
    except json.JSONDecodeError:
        # Try to extract JSON block from mixed content
        # Use brace counting to handle nested JSON objects
        tool_calls_found = []
        remaining_parts = []
        last_end = 0
        
        # Find all potential JSON objects with tool/command fields
        for json_str, start_idx, end_idx in extract_json_objects(content):
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    name = parsed.get("tool") or parsed.get("command")
                    if name and any(t.get_name() == name for t in tools):
                        # Valid tool call found
                        # Capture any text before this JSON block
                        if start_idx > last_end:
                            text_before = content[last_end:start_idx].strip()
                            if text_before:
                                remaining_parts.append(text_before)
                        
                        tool_call = ToolCall(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            type="function",
                            function_name=name,
                            function_arguments=json.dumps(parsed.get("parameters", {})),
                        )
                        tool_calls_found.append(tool_call)
                        last_end = end_idx
            except json.JSONDecodeError:
                continue
        
        # Capture any text after the last JSON block
        if last_end < len(content):
            text_after = content[last_end:].strip()
            if text_after:
                remaining_parts.append(text_after)
        
        if tool_calls_found:
            # Combine remaining text parts
            remaining_text = "\n".join(remaining_parts) if remaining_parts else None
            
            return ToolParseResult(
                is_tool_call=True,
                tool_calls=tool_calls_found,
                remaining_content=remaining_text,
                parser_used="json_embedded",
            )
    
    return ToolParseResult(is_tool_call=False)


def try_xml_format(
    content: str,
    tools: List[ToolDefinition],
) -> ToolParseResult:
    """
    Try XML format: <tool_call> or <tool_use>
    
    Strong anchor: Must have <tool_call> or <tool_use> tag.
    Supports mixed content: extracts text before/after tool call XML.
    """
    if "<tool_call>" not in content and "<tool_use>" not in content:
        return ToolParseResult(is_tool_call=False)
    
    try:
        # Extract XML block
        xml_match = re.search(
            r"<tool_(?:call|use)>([\s\S]*?)</tool_(?:call|use)>",
            content,
            re.DOTALL,
        )
        if not xml_match:
            return ToolParseResult(is_tool_call=False)
        
        xml_content = f"<root>{xml_match.group(0)}</root>"
        root = ET.fromstring(xml_content)
        
        tool_elem = root.find(".//tool_call")
        if tool_elem is None:
            tool_elem = root.find(".//tool_use")
        
        if tool_elem is None:
            return ToolParseResult(is_tool_call=False)
        
        tool_name_elem = tool_elem.find("tool_name")
        if tool_name_elem is None or not tool_name_elem.text:
            return ToolParseResult(is_tool_call=False)
        
        name = tool_name_elem.text.strip()
        
        # Validate tool exists
        if not any(t.get_name() == name for t in tools):
            logger.warning("Tool '%s' not in allowed tools", name)
            return ToolParseResult(is_tool_call=False)
        
        # Parse parameters
        params = {}
        params_elem = tool_elem.find("parameters")
        if params_elem is not None:
            for child in params_elem:
                value = child.text or ""
                # Convert string booleans
                if value.lower() == "false":
                    params[child.tag] = False
                elif value.lower() == "true":
                    params[child.tag] = True
                else:
                    params[child.tag] = value
        
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function_name=name,
            function_arguments=json.dumps(params),
        )
        
        # Extract remaining content (text before and after tool call)
        remaining_parts = []
        text_before = content[:xml_match.start()].strip()
        text_after = content[xml_match.end():].strip()
        
        if text_before:
            remaining_parts.append(text_before)
        if text_after:
            remaining_parts.append(text_after)
        
        remaining_text = "\n".join(remaining_parts) if remaining_parts else None
        
        return ToolParseResult(
            is_tool_call=True,
            tool_calls=[tool_call],
            remaining_content=remaining_text,
            parser_used="xml",
        )
    
    except Exception as e:
        logger.debug("XML format parsing failed: %s", e)
        return ToolParseResult(is_tool_call=False)
