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


def try_canonical_format(
    content: str,
    tools: Optional[List[ToolDefinition]] = None,
) -> ToolParseResult:
    """
    Try to parse canonical PROTOCOL_V1 format.
    
    Format: {"_tool_call": true, "id": "call_xxx", "tool": "name", "parameters": {...}}
    
    Strong anchor: Must have "_tool_call": true at the beginning.
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
        # Maybe embedded in text, try to extract
        match = re.search(
            r'\{\s*["\']_tool_call["\']\s*:\s*true[^}]*\}',
            content,
            re.DOTALL,
        )
        if match:
            try:
                parsed = json.loads(match.group(0))
                if parsed.get("_tool_call") and parsed.get("tool"):
                    tool_name = parsed["tool"]
                    
                    # Validate tool exists
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
                        parser_used="canonical_v1_embedded",
                    )
            except json.JSONDecodeError:
                pass
    
    return ToolParseResult(is_tool_call=False)


def try_legacy_format(
    content: str,
    tools: List[ToolDefinition],
) -> ToolParseResult:
    """
    Try legacy format: [Tool Call: name]\nParameters: {...}
    
    Strong anchor: Must start with [Tool Call:
    """
    if not content.strip().startswith("[Tool Call:"):
        return ToolParseResult(is_tool_call=False)
    
    try:
        match = re.match(
            r"\[Tool Call:\s*([^\]]+)\]\s*\n?Parameters:\s*(.+)",
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
        
        return ToolParseResult(
            is_tool_call=True,
            tool_calls=[tool_call],
            remaining_content=None,
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
    
    Strong anchor: Must have "tool" or "command" field near start.
    """
    # Strong anchor: check for tool/command field in first 100 chars
    if '"tool"' not in content[:100] and '"command"' not in content[:100]:
        return ToolParseResult(is_tool_call=False)
    
    try:
        # Try parsing entire content
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
        # Try to extract JSON block
        match = re.search(r'(\{[^}]*["\'](?:tool|command)["\'][^}]*\})', content)
        if match:
            try:
                parsed = json.loads(match.group(1))
                name = parsed.get("tool") or parsed.get("command")
                if name and any(t.get_name() == name for t in tools):
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
                        parser_used="json_embedded",
                    )
            except json.JSONDecodeError:
                pass
    
    return ToolParseResult(is_tool_call=False)


def try_xml_format(
    content: str,
    tools: List[ToolDefinition],
) -> ToolParseResult:
    """
    Try XML format: <tool_call> or <tool_use>
    
    Strong anchor: Must have <tool_call> or <tool_use> tag.
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
        
        return ToolParseResult(
            is_tool_call=True,
            tool_calls=[tool_call],
            remaining_content=None,
            parser_used="xml",
        )
    
    except Exception as e:
        logger.debug("XML format parsing failed: %s", e)
        return ToolParseResult(is_tool_call=False)
