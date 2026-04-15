"""Deterministic tool call parsing from Amplify responses."""
import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from open_amplify_ai.types import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A detected but not yet validated tool call candidate with its span."""
    start: int
    end: int
    parser_used: str
    function_name: str
    arguments: Any  # dict, pre-parsed
    raw: str


@dataclass
class ToolParseResult:
    """Result of attempting to parse tool calls from content."""
    is_tool_call: bool
    tool_calls: List[ToolCall] = field(default_factory=list)
    remaining_content: str = ""
    parser_used: Optional[str] = None


@dataclass
class MixedOutputResult:
    """Result of handling mixed output (text + tool calls)."""
    has_tool_calls: bool
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    validation_passed: bool = True
    fallback_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Stable ID generation
# ---------------------------------------------------------------------------

def stable_tool_call_id(name: str, args: dict, ordinal: int) -> str:
    """
    Generate a stable, deterministic tool call ID from content.

    The ID is derived from a normalized JSON payload so that the same
    tool call at the same position always produces the same ID.
    This enables deterministic deduplication and caching.
    """
    payload = json.dumps(
        {"name": name, "args": args, "ordinal": ordinal},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"call_{digest}"


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

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

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == "}":
                brace_level -= 1
                if brace_level == 0 and start_idx != -1:
                    json_str = content[start_idx : i + 1]
                    results.append((json_str, start_idx, i + 1))
                    start_idx = -1
                elif brace_level < 0:
                    # Malformed, reset
                    brace_level = 0
                    start_idx = -1

    return results


# ---------------------------------------------------------------------------
# Candidate detectors
# Each function returns List[Candidate].  No validation is performed here.
# ---------------------------------------------------------------------------

def detect_canonical_candidates(content: str) -> List[Candidate]:
    """
    Detect canonical PROTOCOL_V1 tool call candidates.

    Format: {"_tool_call": true, "id": "call_xxx", "tool": "name", "parameters": {...}}

    The strong anchor is the presence of the "_tool_call" key.  Both
    whole-message and embedded variants are supported, because the
    "_tool_call" marker is unambiguous.

    Returns a list of Candidate objects (may be empty).
    """
    if '"_tool_call"' not in content and "'_tool_call'" not in content:
        return []

    candidates: List[Candidate] = []

    # Attempt whole-message parse first
    try:
        parsed = json.loads(content.strip())
        if (
            isinstance(parsed, dict)
            and parsed.get("_tool_call")
            and parsed.get("tool")
        ):
            tool_name = parsed["tool"]
            args = parsed.get("parameters", {})
            if isinstance(args, dict):
                candidates.append(
                    Candidate(
                        start=0,
                        end=len(content),
                        parser_used="canonical_v1",
                        function_name=tool_name,
                        arguments=args,
                        raw=content,
                    )
                )
                return candidates
    except json.JSONDecodeError:
        pass

    # Embedded variant: scan all JSON objects in text
    for json_str, start_idx, end_idx in extract_json_objects(content):
        try:
            parsed = json.loads(json_str)
            if (
                isinstance(parsed, dict)
                and parsed.get("_tool_call")
                and parsed.get("tool")
            ):
                tool_name = parsed["tool"]
                args = parsed.get("parameters", {})
                if isinstance(args, dict):
                    candidates.append(
                        Candidate(
                            start=start_idx,
                            end=end_idx,
                            parser_used="canonical_v1_embedded",
                            function_name=tool_name,
                            arguments=args,
                            raw=json_str,
                        )
                    )
        except json.JSONDecodeError:
            continue

    return candidates


def detect_legacy_candidates(content: str) -> List[Candidate]:
    """
    Detect legacy format tool call candidates.

    Format: [Tool Call: name]\nParameters: {...}

    Text before and after the block is supported.  The "[Tool Call:" marker
    is treated as a strong enough anchor to distinguish live calls from
    explanatory prose, so the trimmed-start constraint from the original
    implementation is removed.

    Known limitation: the regex terminates the parameters capture on a blank
    line (double newline), so pretty-printed JSON with blank lines inside
    will be parsed incorrectly.

    Returns a list of Candidate objects (may be empty).
    """
    if "[Tool Call:" not in content:
        return []

    try:
        match = re.search(
            r"\[Tool Call:\s*([^\]]+)\]\s*\n?Parameters:\s*(\{[\s\S]*?\})(?:\n\n|$)",
            content,
            re.DOTALL,
        )
        if not match:
            return []

        name = match.group(1).strip()
        params_str = match.group(2).strip()

        try:
            args = json.loads(params_str)
        except json.JSONDecodeError:
            logger.debug("Legacy format: invalid JSON in parameters block")
            return []

        if not isinstance(args, dict):
            logger.debug("Legacy format: parameters is not a dict")
            return []

        return [
            Candidate(
                start=match.start(),
                end=match.end(),
                parser_used="legacy",
                function_name=name,
                arguments=args,
                raw=match.group(0),
            )
        ]

    except Exception as exc:
        logger.debug("Legacy format detection failed: %s", exc)
        return []


def detect_json_candidates(content: str) -> List[Candidate]:
    """
    Detect JSON format tool call candidates.

    Format: {"tool": "name", "parameters": {...}}
             or {"command": "name", "parameters": {...}}

    Whole-message only: the entire trimmed content must parse as a single
    JSON object.  Embedded JSON blocks inside explanatory prose are NOT
    promoted, because the anchor is too weak to avoid false positives.

    The 'parameters' key is required.  A dict with 'tool' but without
    'parameters' is rejected as malformed.

    Returns a list of Candidate objects (may be empty).
    """
    if '"tool"' not in content and '"command"' not in content:
        return []

    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        # Not a whole-message JSON; do not scan embedded blocks.
        return []

    if not isinstance(parsed, dict):
        return []

    name = parsed.get("tool") or parsed.get("command")
    if not name:
        return []

    # Require explicit 'parameters' key to avoid silently using {}
    if "parameters" not in parsed:
        logger.debug(
            "JSON format: 'parameters' key missing for tool '%s', rejecting as malformed",
            name,
        )
        return []

    args = parsed["parameters"]
    if not isinstance(args, dict):
        logger.debug(
            "JSON format: 'parameters' is not a dict for tool '%s', rejecting",
            name,
        )
        return []

    return [
        Candidate(
            start=0,
            end=len(content),
            parser_used="json",
            function_name=name,
            arguments=args,
            raw=content,
        )
    ]


def detect_xml_candidates(content: str) -> List[Candidate]:
    """
    Detect XML format tool call candidates.

    Format: <tool_call><tool_name>name</tool_name><parameters>...</parameters></tool_call>
            or <tool_use>...</tool_use>

    LOSSY FORMAT - explicit limitations:
    - All parameter values are extracted as strings from element text.
    - Boolean strings ("true"/"false") are coerced to Python bool.
    - Integer and float values become strings, e.g. <count>3</count> -> "3".
    - Null/none values become the string "null" or empty string.
    - Nested XML structures (lists, dicts) inside parameters are silently
      collapsed to their concatenated text, losing all structure.
    - Attribute values on parameter elements are ignored.

    This format is provided as a compatibility fallback only.  Callers
    that require type-accurate arguments should use canonical format.

    Returns a list of Candidate objects (may be empty).
    """
    if "<tool_call>" not in content and "<tool_use>" not in content:
        return []

    try:
        xml_match = re.search(
            r"<tool_(?:call|use)>[\s\S]*?</tool_(?:call|use)>",
            content,
            re.DOTALL,
        )
        if not xml_match:
            return []

        xml_content = f"<root>{xml_match.group(0)}</root>"
        root = ET.fromstring(xml_content)

        tool_elem = root.find(".//tool_call")
        if tool_elem is None:
            tool_elem = root.find(".//tool_use")
        if tool_elem is None:
            return []

        tool_name_elem = tool_elem.find("tool_name")
        if tool_name_elem is None or not tool_name_elem.text:
            return []

        name = tool_name_elem.text.strip()

        # Parse parameters (lossy: strings only, booleans coerced)
        args: Dict[str, Any] = {}
        params_elem = tool_elem.find("parameters")
        if params_elem is not None:
            for child in params_elem:
                value = child.text or ""
                if value.lower() == "false":
                    args[child.tag] = False
                elif value.lower() == "true":
                    args[child.tag] = True
                else:
                    args[child.tag] = value

        return [
            Candidate(
                start=xml_match.start(),
                end=xml_match.end(),
                parser_used="xml",
                function_name=name,
                arguments=args,
                raw=xml_match.group(0),
            )
        ]

    except Exception as exc:
        logger.debug("XML format detection failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Single validation layer
# ---------------------------------------------------------------------------

def validate_candidates(
    candidates: List[Candidate],
    tools: List[ToolDefinition],
) -> Tuple[List[Candidate], List[Candidate]]:
    """
    Validate a list of candidates against declared tools.

    Returns (promoted, rejected) where:
    - promoted: candidates that pass all checks and become executable tool calls
    - rejected: candidates that failed validation

    Validation checks:
    1. Tool name exists in declared tools
    2. Arguments are a dict (already pre-parsed by detectors)
    """
    promoted: List[Candidate] = []
    rejected: List[Candidate] = []

    for candidate in candidates:
        tool_exists = any(t.get_name() == candidate.function_name for t in tools)
        if not tool_exists:
            logger.warning(
                "Tool '%s' not in declared tools, excluding from response",
                candidate.function_name,
            )
            rejected.append(candidate)
            continue

        if not isinstance(candidate.arguments, dict):
            logger.warning(
                "Tool '%s' arguments are not a dict, excluding from response",
                candidate.function_name,
            )
            rejected.append(candidate)
            continue

        promoted.append(candidate)

    return promoted, rejected


# ---------------------------------------------------------------------------
# Content reconstruction
# ---------------------------------------------------------------------------

def reconstruct_remaining_content(
    content: str,
    promoted: List[Candidate],
) -> str:
    """
    Rebuild remaining text content by removing only promoted candidate spans.

    Rejected (invalid) candidate spans are left in the content so that
    no structured text is silently dropped.

    Returns the remaining text, or empty string if nothing remains.
    """
    if not promoted:
        return content

    # Sort by start position to process in order
    spans = sorted([(c.start, c.end) for c in promoted])

    parts: List[str] = []
    last_end = 0

    for start, end in spans:
        segment = content[last_end:start].strip()
        if segment:
            parts.append(segment)
        last_end = end

    tail = content[last_end:].strip()
    if tail:
        parts.append(tail)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_tool_calls(
    content: str,
    tools: Optional[List[ToolDefinition]] = None,
) -> ToolParseResult:
    """
    Parse tool calls from Amplify response content.

    Pipeline:
    1. Detect candidates using each format's detector.
    2. If tools are declared, validate candidates against them.
    3. Reconstruct remaining content from non-promoted spans.
    4. Return ToolParseResult.

    Canonical format is tried first and is the only format that supports
    embedded tool calls inside text.  Compatibility formats (legacy, JSON,
    XML) only match whole-message content to avoid false positives.
    """
    if not content:
        return ToolParseResult(is_tool_call=False, remaining_content="")

    # Step 1: detect candidates from canonical format (supports embedded)
    candidates = detect_canonical_candidates(content)

    if not candidates:
        # Only try compatibility parsers if tools are declared
        if not tools:
            return ToolParseResult(is_tool_call=False, remaining_content=content)

        candidates.extend(detect_legacy_candidates(content))

    if not candidates and tools:
        candidates.extend(detect_json_candidates(content))

    if not candidates and tools:
        candidates.extend(detect_xml_candidates(content))

    if not candidates:
        return ToolParseResult(is_tool_call=False, remaining_content=content)

    # Step 2: validate (requires declared tools)
    if not tools:
        # Canonical format detected but no tools declared; fall back
        return ToolParseResult(is_tool_call=False, remaining_content=content)

    promoted, _rejected = validate_candidates(candidates, tools)

    if not promoted:
        return ToolParseResult(is_tool_call=False, remaining_content=content)

    # Step 3: build ToolCall objects from promoted candidates
    tool_calls: List[ToolCall] = []
    for ordinal, candidate in enumerate(promoted):
        # Use the ID from the canonical payload if present, else derive stable ID
        call_id: Optional[str] = None
        if candidate.parser_used in ("canonical_v1", "canonical_v1_embedded"):
            try:
                raw_parsed = json.loads(candidate.raw)
                call_id = raw_parsed.get("id")
            except (json.JSONDecodeError, AttributeError):
                pass

        if not call_id:
            call_id = stable_tool_call_id(candidate.function_name, candidate.arguments, ordinal)

        tool_calls.append(
            ToolCall(
                id=call_id,
                type="function",
                function_name=candidate.function_name,
                function_arguments=json.dumps(candidate.arguments),
            )
        )

    # Step 4: reconstruct remaining content excluding promoted spans
    remaining = reconstruct_remaining_content(content, promoted)

    # Use the parser name from the first promoted candidate
    parser_used = promoted[0].parser_used if promoted else None

    logger.info(
        "Parsed %d tool call(s) using parser '%s'",
        len(tool_calls),
        parser_used,
    )

    return ToolParseResult(
        is_tool_call=True,
        tool_calls=tool_calls,
        remaining_content=remaining,
        parser_used=parser_used,
    )


def handle_mixed_output(
    content: str,
    tools: Optional[List[ToolDefinition]] = None,
) -> MixedOutputResult:
    """
    Handle mixed output from Amplify, splitting text commentary and tool calls.

    Steps:
    1. Detect and validate tool calls via parse_tool_calls().
    2. If tool calls found, return them with remaining text.
    3. If no tool calls, return content as plain text.
    4. If tools are declared but no valid calls found, fall back to plain text.

    Note: transport-envelope unwrapping (JSON success/data wrapper) is
    intentionally not performed here.  That belongs in the Amplify response
    adapter upstream.
    """
    if not content:
        return MixedOutputResult(has_tool_calls=False, content="", tool_calls=[])

    parse_result = parse_tool_calls(content, tools)

    if not parse_result.is_tool_call:
        return MixedOutputResult(
            has_tool_calls=False,
            content=content,
            tool_calls=[],
        )

    # Tool calls were detected and validated by parse_tool_calls
    return MixedOutputResult(
        has_tool_calls=True,
        content=parse_result.remaining_content,
        tool_calls=parse_result.tool_calls,
        validation_passed=True,
    )
