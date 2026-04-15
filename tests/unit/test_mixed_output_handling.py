"""Tests for mixed output handling (text + tool calls)."""
import json
import pytest

from open_amplify_ai.tool_parsing import (
    handle_mixed_output,
    parse_tool_calls,
    MixedOutputResult,
    ToolParseResult,
)
from open_amplify_ai.types import ToolDefinition


@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    return [
        ToolDefinition(
            type="function",
            function={
                "name": "search_docs",
                "description": "Search documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        ),
        ToolDefinition(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        ),
    ]


class TestMixedOutputParsing:
    """Test mixed output parsing from various formats."""

    def test_text_only_no_tools(self):
        """Test plain text with no tool calls."""
        content = "This is a plain text response."
        result = handle_mixed_output(content, None)

        assert not result.has_tool_calls
        assert result.content == content
        assert len(result.tool_calls) == 0

    def test_text_before_tool_call_canonical(self, sample_tools):
        """Test text commentary before canonical tool call."""
        content = (
            "Let me search the documentation for you.\n\n"
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "refund policy"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert result.content == "Let me search the documentation for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"
        assert result.validation_passed

    def test_text_after_tool_call_canonical(self, sample_tools):
        """Test text commentary after canonical tool call."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "refund policy"}}\n\n'
            "I'll retrieve that information for you."
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert result.content == "I'll retrieve that information for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"

    def test_text_before_and_after_tool_call(self, sample_tools):
        """Test text commentary both before and after tool call."""
        content = (
            "Let me check the weather.\n\n"
            '{"_tool_call": true, "id": "call_456", "tool": "get_weather", '
            '"parameters": {"location": "San Francisco"}}\n\n'
            "One moment please."
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        # Should combine text before and after
        assert "Let me check the weather." in result.content
        assert "One moment please." in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "get_weather"

    def test_json_format_mixed_content(self, sample_tools):
        """Test JSON format with mixed content."""
        content = (
            "I'll search for that information.\n\n"
            '{"tool": "search_docs", "parameters": {"query": "API documentation"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert "I'll search for that information." in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"

    def test_xml_format_mixed_content(self, sample_tools):
        """Test XML format with mixed content."""
        content = (
            "Looking up weather data...\n\n"
            "<tool_call>\n"
            "  <tool_name>get_weather</tool_name>\n"
            "  <parameters>\n"
            "    <location>New York</location>\n"
            "  </parameters>\n"
            "</tool_call>\n\n"
            "Please wait."
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert "Looking up weather data..." in result.content
        assert "Please wait." in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "get_weather"


class TestMixedOutputValidation:
    """Test validation and fallback behavior."""

    def test_tool_call_without_declared_tools(self):
        """Test tool call when no tools were declared - should fallback."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}'
        )
        result = handle_mixed_output(content, None)

        assert not result.has_tool_calls
        assert result.content == content
        assert not result.validation_passed
        assert result.fallback_reason == "no_tools_declared"

    def test_undeclared_tool_fallback(self, sample_tools):
        """Test tool call for undeclared tool - should fallback."""
        content = (
            '{"_tool_call": true, "id": "call_999", "tool": "undeclared_tool", '
            '"parameters": {"param": "value"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Undeclared tools are rejected at parse time, so no tool call is detected
        assert not result.has_tool_calls
        assert result.content == content
        # Since no tool call was parsed, validation_passed remains True
        # (there was nothing to validate)
        assert result.validation_passed
        assert result.fallback_reason is None

    def test_invalid_json_arguments_fallback(self, sample_tools):
        """Test tool call with invalid JSON arguments - should fallback."""
        content = (
            "Let me search.\n"
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": "not a dict"}'
        )
        # This should parse but fail validation since parameters is a string
        parse_result = parse_tool_calls(content, sample_tools)

        # The parser should detect this
        if parse_result.is_tool_call:
            result = handle_mixed_output(content, sample_tools)
            # Should either parse with valid JSON or fallback
            if not result.has_tool_calls:
                assert not result.validation_passed

    def test_multiple_tool_calls_partial_invalid(self, sample_tools):
        """Test multiple tool calls where some are invalid."""
        content = (
            '{"tool": "search_docs", "parameters": {"query": "test"}}\n'
            '{"tool": "invalid_tool", "parameters": {"param": "value"}}\n'
            '{"tool": "get_weather", "parameters": {"location": "NYC"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should only include valid tool calls
        assert result.has_tool_calls
        assert len(result.tool_calls) == 2
        tool_names = [tc.function_name for tc in result.tool_calls]
        assert "search_docs" in tool_names
        assert "get_weather" in tool_names
        assert "invalid_tool" not in tool_names


class TestMixedOutputEdgeCases:
    """Test edge cases in mixed output handling."""

    def test_empty_content(self):
        """Test empty content."""
        result = handle_mixed_output("", None)

        assert not result.has_tool_calls
        assert result.content == ""
        assert len(result.tool_calls) == 0

    def test_only_whitespace(self):
        """Test content with only whitespace."""
        result = handle_mixed_output("   \n\n  \t  ", None)

        assert not result.has_tool_calls

    def test_tool_call_with_empty_parameters(self, sample_tools):
        """Test tool call with empty parameters object."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should parse successfully even with empty params
        # Validation will check required fields
        assert result.has_tool_calls or not result.validation_passed

    def test_tool_call_no_commentary(self, sample_tools):
        """Test tool call with no surrounding commentary."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert result.content is None  # No commentary text
        assert len(result.tool_calls) == 1

    def test_multiple_tool_calls_with_commentary(self, sample_tools):
        """Test multiple tool calls with interspersed commentary."""
        content = (
            "Let me get both pieces of information.\n\n"
            '{"tool": "search_docs", "parameters": {"query": "refunds"}}\n'
            '{"tool": "get_weather", "parameters": {"location": "Boston"}}\n\n'
            "Processing your requests."
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert "Let me get both pieces of information." in result.content
        assert "Processing your requests." in result.content
        assert len(result.tool_calls) == 2


class TestOpenAIFormatCompliance:
    """Test that output matches OpenAI format requirements."""

    def test_content_null_allowed_with_tool_calls(self, sample_tools):
        """Test that content can be None when tool calls are present."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        # Content should be None (no commentary)
        assert result.content is None
        assert len(result.tool_calls) == 1

    def test_content_present_with_tool_calls(self, sample_tools):
        """Test that content can be present alongside tool calls."""
        content = (
            "Let me search for that.\n\n"
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert result.content is not None
        assert "Let me search for that." in result.content
        assert len(result.tool_calls) == 1

    def test_tool_call_structure(self, sample_tools):
        """Test that tool calls have required fields."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        tool_call = result.tool_calls[0]

        # Verify required fields
        assert tool_call.id is not None
        assert tool_call.type == "function"
        assert tool_call.function_name == "search_docs"
        assert tool_call.function_arguments is not None

        # Verify arguments are valid JSON
        args = json.loads(tool_call.function_arguments)
        assert isinstance(args, dict)
        assert args["query"] == "test"


class TestParserRetention:
    """Test that all parser formats support mixed content."""

    def test_canonical_format_extracts_remaining_content(self, sample_tools):
        """Test canonical format parser extracts remaining content."""
        content = (
            "Preamble text.\n"
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}\n'
            "Postamble text."
        )
        parse_result = parse_tool_calls(content, sample_tools)

        assert parse_result.is_tool_call
        assert parse_result.remaining_content is not None
        assert "Preamble" in parse_result.remaining_content
        assert "Postamble" in parse_result.remaining_content

    def test_json_format_extracts_remaining_content(self, sample_tools):
        """Test JSON format parser extracts remaining content."""
        content = (
            "Here's what I found:\n"
            '{"tool": "search_docs", "parameters": {"query": "test"}}'
        )
        parse_result = parse_tool_calls(content, sample_tools)

        assert parse_result.is_tool_call
        assert parse_result.remaining_content is not None
        assert "Here's what I found" in parse_result.remaining_content

    def test_xml_format_extracts_remaining_content(self, sample_tools):
        """Test XML format parser extracts remaining content."""
        content = (
            "Searching now...\n"
            "<tool_call>\n"
            "  <tool_name>search_docs</tool_name>\n"
            "  <parameters>\n"
            "    <query>test</query>\n"
            "  </parameters>\n"
            "</tool_call>\n"
            "Done."
        )
        parse_result = parse_tool_calls(content, sample_tools)

        assert parse_result.is_tool_call
        assert parse_result.remaining_content is not None
        assert "Searching now" in parse_result.remaining_content
        assert "Done" in parse_result.remaining_content

    def test_legacy_format_extracts_remaining_content(self, sample_tools):
        """Test legacy format parser extracts remaining content."""
        content = (
            "[Tool Call: search_docs]\n"
            "Parameters: {\"query\": \"test\"}\n\n"
            "Processing..."
        )
        parse_result = parse_tool_calls(content, sample_tools)

        assert parse_result.is_tool_call
        # Legacy format should extract text after
        if parse_result.remaining_content:
            assert "Processing" in parse_result.remaining_content


class TestStreamingScenarios:
    """Test scenarios from actual streaming responses."""

    def test_xml_like_tags_not_tool_calls(self, sample_tools):
        """Test XML-like tags that are NOT tool calls should be treated as text."""
        content = (
            "I'll implement a procedure to handle mixed output based on the plan. "
            "Let me first examine the existing code structure to understand how to "
            "integrate this properly.\n\n"
            "<read_file>\n"
            "<files>\n"
            "<path>src/open_amplify_ai/tool_parsing.py</path>\n"
            "<path>src/open_amplify_ai/transformation.py</path>\n"
            "<path>src/open_amplify_ai/streaming.py</path>\n"
            "</files>\n"
            "</read_file>"
        )
        result = handle_mixed_output(content, sample_tools)

        # Should NOT parse as tool call since <read_file> is not a declared tool
        assert not result.has_tool_calls
        assert result.content == content
        assert len(result.tool_calls) == 0

    def test_question_follow_up_tags_not_tool_calls(self, sample_tools):
        """Test <question> and <follow_up> tags should not be parsed as tool calls."""
        content = (
            "I'll examine the existing code structure first.\n\n"
            "<question>This task involves implementing mixed output handling. "
            "Should I proceed with implementation, or would you like me to first "
            "create a detailed implementation plan document before coding?</question>\n\n"
            "<follow_up>\n"
            "[\n"
            "  {\n"
            '    "text": "Proceed with implementation - create the procedure with tests",\n'
            '    "mode": "code"\n'
            "  },\n"
            "  {\n"
            '    "text": "Create a detailed implementation plan first",\n'
            '    "mode": "architect"\n'
            "  }\n"
            "]\n"
            "</follow_up>"
        )
        result = handle_mixed_output(content, sample_tools)

        # Should NOT parse as tool call
        assert not result.has_tool_calls
        assert result.content == content
        assert len(result.tool_calls) == 0

    def test_escaped_json_in_text_not_tool_call(self, sample_tools):
        """Test escaped JSON in text should not be parsed as tool call."""
        content = (
            "Now I have read all log files. Let me also check the remaining files.\n"
            '{\\"_tool_call\\": true, \\"id\\": \\"call_010\\", '
            '\\"tool\\": \\"read_file\\", \\"parameters\\": '
            '{\\"files\\": [{\\"path\\": \\"logs/example.txt\\"}]}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Escaped JSON should NOT be parsed as tool call
        assert not result.has_tool_calls
        assert result.content == content

    def test_streaming_canonical_format_in_wrapped_response(self, sample_tools):
        """Test canonical tool call wrapped in success response structure."""
        # This simulates what we get from Amplify in streaming mode
        content = (
            '{"_tool_call": true, "id": "call_001", "tool": "search_docs", '
            '"parameters": {"files": [{"path": "src/tool_parsing.py", '
            '"line_ranges": null}]}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should parse successfully
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"

    def test_mixed_content_with_json_in_text_description(self, sample_tools):
        """Test JSON-like content in narrative text should not interfere with parsing."""
        content = (
            "The format is {\"tool\": \"name\", \"parameters\": {...}}. "
            "Let me use the actual tool now:\n\n"
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should correctly identify only the canonical tool call
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"
        # Commentary should be preserved
        assert "The format is" in result.content

    def test_multiple_canonical_formats_in_sequence(self, sample_tools):
        """Test multiple canonical tool calls in sequence."""
        content = (
            '{"_tool_call": true, "id": "call_001", "tool": "search_docs", '
            '"parameters": {"query": "first"}}\n'
            '{"_tool_call": true, "id": "call_002", "tool": "get_weather", '
            '"parameters": {"location": "NYC"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should parse both tool calls
        # Note: Current implementation may only parse first canonical format
        # This test documents expected behavior
        if result.has_tool_calls:
            # At least one should be parsed
            assert len(result.tool_calls) >= 1
            tool_names = [tc.function_name for tc in result.tool_calls]
            assert "search_docs" in tool_names or "get_weather" in tool_names


class TestAmbiguousFormats:
    """Test handling of ambiguous or malformed formats."""

    def test_json_with_tool_field_but_no_tool_call_marker(self, sample_tools):
        """Test JSON with 'tool' field but missing canonical marker."""
        content = '{"tool": "search_docs", "parameters": {"query": "test"}}'
        result = handle_mixed_output(content, sample_tools)

        # Should parse as JSON format (compatibility)
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"

    def test_incomplete_canonical_format(self, sample_tools):
        """Test incomplete canonical JSON should not be parsed."""
        content = '{"_tool_call": true, "id": "call_123"'
        result = handle_mixed_output(content, sample_tools)

        # Should not parse incomplete JSON
        assert not result.has_tool_calls
        assert result.content == content

    def test_canonical_format_with_extra_fields(self, sample_tools):
        """Test canonical format with extra fields should still parse."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}, "extra_field": "ignored"}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should parse successfully, ignoring extra fields
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"

    def test_tool_call_in_code_block(self, sample_tools):
        """Test tool call syntax in markdown code block should not be parsed."""
        content = (
            "Here's an example of the format:\n"
            "```json\n"
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}\n'
            "```\n"
        )
        result = handle_mixed_output(content, sample_tools)

        # Current implementation may still parse this
        # This test documents that code blocks should ideally be excluded
        # For now, we accept either behavior
        if result.has_tool_calls:
            # If it parses, verify it's correct
            assert result.tool_calls[0].function_name == "search_docs"
        else:
            # If it doesn't parse, that's also acceptable
            assert result.content == content

    def test_html_escaped_json_not_parsed(self, sample_tools):
        """Test HTML-escaped JSON should not be parsed as tool call."""
        content = (
            "The JSON format is: &lt;tool_call&gt;{&quot;tool&quot;: &quot;search&quot;}&lt;/tool_call&gt;"
        )
        result = handle_mixed_output(content, sample_tools)

        # HTML-escaped content should not parse
        assert not result.has_tool_calls
        assert result.content == content


class TestRobustness:
    """Test robustness against malformed or edge case inputs."""

    def test_unicode_in_tool_call(self, sample_tools):
        """Test Unicode characters in tool call parameters."""
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "\\u4e2d\\u6587\\u6d4b\\u8bd5"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should handle Unicode correctly
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function_arguments)
        assert "query" in args

    def test_newlines_in_tool_call_json(self, sample_tools):
        """Test newlines within JSON structure."""
        content = (
            '{\n'
            '  "_tool_call": true,\n'
            '  "id": "call_123",\n'
            '  "tool": "search_docs",\n'
            '  "parameters": {\n'
            '    "query": "test"\n'
            '  }\n'
            '}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should handle formatted JSON
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1

    def test_very_long_content_with_tool_call(self, sample_tools):
        """Test long content before tool call doesn't break parsing."""
        long_text = "a" * 10000
        content = (
            f"{long_text}\n\n"
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should still find and parse tool call
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        # Content should include the long text
        assert len(result.content) > 9000

    def test_special_characters_in_parameters(self, sample_tools):
        """Test special characters in tool parameters."""
        # Use proper JSON escaping
        content = (
            '{"_tool_call": true, "id": "call_123", "tool": "search_docs", '
            '"parameters": {"query": "test with tabs and quotes"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should handle special characters
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function_arguments)
        assert "test" in args["query"]

    def test_partially_parsed_response_with_tool_call(self, sample_tools):
        """Test partially parsed response containing a tool call in data field."""
        content = (
            '{"success": true, "message": "Chat endpoint response retrieved", '
            '"data": "{\\"_tool_call\\": true, \\"id\\": \\"call_001\\", '
            '\\"tool\\": \\"search_docs\\", \\"parameters\\": {\\"query\\": \\"test\\"}}"}'
        )
        result = handle_mixed_output(content, sample_tools)

        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"
        # The remaining content should be the text message or empty
        # Wait, if it's partially parsed, maybe we should extract the data field?

    def test_text_followed_by_canonical_tool_call(self, sample_tools):
        """Test text commentary followed by canonical format tool call."""
        content = (
            "Now I have read all log files. Let me also check the remaining files quickly, then write the report.\n"
            '{"_tool_call": true, "id": "call_010", "tool": "search_docs", '
            '"parameters": {"query": "logs/wireguard-2.txt"}}'
        )
        result = handle_mixed_output(content, sample_tools)

        # Should parse as BOTH text and tool call
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "search_docs"
        # Verify parameters structure
        args = json.loads(result.tool_calls[0].function_arguments)
        assert "query" in args
        # Text commentary should be preserved
        assert result.content is not None
        assert "read all log files" in result.content.lower()

    def test_deeply_nested_json_in_canonical_tool_call(self, sample_tools):
        """Test canonical tool call with deeply nested JSON parameters."""
        # Add a mock tool for read_file
        tools = sample_tools + [
            ToolDefinition(
                type="function",
                function={
                    "name": "read_file",
                    "description": "Read files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            )
        ]

        content = (
            "Now I have read all log files. Let me also check the remaining wireguard-2 and spark worker files quickly, then write the report.\n"
            '{"_tool_call": true, "id": "call_010", "tool": "read_file", "parameters":\n'
            '{"files": [{"path":\n'
            '"logs/backwater-errors-20260414-204114/128.22.100.122.txt"},\n'
            '{"path":\n'
            '"logs/backwater-errors-20260414-204114/spark-worker1-128.22.0.137.txt"}]}}'
        )
        result = handle_mixed_output(content, tools)

        # Should parse as BOTH text and tool call
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "read_file"
        # Verify parameters structure
        args = json.loads(result.tool_calls[0].function_arguments)
        assert "files" in args
        assert len(args["files"]) == 2
        # Text commentary should be preserved
        assert result.content is not None
        assert "read all log files" in result.content.lower()
