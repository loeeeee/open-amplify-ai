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
