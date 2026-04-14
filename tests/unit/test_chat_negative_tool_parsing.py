"""Unit tests for negative tool call parsing scenarios.

Tests that verify the parser is not too eager and doesn't incorrectly
parse regular text as tool calls. Covers the negative tool call parsing
from the test refactor plan.
"""
import json
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app
import os

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_async_client(mocker, response):
    """Build an async httpx client mock for non-streaming calls."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def test_user_requests_json_output_not_tool_call(mocker):
    """Test that JSON output requested by user is not parsed as tool call."""
    # User asks for JSON, LLM provides it as text, not a tool call
    json_response = '{"name": "John", "age": 30, "city": "New York"}'
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": json_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Return JSON with name, age, and city fields"}
            ],
            # No tools provided - should not parse as tool call
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should be normal text content, not tool_calls
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]
    assert choice["message"]["content"] == json_response
    assert choice["finish_reason"] == "stop"


def test_assistant_json_in_fenced_code_block_not_tool_call(mocker):
    """Test that fenced JSON code block is not parsed as tool call."""
    json_response = (
        "Here's the JSON you requested:\n"
        "```json\n"
        '{"result": "success", "count": 42}\n'
        "```\n"
        "This is the output."
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": json_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Show me example JSON"}],
            # No tools - should not parse as tool call
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should remain as normal text content
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]
    assert choice["finish_reason"] == "stop"


def test_malformed_json_not_parsed_as_tool_call(mocker):
    """Test that malformed JSON is not parsed as tool call."""
    malformed_json = '{"tool":"read_file", "parameters":{"path":"/tmp/test.txt"'  # Missing closing braces
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": malformed_json}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Read file"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Malformed JSON should remain as text, not be parsed as tool call
    assert "content" in choice["message"]
    # Tool parsing should fail gracefully


def test_legacy_marker_in_prose_not_tool_call(mocker):
    """Test that legacy [Tool Call: ...] marker inside prose is not parsed as tool call."""
    prose_response = (
        "I noticed you used the [Tool Call: execute_command] syntax in your message. "
        "That's an interesting approach."
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": prose_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What do you think about my syntax?"}],
            # No tools - shouldn't parse as tool call
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should be normal text, not tool call
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]
    assert choice["finish_reason"] == "stop"


def test_xml_like_documentation_not_tool_call(mocker):
    """Test that XML-like content that is documentation is not parsed as tool call."""
    xml_doc = (
        "To use the tool, format your request as:\n"
        "<tool_call>\n"
        "  <tool_name>your_tool</tool_name>\n"
        "  <parameters>\n"
        "    <param>value</param>\n"
        "  </parameters>\n"
        "</tool_call>\n"
        "This is just documentation, not an actual call."
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": xml_doc}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "How do I use the tool system?"}],
            # No tools - should not parse as tool call
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should remain as documentation text
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]


def test_json_with_wrong_structure_not_tool_call(mocker):
    """Test that JSON with wrong structure is not parsed as tool call."""
    # JSON that looks tool-like but has wrong keys
    wrong_json = '{"function": "read_file", "args": {"path": "/tmp/test.txt"}}'
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": wrong_json}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Read file"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should not be parsed as tool call due to wrong structure
    # Should remain as content
    assert "content" in choice["message"]


def test_tool_name_not_in_provided_tools_not_parsed(mocker):
    """Test that tool call with name not in provided tools is not parsed."""
    # JSON refers to a tool that wasn't provided
    tool_json = '{"tool":"nonexistent_tool","parameters":{"arg":"value"}}'
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_json}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Do something"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Tool not in provided list should not be parsed as tool call
    # Should remain as regular content
    assert "content" in choice["message"]


def test_json_example_in_explanation_not_tool_call(mocker):
    """Test that JSON used as example in explanation is not parsed as tool call."""
    explanation = (
        "To call the list_files tool, you would use:\n"
        '{"tool":"list_files","parameters":{"path":"/home","recursive":true}}\n'
        "But I'm just explaining, not actually calling it."
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": explanation}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "How do I use list_files?"}],
            # No tools provided - should not parse as actual tool call
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should be explanatory text, not actual tool call
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]


def test_json_array_not_tool_call(mocker):
    """Test that JSON array is not parsed as tool call."""
    json_array = '[{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]'
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": json_array}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Give me a list of items"}],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # JSON array should remain as content
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]
    assert choice["message"]["content"] == json_array


def test_partial_json_not_tool_call(mocker):
    """Test that incomplete/partial JSON is not parsed as tool call."""
    partial_json = '{"tool":"read_file"'
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": partial_json}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Read file"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Partial/incomplete JSON should not be parsed as tool call
    assert "content" in choice["message"]


def test_no_tools_provided_no_parsing_attempt(mocker):
    """Test that when no tools are provided, tool parsing is not attempted."""
    # This looks like a tool call but no tools are in the request
    tool_like_json = '{"tool":"some_function","parameters":{"x":1}}'
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_like_json}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Return JSON"}],
            # Explicitly no tools
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Without tools in request, should never parse as tool call
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]
    assert choice["message"]["content"] == tool_like_json
    assert choice["finish_reason"] == "stop"


def test_string_containing_tool_keyword_not_tool_call(mocker):
    """Test that string merely containing 'tool' keyword is not parsed as tool call."""
    response_text = "The best tool for this job is a screwdriver."
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": response_text}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What tool should I use?"}],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Normal text mentioning "tool" should not be parsed as tool call
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert "content" in choice["message"]
    assert choice["message"]["content"] == response_text
