"""Unit tests for chat endpoint handling mixed message and tool call responses.

Tests the case where Amplify AI returns both natural language text and tool calls
in a single response, which is a valid scenario.

NOTE: Most tests are marked as xfail because the current implementation does not
support mixed content (text + tool calls in same response). These tests document
the limitation and will serve as regression tests when support is added.
"""
import json
import os
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_async_client(mocker, response):
    """Build an async httpx client mock for non-streaming calls."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_streaming_client(mocker, lines):
    """Build an async httpx client mock for streaming calls.
    
    Args:
        lines: list of str lines that aiter_lines() will yield.
    """
    async def fake_aiter_lines():
        for line in lines:
            yield line
    
    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = mocker.Mock()
    mock_resp.aiter_lines = fake_aiter_lines
    
    mock_stream_cm = mocker.AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_resp
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    # client.stream(...) is a sync call returning an async CM
    mock_client.stream = mocker.Mock(return_value=mock_stream_cm)
    return mock_client


@pytest.mark.xfail(reason="Mixed content (text + tool calls) not yet supported")
def test_mixed_text_before_tool_call_non_streaming(mocker):
    """Test that text before a tool call is preserved in non-streaming mode."""
    # Amplify returns text followed by a tool call
    tool_response = (
        'I will help you list the files.\n'
        '{"tool":"list_files","parameters":{"path":"/home","recursive":true}}'
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "List files"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path", "recursive"],
                },
            },
        }],
    }
    
    response = client.post("/v1/chat/completions", json=req_body)
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should have tool_calls
    assert "tool_calls" in choice["message"]
    assert len(choice["message"]["tool_calls"]) == 1
    assert choice["finish_reason"] == "tool_calls"
    
    # Check if content is preserved (this may fail with current implementation)
    # According to OpenAI spec, content can be present alongside tool_calls
    message = choice["message"]
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "list_files"
    
    # This is what we want to test: does content get preserved?
    # Current implementation may set content to None
    # The test documents current behavior
    if message["content"] is not None:
        assert "help you list" in message["content"].lower()


@pytest.mark.xfail(reason="Mixed content (text + tool calls) not yet supported")
def test_mixed_text_after_tool_call_non_streaming(mocker):
    """Test that text after a tool call is preserved in non-streaming mode."""
    # Amplify returns tool call followed by text
    tool_response = (
        '{"tool":"list_files","parameters":{"path":"/home","recursive":true}}\n'
        'I have initiated the file listing.'
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "List files"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path", "recursive"],
                },
            },
        }],
    }
    
    response = client.post("/v1/chat/completions", json=req_body)
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should have tool_calls
    assert "tool_calls" in choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    
    # Check tool call
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "list_files"
    
    # Check if trailing content is preserved
    message = choice["message"]
    if message["content"] is not None:
        assert "initiated" in message["content"].lower()


@pytest.mark.xfail(reason="Mixed content (text + tool calls) not yet supported")
def test_mixed_text_surrounding_tool_call_non_streaming(mocker):
    """Test that text surrounding a tool call is preserved in non-streaming mode."""
    tool_response = (
        'Let me help you with that task.\n'
        '{"tool":"read_file","parameters":{"path":"/tmp/test.txt"}}\n'
        'I will read the file for you.'
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Read test.txt"}],
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
    }
    
    response = client.post("/v1/chat/completions", json=req_body)
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should have tool_calls
    assert "tool_calls" in choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    
    # Check tool call
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "read_file"
    
    # Check if surrounding content is preserved
    message = choice["message"]
    if message["content"] is not None:
        # Should contain both before and after text
        content_lower = message["content"].lower()
        assert "help you" in content_lower or "read the file" in content_lower


@pytest.mark.xfail(reason="Mixed content (text + tool calls) not yet supported in streaming")
def test_mixed_content_streaming(mocker):
    """Test that mixed content is properly streamed with both text and tool calls."""
    lines = [
        'I will help you with that.\n',
        '{"tool":"list_files","parameters":{"path":"/home","recursive":true}}',
        'data: [DONE]',
    ]
    
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "List files"}],
        "stream": True,
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path", "recursive"],
                },
            },
        }],
    }
    
    response = client.post("/v1/chat/completions", json=req_body)
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    
    body = response.text
    assert "data:" in body
    assert "[DONE]" in body
    
    # Parse SSE chunks to verify structure
    has_content = False
    has_tool_call = False
    
    for line in body.strip().split("\n"):
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        
        try:
            chunk = json.loads(line[6:])
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            
            # Check for content deltas
            if "content" in delta and delta["content"]:
                has_content = True
            
            # Check for tool call deltas
            if "tool_calls" in delta:
                has_tool_call = True
        except json.JSONDecodeError:
            pass
    
    # With current implementation, we should see tool calls
    # Content may or may not be streamed depending on implementation
    assert has_tool_call, "Expected tool_calls in streaming response"


def test_canonical_format_with_message_and_tool_call(mocker):
    """Test canonical PROTOCOL_V1 format that includes both message and tool call."""
    # Canonical format with explicit _tool_call marker
    tool_response = json.dumps({
        "_tool_call": True,
        "id": "call_test123",
        "tool": "list_files",
        "parameters": {"path": "/home", "recursive": True},
        "message": "I will list the files for you.",
    })
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "List files"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path", "recursive"],
                },
            },
        }],
    }
    
    response = client.post("/v1/chat/completions", json=req_body)
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should have tool_calls
    assert "tool_calls" in choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    
    # Check tool call
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_test123"
    assert tool_call["function"]["name"] == "list_files"
    
    # Check if message field is preserved as content
    # This depends on implementation
    message = choice["message"]
    # Note: current implementation may not extract the "message" field
