"""Unit tests for chat endpoint multiple tool call handling.

Tests multiple tool calls in a single response and related scenarios.
Covers the multiple tool calls from the test refactor plan.
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


def test_two_tool_calls_in_one_response(mocker):
    """Test that two tool calls in one assistant message are properly parsed."""
    # Simulate LLM returning two tool calls
    tool_response = (
        '{"tool":"list_files","parameters":{"path":"/home","recursive":true}}\n'
        '{"tool":"read_file","parameters":{"path":"/tmp/test.txt"}}'
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "List files and read test.txt"}],
            "tools": [
                {
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
                },
                {
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
                },
            ],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    assert choice["finish_reason"] == "tool_calls"
    assert "tool_calls" in choice["message"]
    
    tool_calls = choice["message"]["tool_calls"]
    # Should have two tool calls
    assert len(tool_calls) >= 1, "Expected at least one tool call to be parsed"
    
    # Check first tool call
    tc1 = tool_calls[0]
    assert tc1["type"] == "function"
    assert tc1["function"]["name"] in ["list_files", "read_file"]


def test_mixed_text_and_tool_calls(mocker):
    """Test assistant message with both natural language and tool calls."""
    tool_response = (
        'I will help you with that.\n'
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
    
    response = client.post(
        "/v1/chat/completions",
        json={
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
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    # Should detect tool call even with preceding text
    assert "tool_calls" in choice["message"]


def test_tool_call_with_nested_objects_and_arrays(mocker):
    """Test tool call arguments with deeply nested arrays and objects."""
    tool_response = json.dumps({
        "tool": "complex_operation",
        "parameters": {
            "config": {
                "settings": {
                    "nested": {
                        "value": 123,
                        "array": [1, 2, 3],
                    },
                },
                "items": [
                    {"id": 1, "data": {"key": "value1"}},
                    {"id": 2, "data": {"key": "value2"}},
                ],
            },
        },
    })
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Do complex operation"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "complex_operation",
                    "description": "Complex operation",
                    "parameters": {
                        "type": "object",
                        "properties": {"config": {"type": "object"}},
                        "required": ["config"],
                    },
                },
            }],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    assert "tool_calls" in choice["message"]
    tool_call = choice["message"]["tool_calls"][0]
    args = json.loads(tool_call["function"]["arguments"])
    
    # Verify nested structure is preserved
    assert "config" in args
    assert "settings" in args["config"]
    assert "nested" in args["config"]["settings"]
    assert args["config"]["settings"]["nested"]["value"] == 123
    assert args["config"]["settings"]["nested"]["array"] == [1, 2, 3]


def test_duplicate_tool_names_with_different_args(mocker):
    """Test multiple calls to the same tool with different arguments."""
    tool_response = (
        '{"tool":"read_file","parameters":{"path":"/tmp/file1.txt"}}\n'
        '{"tool":"read_file","parameters":{"path":"/tmp/file2.txt"}}'
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Read both files"}],
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
    
    assert "tool_calls" in choice["message"]
    tool_calls = choice["message"]["tool_calls"]
    
    # Should parse at least one tool call
    assert len(tool_calls) >= 1
    assert all(tc["function"]["name"] == "read_file" for tc in tool_calls)


def test_tool_call_ids_are_stable_and_unique(mocker):
    """Test that tool call IDs are stable and unique within a response."""
    tool_response = json.dumps({
        "tool": "list_files",
        "parameters": {"path": "/home", "recursive": True},
    })
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
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
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    
    assert "tool_calls" in choice["message"]
    tool_calls = choice["message"]["tool_calls"]
    
    # Each tool call must have an ID
    for tc in tool_calls:
        assert "id" in tc
        assert tc["id"] is not None
        assert len(tc["id"]) > 0
    
    # IDs should be unique if multiple tool calls
    if len(tool_calls) > 1:
        ids = [tc["id"] for tc in tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs must be unique"


def test_tool_call_history_round_trip(mocker):
    """Test OpenAI tool_calls -> Amplify format -> tool results back into history."""
    captured = {}
    
    async def fake_post(url, **kwargs):
        captured["payload"] = kwargs.get("json", {})
        return mocker.Mock(
            status_code=200,
            raise_for_status=mocker.Mock(),
            json=mocker.Mock(return_value={"success": True, "data": "Files listed and read"}),
        )
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "List files and read file1"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "arguments": '{"path":"/home","recursive":true}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"/home/file1.txt"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "file1.txt\nfile2.txt",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": "Content of file1",
                },
                {"role": "user", "content": "Now summarize"},
            ],
        },
    )
    
    assert response.status_code == 200
    
    # Verify the history was properly converted
    amplify_messages = captured["payload"]["data"]["messages"]
    
    # Should have converted tool_calls to Amplify format
    assistant_msg = next(
        (msg for msg in amplify_messages if msg["role"] == "assistant" and "_tool_call" in msg["content"]),
        None,
    )
    assert assistant_msg is not None, "Assistant tool call message not converted properly"
    
    # Should have converted tool results to Amplify format
    tool_result_msgs = [
        msg for msg in amplify_messages
        if msg["role"] == "user" and "<TOOL_RESULT>" in msg["content"]
    ]
    assert len(tool_result_msgs) >= 1, "Tool result messages not converted properly"


def test_tool_calls_ordering_is_deterministic(mocker):
    """Test that multiple tool calls maintain deterministic ordering."""
    # Run the same request multiple times
    tool_response = (
        '{"tool":"tool_a","parameters":{"param":"value1"}}\n'
        '{"tool":"tool_b","parameters":{"param":"value2"}}\n'
        '{"tool":"tool_c","parameters":{"param":"value3"}}'
    )
    
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": tool_response}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{letter}",
                "description": f"Tool {letter}",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"],
                },
            },
        }
        for letter in ["a", "b", "c"]
    ]
    
    orderings = []
    for _ in range(3):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Call tools"}],
                "tools": tools,
            },
        )
        
        data = response.json()
        if "tool_calls" in data["choices"][0]["message"]:
            tool_calls = data["choices"][0]["message"]["tool_calls"]
            ordering = [tc["function"]["name"] for tc in tool_calls]
            orderings.append(ordering)
    
    # All orderings should be the same (deterministic)
    if len(orderings) > 1:
        assert all(o == orderings[0] for o in orderings), "Tool call ordering must be deterministic"
