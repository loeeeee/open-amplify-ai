import io
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
    """Build an async httpx client mock for streaming calls via utils.stream_amplify_chat.

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


def test_chat_completions_success(mocker):
    """POST /v1/chat/completions returns OpenAI format from Amplify /chat."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": "This is a mocked response from Amplify.",
    }
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4o"
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "This is a mocked response from Amplify."
    assert "usage" in data
    assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_invalid_request():
    """POST /v1/chat/completions returns 400 when messages are malformed."""
    req_body = {"messages": "this is not a list"}
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 400


def test_chat_completions_extra_fields(mocker):
    """POST /v1/chat/completions ignores extra fields like 'name' in messages without failing."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello", "name": "Cline"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_list_content(mocker):
    """POST /v1/chat/completions extracts text if content is a list of dicts."""
    captured = {}

    async def fake_post(url, **kwargs):
        captured["payload"] = kwargs.get("json", {})
        m = mocker.Mock()
        m.raise_for_status = mocker.Mock()
        m.json.return_value = {"success": True, "data": "ok"}
        return m

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)

    req_body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is "},
                    {"type": "text", "text": "Linux?"},
                ],
            }
        ],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert captured["payload"]["data"]["messages"][0]["content"] == "What is Linux?"


def test_chat_completions_stream_options(mocker):
    """POST /v1/chat/completions with stream_options and include_usage emits a usage chunk."""
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, ["data: Hello"]),
    )

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    body = response.text
    assert '"choices": []' in body
    assert '"usage":' in body


def test_chat_completions_tool_call_parsing(mocker):
    """POST /v1/chat/completions parses a JSON string command into structured tool_calls."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": '{"command":"list_files","parameters":{"path":"","recursive":true}}',
    }
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "List files in the dir"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert "tool_calls" in choice["message"]
    assert "content" not in choice["message"] or choice["message"]["content"] is None

    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "list_files"
    args = json.loads(tool_call["function"]["arguments"])
    assert args["recursive"] is True


def test_chat_completions_streaming(mocker):
    """POST /v1/chat/completions with stream=True returns text/event-stream SSE."""
    lines = [
        'data: {"data":"Hello"}',
        'data: {"data":"{\\"command\\":\\"foo\\",\\"parameters\\":{}}"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    assert "data:" in body
    assert "[DONE]" in body
    assert "Hello" in body
    assert "tool_calls" in body
    assert "foo" in body


def test_chat_completions_custom_params(mocker):
    """POST /v1/chat/completions forwards temperature and max_tokens to Amplify."""
    captured = {}

    async def fake_post(url, **kwargs):
        captured["payload"] = kwargs.get("json", {})
        m = mocker.Mock()
        m.raise_for_status = mocker.Mock()
        m.json.return_value = {"success": True, "data": "ok"}
        return m

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)

    req_body = {
        "model": "claude-3",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert captured["payload"]["data"]["temperature"] == 0.2
    assert captured["payload"]["data"]["max_tokens"] == 512
    assert captured["payload"]["data"]["options"]["model"]["id"] == "claude-3"


def test_chat_completions_tool_calls_json_format(mocker):
    """POST /v1/chat/completions converts tool_calls in history to JSON format for Amplify."""
    captured = {}

    async def fake_post(url, **kwargs):
        captured["payload"] = kwargs.get("json", {})
        m = mocker.Mock()
        m.raise_for_status = mocker.Mock()
        m.json.return_value = {"success": True, "data": "ok"}
        return m

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)

    req_body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": '{"path": "/home", "recursive": true}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "list_files",
                "content": "file1.txt\nfile2.txt",
            },
            {"role": "user", "content": "Now read file1.txt"},
        ],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200

    amplify_messages = captured["payload"]["data"]["messages"]
    assert len(amplify_messages) == 4

    assistant_msg = amplify_messages[1]
    assert assistant_msg["role"] == "assistant"
    assert '{"tool": "list_files"' in assistant_msg["content"]
    assert '"parameters"' in assistant_msg["content"]
    assert "[Tool Call:" not in assistant_msg["content"]

    tool_msg = amplify_messages[2]
    assert tool_msg["role"] == "user"
    assert '{"tool_result": "list_files"' in tool_msg["content"]
    assert "[Tool Result:" not in tool_msg["content"]


def test_chat_completions_legacy_tool_call_parsing(mocker):
    """POST /v1/chat/completions defensively parses legacy [Tool Call: ...] format from LLM."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": '[Tool Call: update_todo_list]\nParameters: {"todos": "[x] Done"}',
    }
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Update the todo list"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert "tool_calls" in choice["message"]

    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "update_todo_list"
    args = json.loads(tool_call["function"]["arguments"])
    assert "todos" in args


@pytest.mark.parametrize(
    "xml_content,expected_name,expected_args",
    [
        (
            '<tool_call>\n<tool_name>list_files</tool_name>\n<parameters>\n<path>docs-vibe</path>\n<recursive>false</recursive>\n</parameters>\n</tool_call>',
            "list_files",
            {"path": "docs-vibe", "recursive": False},
        ),
        (
            '<tool_use>\n<tool_name>update_todo_list</tool_name>\n<parameters>\n<todos>[x] Done\n[ ] Next</todos>\n</parameters>\n</tool_use>',
            "update_todo_list",
            {"todos": "[x] Done\n[ ] Next"},
        ),
        (
            '<tool_call>\n<tool_name>read_file</tool_name>\n<parameters>\n<path>src/main.py</path>\n</parameters>\n</tool_call>',
            "read_file",
            {"path": "src/main.py"},
        ),
        (
            '<tool_call>\n<tool_name>execute_command</tool_name>\n<parameters>\n<command>ls -la</command>\n<cwd></cwd>\n</parameters>\n</tool_call>',
            "execute_command",
            {"command": "ls -la", "cwd": ""},
        ),
    ],
)
def test_chat_completions_xml_tool_call_parsing(mocker, xml_content, expected_name, expected_args):
    """POST /v1/chat/completions parses XML format tool calls from Opus model."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": xml_content,
    }
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    req_body = {
        "model": "us.anthropic.claude-opus-4-6-v1",
        "messages": [{"role": "user", "content": "Execute tool"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert "tool_calls" in choice["message"]

    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == expected_name
    args = json.loads(tool_call["function"]["arguments"])
    assert args == expected_args


def test_chat_completions_streaming_legacy_tool_call(mocker):
    """POST /v1/chat/completions defensively parses legacy [Tool Call: ...] in streaming."""
    lines = [
        '[Tool Call: read_file]\nParameters: {"path": "/tmp/test.txt"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Read the file"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200

    body = response.text
    has_tool_call = False
    for line in body.strip().split("\n"):
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if "tool_calls" in delta:
            has_tool_call = True
            tc = delta["tool_calls"][0]
            assert tc["function"]["name"] == "read_file"

    assert has_tool_call, "No tool_calls chunk found in streaming response for legacy format"
