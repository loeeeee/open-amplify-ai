import io
import pytest
import os
import requests
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# Set up dummy environment variable for tests to bypass token validation failure
os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)



def test_chat_completions_success(mocker):
    """POST /v1/chat/completions returns OpenAI format from Amplify /chat."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": "This is a mocked response from Amplify.",
    }
    mocker.patch("open_amplify_ai.routers.chat.requests.post", return_value=mock_response)

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
    mock_response.json.return_value = {
        "success": True,
        "data": "ok",
    }
    mocker.patch("open_amplify_ai.routers.chat.requests.post", return_value=mock_response)

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
    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        mock = mocker.Mock()
        mock.raise_for_status = mocker.Mock()
        mock.json.return_value = {"success": True, "data": "ok"}
        return mock
    mocker.patch("open_amplify_ai.routers.chat.requests.post", side_effect=fake_post)

    req_body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is "},
                    {"type": "text", "text": "Linux?"}
                ]
            }
        ]
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert captured["payload"]["data"]["messages"][0]["content"] == "What is Linux?"


def test_chat_completions_stream_options(mocker):
    """POST /v1/chat/completions with stream_options and include_usage emits a usage chunk."""
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__ = mocker.Mock(return_value=mock_cm)
    mock_cm.__exit__ = mocker.Mock(return_value=False)
    mock_cm.status_code = 200
    mock_cm.raise_for_status = mocker.Mock()
    mock_cm.iter_lines = mocker.Mock(
        return_value=[b"data: Hello"]
    )
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_cm)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    body = response.text
    # Should contain a chunk with an empty choices list and a usage block
    assert '"choices": []' in body
    assert '"usage":' in body


def test_chat_completions_tool_call_parsing(mocker):
    """POST /v1/chat/completions parses a JSON string command into structured tool_calls."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": '{"command":"list_files","parameters":{"path":"","recursive":true}}',
    }
    mocker.patch("open_amplify_ai.routers.chat.requests.post", return_value=mock_response)

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
    import json
    args = json.loads(tool_call["function"]["arguments"])
    assert args["recursive"] is True


def test_chat_completions_streaming(mocker):
    """POST /v1/chat/completions with stream=True returns text/event-stream SSE."""
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__ = mocker.Mock(return_value=mock_cm)
    mock_cm.__exit__ = mocker.Mock(return_value=False)
    mock_cm.status_code = 200
    mock_cm.raise_for_status = mocker.Mock()
    mock_cm.iter_lines = mocker.Mock(
        return_value=[
            b'data: {"data":"Hello"}',
            b'data: {"data":"{\\"command\\":\\"foo\\",\\"parameters\\":{}}"}',
            b"data: [DONE]"
        ]
    )
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_cm)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    
    body = response.text
    # Response body must contain SSE data lines and terminator
    assert "data:" in body
    assert "[DONE]" in body
    
    # We should see content parsed out
    assert 'Hello' in body
    
    # We should see tool call parsed out
    assert 'tool_calls' in body
    assert 'foo' in body


def test_chat_completions_custom_params(mocker):
    """POST /v1/chat/completions forwards temperature and max_tokens to Amplify."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        mock = mocker.Mock()
        mock.raise_for_status = mocker.Mock()
        mock.json.return_value = {"success": True, "data": "ok"}
        return mock

    mocker.patch("open_amplify_ai.routers.chat.requests.post", side_effect=fake_post)

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

