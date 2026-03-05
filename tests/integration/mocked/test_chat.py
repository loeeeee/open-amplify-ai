"""Comprehensive integration tests simulating cline/kilo/openclaw usage patterns.

These tests exercise the full FastAPI request/response cycle with mocked
Amplify upstream calls. No live AMPLIFY_AI_TOKEN is needed.

Covers all endpoint groups:
  1. Model discovery and retrieval
  2. Chat completions (simple, streaming, tool calls, multi-turn)
  3. File operations (upload, list, retrieve, delete, content download)
  4. Assistant lifecycle (create, list, retrieve, modify, delete)
  5. Thread deletion
  6. Vector store lifecycle (create, retrieve, delete, files)
  7. Edge cases and error handling
  8. Unsupported endpoint stubs (501)

To run:
    nix-shell --run "uv run pytest src/open_amplify_ai/test_chat_client_integration.py -v"
"""
import io
import json
import os
import pytest
from typing import Any, Dict, List
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_amplify_models_response(models: List[Dict[str, Any]]) -> Any:
    """Build a mock response for Amplify GET /available_models."""
    mock = type("MockResponse", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: {"success": True, "data": {"models": models}},
    })()
    return mock


def _make_amplify_chat_response(content: str) -> Any:
    """Build a mock response for Amplify POST /chat (non-streaming)."""
    mock = type("MockResponse", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: {"success": True, "data": content},
    })()
    return mock


def _make_amplify_stream_response(data_lines: List[bytes]) -> Any:
    """Build a mock context-manager response for Amplify POST /chat (streaming)."""
    mock = type("MockStreamResponse", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "iter_lines": lambda self: iter(data_lines),
        "__enter__": lambda self: self,
        "__exit__": lambda self, *args: False,
    })()
    return mock


# ---------------------------------------------------------------------------
# 1. Model discovery - cline/kilo list models on startup
# ---------------------------------------------------------------------------


def test_client_discovers_models(mocker: Any) -> None:
    """Chat client enumerates available models on startup (cline/kilo pattern)."""
    mocker.patch(
        "open_amplify_ai.routers.models.requests.get",
        return_value=_make_amplify_models_response([
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "Azure"},
            {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "AWS"},
        ]),
    )

    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2

    model_ids = [m["id"] for m in data["data"]]
    assert "gpt-4o" in model_ids
    assert "claude-3.5-sonnet" in model_ids

    for model in data["data"]:
        assert model["object"] == "model"
        assert model["owned_by"] == "amplify-ai"
        assert "created" in model


# ---------------------------------------------------------------------------
# 2. Simple chat - single user message, non-streaming
# ---------------------------------------------------------------------------


def test_client_simple_chat(mocker: Any) -> None:
    """Chat client sends a single user message without streaming."""
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response("4"),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    })
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "gpt-4o"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "4"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data
    assert data["id"].startswith("chatcmpl-")
    assert "system_fingerprint" in data


# ---------------------------------------------------------------------------
# 3. System prompt chat - cline always prepends a system message
# ---------------------------------------------------------------------------


def test_client_system_prompt_chat(mocker: Any) -> None:
    """Cline sends a system role message followed by the user prompt."""
    captured: Dict[str, Any] = {}

    def _capture_post(url: str, headers: Any, json: Any, timeout: int) -> Any:
        """Intercept the Amplify POST to verify payload shape."""
        captured["payload"] = json
        return _make_amplify_chat_response("Hello! How can I help?")

    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        side_effect=_capture_post,
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.0,
    })
    assert response.status_code == 200

    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"

    # Verify both messages were forwarded to Amplify
    amplify_messages = captured["payload"]["data"]["messages"]
    assert len(amplify_messages) == 2
    assert amplify_messages[0]["role"] == "system"
    assert amplify_messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# 4. List-typed content - cline passes content as [{type, text}] parts
# ---------------------------------------------------------------------------


def test_client_list_content_chat(mocker: Any) -> None:
    """Cline passes content as a list of typed parts; server must flatten to string."""
    captured: Dict[str, Any] = {}

    def _capture_post(url: str, headers: Any, json: Any, timeout: int) -> Any:
        """Intercept the Amplify POST to verify flattened content."""
        captured["payload"] = json
        return _make_amplify_chat_response("4")

    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        side_effect=_capture_post,
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is "},
                {"type": "text", "text": "2+2?"},
            ],
        }],
    })
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "4"

    # Verify content was flattened into a single string for Amplify
    amplify_content = captured["payload"]["data"]["messages"][0]["content"]
    assert amplify_content == "What is 2+2?"


# ---------------------------------------------------------------------------
# 5. Extra message fields - kilo includes `name` on messages
# ---------------------------------------------------------------------------


def test_client_extra_fields_chat(mocker: Any) -> None:
    """Kilo sends messages with extra fields like 'name'; server must not reject."""
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response("ok"),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": "Hello",
            "name": "kilo-agent",
        }],
    })
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"


# ---------------------------------------------------------------------------
# 6. Streaming response - cline/kilo default to stream=True
# ---------------------------------------------------------------------------


def test_client_streaming_chat(mocker: Any) -> None:
    """Chat client uses stream=True and expects text/event-stream SSE chunks."""
    stream_lines = [
        b'data: {"data":"Hello"}',
        b'data: {"data":" world"}',
        b"data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_stream_response(stream_lines),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    sse_lines = [l for l in body.strip().split("\n") if l.startswith("data: ")]

    # Should have content chunks + finish chunk + [DONE]
    assert len(sse_lines) >= 3

    # Parse content chunks
    content_parts: List[str] = []
    for line in sse_lines:
        payload = line[6:]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if "content" in delta:
            content_parts.append(delta["content"])

    assert "Hello" in " ".join(content_parts)

    # Final line should be [DONE]
    assert sse_lines[-1] == "data: [DONE]"


# ---------------------------------------------------------------------------
# 7. Streaming with usage - kilo/cline expect final usage chunk
# ---------------------------------------------------------------------------


def test_client_streaming_with_usage(mocker: Any) -> None:
    """Kilo/cline use stream_options.include_usage=True and expects a usage chunk."""
    stream_lines = [
        b'data: {"data":"Hi"}',
        b"data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_stream_response(stream_lines),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert response.status_code == 200

    body = response.text
    has_usage = False
    has_done = False

    for line in body.strip().split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            has_done = True
            continue
        chunk = json.loads(payload)
        if chunk.get("choices") == [] and "usage" in chunk:
            has_usage = True
            assert "prompt_tokens" in chunk["usage"]
            assert "completion_tokens" in chunk["usage"]
            assert "total_tokens" in chunk["usage"]

    assert has_usage, "Missing final usage chunk (stream_options.include_usage=True)"
    assert has_done, "Stream did not terminate with [DONE]"


# ---------------------------------------------------------------------------
# 8. Tool call detection - kilo expects JSON tool calls parsed
# ---------------------------------------------------------------------------


def test_client_tool_call_detection(mocker: Any) -> None:
    """Server detects JSON tool call in response and converts to tool_calls block."""
    tool_response = '{"tool":"list_files","parameters":{"path":"/home","recursive":true}}'
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(tool_response),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "List files in home dir"}],
    })
    assert response.status_code == 200

    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"

    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1

    tc = tool_calls[0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "list_files"

    args = json.loads(tc["function"]["arguments"])
    assert args["path"] == "/home"
    assert args["recursive"] is True

    # content should be absent or None when tool calls are present
    assert choice["message"].get("content") is None


# ---------------------------------------------------------------------------
# 9. Streaming tool call - tool call detection in streaming mode
# ---------------------------------------------------------------------------


def test_client_streaming_tool_call(mocker: Any) -> None:
    """Server detects JSON tool call in streaming response and emits tool_calls chunk."""
    tool_json = '{"tool":"read_file","parameters":{"path":"/tmp/test.txt"}}'
    stream_lines = [
        f'data: {{"data":"{tool_json.replace(chr(34), chr(92)+chr(34))}"}}'.encode(),
        b"data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_stream_response(stream_lines),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Read the file"}],
        "stream": True,
    })
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

    assert has_tool_call, "No tool_calls chunk found in streaming response"


# ---------------------------------------------------------------------------
# 10. Multi-turn conversation - cline sends full context window
# ---------------------------------------------------------------------------


def test_client_multi_turn_conversation(mocker: Any) -> None:
    """Cline sends multiple user/assistant turns; all are forwarded to Amplify."""
    captured: Dict[str, Any] = {}

    def _capture_post(url: str, headers: Any, json: Any, timeout: int) -> Any:
        """Intercept Amplify POST to verify all turns are forwarded."""
        captured["payload"] = json
        return _make_amplify_chat_response("The file contains test data.")

    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        side_effect=_capture_post,
    )

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Read test.txt"},
        {"role": "assistant", "content": "I will read the file for you."},
        {"role": "user", "content": "What does it contain?"},
    ]

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": messages,
    })
    assert response.status_code == 200

    # All 4 messages should be forwarded to Amplify
    amplify_messages = captured["payload"]["data"]["messages"]
    assert len(amplify_messages) == 4
    assert amplify_messages[0]["role"] == "system"
    assert amplify_messages[1]["role"] == "user"
    assert amplify_messages[2]["role"] == "assistant"
    assert amplify_messages[3]["role"] == "user"


# ---------------------------------------------------------------------------
# 11. Custom params forwarded - temperature and max_tokens
# ---------------------------------------------------------------------------


def test_client_custom_params_forwarded(mocker: Any) -> None:
    """Temperature and max_tokens are forwarded to the Amplify request."""
    captured: Dict[str, Any] = {}

    def _capture_post(url: str, headers: Any, json: Any, timeout: int) -> Any:
        """Intercept Amplify POST to verify custom params."""
        captured["payload"] = json
        return _make_amplify_chat_response("ok")

    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        side_effect=_capture_post,
    )

    response = client.post("/v1/chat/completions", json={
        "model": "claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "max_tokens": 512,
    })
    assert response.status_code == 200

    amplify_data = captured["payload"]["data"]
    assert amplify_data["temperature"] == 0.2
    assert amplify_data["max_tokens"] == 512
    assert amplify_data["options"]["model"]["id"] == "claude-3.5-sonnet"


# ---------------------------------------------------------------------------
# 12. Error handling - Amplify returns failure
# ---------------------------------------------------------------------------


def test_client_amplify_error_handling(mocker: Any) -> None:
    """Server returns 500 when Amplify upstream signals failure."""
    import requests as req_lib
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        side_effect=req_lib.exceptions.ConnectionError("Connection refused"),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert response.status_code == 500
    assert "Error communicating with Amplify AI" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 13. Malformed request - invalid body returns 400
# ---------------------------------------------------------------------------


def test_client_malformed_request() -> None:
    """Server returns 400 when the request body is malformed."""
    response = client.post("/v1/chat/completions", json={
        "messages": "this is not a list",
    })
    assert response.status_code == 400


# ===========================================================================
# MODELS - Edge cases
# ===========================================================================


def test_client_retrieves_single_model(mocker: Any) -> None:
    """Client retrieves a single model by ID (filters from full list)."""
    mocker.patch(
        "open_amplify_ai.routers.models.requests.get",
        return_value=_make_amplify_models_response([
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
        ]),
    )

    response = client.get("/v1/models/gpt-4o")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "gpt-4o"
    assert data["object"] == "model"
    assert data["owned_by"] == "amplify-ai"
    assert "created" in data


def test_client_retrieves_model_not_found(mocker: Any) -> None:
    """Client gets 404 when requesting a model that does not exist."""
    mocker.patch(
        "open_amplify_ai.routers.models.requests.get",
        return_value=_make_amplify_models_response([
            {"id": "gpt-4o", "name": "GPT-4o"},
        ]),
    )

    response = client.get("/v1/models/nonexistent-model")
    assert response.status_code == 404


def test_client_delete_model_rejected() -> None:
    """DELETE /v1/models/{model} always returns 405 (not supported by Amplify)."""
    response = client.delete("/v1/models/gpt-4o")
    assert response.status_code == 405


# ===========================================================================
# CHAT - Edge cases
# ===========================================================================


def test_client_chat_empty_content(mocker: Any) -> None:
    """Server handles empty string content gracefully."""
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(""),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": ""}],
    })
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == ""
    assert data["choices"][0]["finish_reason"] == "stop"
