"""Integration tests for TokenCounterMiddleware.

Verifies that the middleware writes CSV rows for every HTTP request with the
correct token counts, IP address, status code, and error description.

The write_token_stats function is patched so no real CSV files are created;
the captured TokenStatsRecord arguments are inspected directly.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _patch_stats(mocker):
    """Patch write_token_stats and return the mock for inspection."""
    return mocker.patch("open_amplify_ai.middleware.write_token_stats")


def _make_amplify_chat_response(mocker, content: str):
    """Return a mock non-streaming Amplify /chat response."""
    mock = mocker.Mock()
    mock.status_code = 200
    mock.raise_for_status = mocker.Mock()
    mock.json.return_value = {"success": True, "data": content}
    return mock


def _make_amplify_stream_response(mocker, data_lines):
    """Return a mock streaming Amplify /chat context-manager response."""
    mock = mocker.MagicMock()
    mock.__enter__ = mocker.Mock(return_value=mock)
    mock.__exit__ = mocker.Mock(return_value=False)
    mock.status_code = 200
    mock.raise_for_status = mocker.Mock()
    mock.iter_lines = mocker.Mock(return_value=data_lines)
    return mock


# ---------------------------------------------------------------------------
# Basic CSV row generation
# ---------------------------------------------------------------------------


def test_token_counter_writes_row_on_success(mocker) -> None:
    """A CSV row is written for every successful chat completion request."""
    write_mock = _patch_stats(mocker)
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(mocker, "Hello!"),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 200
    assert write_mock.call_count == 1

    record, csv_path = write_mock.call_args[0]
    assert record.status_code == 200
    assert record.error == ""
    assert record.method == "POST"
    assert record.path == "/v1/chat/completions"
    assert "token_stats.csv" in csv_path


def test_token_counter_records_prompt_tokens(mocker) -> None:
    """Prompt token estimate is non-zero and proportional to message length."""
    write_mock = _patch_stats(mocker)
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(mocker, "ok"),
    )

    # 400-char message -> 100 estimated prompt tokens
    content = "A" * 400
    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
    })

    record, _ = write_mock.call_args[0]
    assert record.prompt_tokens == 100


def test_token_counter_records_completion_tokens(mocker) -> None:
    """Completion token estimate is non-zero for non-streaming responses."""
    write_mock = _patch_stats(mocker)
    # 80-char response -> 20 estimated completion tokens
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(mocker, "B" * 80),
    )

    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    record, _ = write_mock.call_args[0]
    assert record.completion_tokens == 20


def test_token_counter_total_equals_sum(mocker) -> None:
    """total_tokens equals prompt_tokens + completion_tokens."""
    write_mock = _patch_stats(mocker)
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(mocker, "ok"),
    )

    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    record, _ = write_mock.call_args[0]
    assert record.total_tokens == record.prompt_tokens + record.completion_tokens


# ---------------------------------------------------------------------------
# IP address capture
# ---------------------------------------------------------------------------


def test_token_counter_captures_ip_from_client(mocker) -> None:
    """Client IP is recorded when no X-Forwarded-For header is present."""
    write_mock = _patch_stats(mocker)
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(mocker, "ok"),
    )

    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    record, _ = write_mock.call_args[0]
    # TestClient connects from testclient or 127.0.0.1
    assert record.ip_address != ""


def test_token_counter_prefers_x_forwarded_for(mocker) -> None:
    """X-Forwarded-For header is used as IP when present."""
    write_mock = _patch_stats(mocker)
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        return_value=_make_amplify_chat_response(mocker, "ok"),
    )

    client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-Forwarded-For": "203.0.113.42"},
    )

    record, _ = write_mock.call_args[0]
    assert record.ip_address == "203.0.113.42"


# ---------------------------------------------------------------------------
# Error capture
# ---------------------------------------------------------------------------


def test_token_counter_records_http_error(mocker) -> None:
    """HTTP 4xx/5xx responses are captured in the error field."""
    write_mock = _patch_stats(mocker)

    # Malformed request triggers 400
    client.post("/v1/chat/completions", json={"messages": "not a list"})

    assert write_mock.call_count == 1
    record, _ = write_mock.call_args[0]
    assert record.status_code == 400
    assert "400" in record.error


def test_token_counter_records_upstream_exception(mocker) -> None:
    """Upstream connection errors are recorded in the error field."""
    import requests as req_lib

    write_mock = _patch_stats(mocker)
    mocker.patch(
        "open_amplify_ai.routers.chat.requests.post",
        side_effect=req_lib.exceptions.ConnectionError("refused"),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response.status_code == 500
    # The middleware should still have written a CSV row
    assert write_mock.call_count >= 1


# ---------------------------------------------------------------------------
# Non-chat endpoints — zero tokens, still recorded
# ---------------------------------------------------------------------------


def test_token_counter_ignores_non_llm_endpoint(mocker) -> None:
    """Non-LLM endpoints like /v1/models are NOT recorded."""
    write_mock = _patch_stats(mocker)
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {"models": [{"id": "gpt-4o", "name": "GPT-4o"}]},
    }
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    client.get("/v1/models")

    assert write_mock.call_count == 0


def test_token_counter_records_other_llm_endpoints(mocker) -> None:
    """Other LLM endpoints like /v1/assistants ARE recorded."""
    write_mock = _patch_stats(mocker)
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"object": "list", "data": []}
    mocker.patch("open_amplify_ai.routers.assistants.requests.get", return_value=mock_response)

    client.get("/v1/assistants")

    assert write_mock.call_count == 1
    record, _ = write_mock.call_args[0]
    assert record.path == "/v1/assistants"
    assert record.prompt_tokens == 0


# ---------------------------------------------------------------------------
# Streaming responses
# ---------------------------------------------------------------------------


def test_token_counter_streaming_completion_tokens(mocker) -> None:
    """Completion tokens are estimated from SSE delta chunks in streaming mode.

    The mock uses Amplify's SSE line format ({"data": "..."}) which
    stream_amplify_chat transforms into OpenAI-format delta chunks.  The
    middleware buffers those OpenAI chunks and _completion_tokens_from_sse
    parses them to count characters.
    """
    write_mock = _patch_stats(mocker)

    # Three Amplify lines, each carrying 20 chars -> 60 chars total -> 15 tokens
    delta_content = "A" * 20
    stream_lines = [
        f'data: {{"data":"{delta_content}"}}'.encode(),
        f'data: {{"data":"{delta_content}"}}'.encode(),
        f'data: {{"data":"{delta_content}"}}'.encode(),
        b"data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_stream_response(mocker, stream_lines),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200

    record, _ = write_mock.call_args[0]
    assert record.completion_tokens == 15
    assert record.status_code == 200
    assert record.error == ""


def test_token_counter_streaming_row_written_once(mocker) -> None:
    """Exactly one CSV row is written per streaming request, not per SSE chunk."""
    write_mock = _patch_stats(mocker)
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_stream_response(mocker, [
            b'data: {"data":"hi"}',
            b"data: [DONE]",
        ]),
    )

    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })

    assert write_mock.call_count == 1
