"""Integration tests for TokenCounterMiddleware.

Verifies that the middleware writes CSV rows for every HTTP request with the
correct token counts, IP address, status code, and error description.

The write_token_stats function is patched so no real CSV files are created;
the captured TokenStatsRecord arguments are inspected directly.
"""
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _patch_stats(mocker):
    """Patch write_token_stats and return the mock for inspection."""
    return mocker.patch("open_amplify_ai.middleware.write_token_stats")


def _make_async_client(mocker, response):
    """Build an async httpx client mock for non-streaming calls."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = mocker.AsyncMock(return_value=response)
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_json_response(mocker, json_data):
    """Build a generic sync mock response."""
    mock = mocker.Mock()
    mock.status_code = 200
    mock.raise_for_status = mocker.Mock()
    mock.json.return_value = json_data
    return mock


def _make_streaming_client(mocker, lines):
    """Build an async httpx streaming client mock for utils.stream_amplify_chat.

    lines: list of str/bytes lines that aiter_lines() will yield.
    """
    async def fake_aiter_lines():
        for line in lines:
            yield line.decode("utf-8") if isinstance(line, bytes) else line

    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = mocker.Mock()
    mock_resp.aiter_lines = fake_aiter_lines

    mock_stream_cm = mocker.AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_resp

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = mocker.Mock(return_value=mock_stream_cm)
    return mock_client


# ---------------------------------------------------------------------------
# Basic CSV row generation
# ---------------------------------------------------------------------------


def test_token_counter_writes_row_on_success(mocker) -> None:
    """A CSV row is written for every successful chat completion request."""
    write_mock = _patch_stats(mocker)
    resp = _make_json_response(mocker, {"success": True, "data": "Hello!"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
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
    assert record.model == "gpt-4o"
    assert "token_stats.csv" in csv_path


def test_token_counter_records_prompt_tokens(mocker) -> None:
    """Prompt token estimate is non-zero and proportional to message length."""
    write_mock = _patch_stats(mocker)
    resp = _make_json_response(mocker, {"success": True, "data": "ok"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

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
    resp = _make_json_response(mocker, {"success": True, "data": "B" * 80})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
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
    resp = _make_json_response(mocker, {"success": True, "data": "ok"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
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
    resp = _make_json_response(mocker, {"success": True, "data": "ok"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    record, _ = write_mock.call_args[0]
    assert record.ip_address != ""


def test_token_counter_prefers_x_forwarded_for(mocker) -> None:
    """X-Forwarded-For header is used as IP when present."""
    write_mock = _patch_stats(mocker)
    resp = _make_json_response(mocker, {"success": True, "data": "ok"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
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

    client.post("/v1/chat/completions", json={"messages": "not a list"})

    assert write_mock.call_count == 1
    record, _ = write_mock.call_args[0]
    assert record.status_code == 400
    assert "400" in record.error


def test_token_counter_records_upstream_exception(mocker) -> None:
    """Upstream connection errors are recorded in the error field."""
    write_mock = _patch_stats(mocker)

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(
        side_effect=httpx.ConnectError("refused")
    )
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    # ConnectError maps to 502 per error_handling.py
    assert response.status_code == 502
    assert write_mock.call_count >= 1


# ---------------------------------------------------------------------------
# Non-chat endpoints — zero tokens, still recorded
# ---------------------------------------------------------------------------


def test_token_counter_ignores_non_llm_endpoint(mocker) -> None:
    """Non-LLM endpoints like /v1/models are NOT recorded."""
    write_mock = _patch_stats(mocker)
    resp = _make_json_response(mocker, {
        "success": True,
        "data": {"models": [{"id": "gpt-4o", "name": "GPT-4o"}]},
    })
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    client.get("/v1/models")
    assert write_mock.call_count == 0


def test_token_counter_records_other_llm_endpoints(mocker) -> None:
    """Other LLM endpoints like /v1/assistants ARE recorded."""
    write_mock = _patch_stats(mocker)
    resp = _make_json_response(mocker, {"success": True, "data": []})
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    client.get("/v1/assistants")

    assert write_mock.call_count == 1
    record, _ = write_mock.call_args[0]
    assert record.path == "/v1/assistants"
    assert record.prompt_tokens == 0


# ---------------------------------------------------------------------------
# Streaming responses
# ---------------------------------------------------------------------------


def test_token_counter_streaming_completion_tokens(mocker) -> None:
    """Completion tokens are estimated from SSE delta chunks in streaming mode."""
    write_mock = _patch_stats(mocker)

    delta_content = "A" * 20
    stream_lines = [
        f'data: {{"data":"{delta_content}"}}'.encode(),
        f'data: {{"data":"{delta_content}"}}'.encode(),
        f'data: {{"data":"{delta_content}"}}'.encode(),
        b"data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, stream_lines),
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
    stream_lines = [
        b'data: {"data":"hi"}',
        b"data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, stream_lines),
    )

    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })

    assert write_mock.call_count == 1
