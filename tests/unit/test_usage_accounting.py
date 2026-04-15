"""Tests for local usage accounting in chat completions.

Verifies that token counts in chat completion responses are non-zero
estimates computed from the actual request/response content, not
hardcoded zeroes.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from open_amplify_ai.server import app
from open_amplify_ai.token_counting import (
    calculate_cost,
    count_completion_tokens,
    count_message_tokens,
    count_prompt_tokens,
    estimate_tokens,
)

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_async_client(mocker, response):
    """Build an async httpx client mock for non-streaming calls."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_json_response(mocker, json_data):
    """Build a mock httpx response."""
    mock = mocker.Mock()
    mock.status_code = 200
    mock.raise_for_status = mocker.Mock()
    mock.json.return_value = json_data
    return mock


def _make_streaming_client(mocker, lines):
    """Build an async httpx streaming client mock."""
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
# Unit tests for token_counting module
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty() -> None:
    """Empty string returns 0."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_none() -> None:
    """None input returns 0."""
    assert estimate_tokens(None) == 0


def test_estimate_tokens_basic() -> None:
    """80 characters yields 20 tokens (chars/4)."""
    assert estimate_tokens("A" * 80) == 20


def test_estimate_tokens_short() -> None:
    """3 characters yields 0 tokens (truncated integer division)."""
    assert estimate_tokens("abc") == 0


def test_count_message_tokens_single() -> None:
    """Single message with 80-char content: 20 content + 4 overhead = 24."""
    messages = [{"role": "user", "content": "A" * 80}]
    assert count_message_tokens(messages) == 24


def test_count_message_tokens_multiple() -> None:
    """Multiple messages each contribute content tokens plus overhead."""
    messages = [
        {"role": "system", "content": "A" * 40},  # 10 + 4 = 14
        {"role": "user", "content": "B" * 80},    # 20 + 4 = 24
    ]
    assert count_message_tokens(messages) == 38


def test_count_message_tokens_empty_content() -> None:
    """Empty content still contributes the 4-token overhead."""
    messages = [{"role": "user", "content": ""}]
    assert count_message_tokens(messages) == 4


def test_count_prompt_tokens_amplify_request() -> None:
    """Count from a fully rendered Amplify request dict."""
    request = {
        "data": {
            "messages": [
                {"role": "user", "content": "A" * 80},
            ],
            "temperature": 0.7,
        }
    }
    assert count_prompt_tokens(request) == 24  # 20 + 4 overhead


def test_count_prompt_tokens_with_tool_injection() -> None:
    """System message with injected tool schema produces substantial token count."""
    tool_schema = json.dumps([{"type": "function", "function": {"name": "test"}}])
    request = {
        "data": {
            "messages": [
                {"role": "system", "content": "You are helpful. " + tool_schema},
                {"role": "user", "content": "Hi"},
            ],
        }
    }
    tokens = count_prompt_tokens(request)
    # Should be more than just "Hi" alone
    assert tokens > 4


def test_count_completion_tokens_basic() -> None:
    """80 characters yields 20 tokens."""
    assert count_completion_tokens("A" * 80) == 20


def test_count_completion_tokens_empty() -> None:
    """Empty string yields 0."""
    assert count_completion_tokens("") == 0


# ---------------------------------------------------------------------------
# Integration tests: non-streaming usage accounting
# ---------------------------------------------------------------------------


def test_non_streaming_usage_nonzero(mocker) -> None:
    """Non-streaming response has non-zero usage values."""
    resp = _make_json_response(mocker, {"success": True, "data": "A" * 80})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello world"}],
    })
    assert response.status_code == 200
    data = response.json()

    usage = data["usage"]
    assert usage["prompt_tokens"] > 0, "prompt_tokens should be non-zero"
    assert usage["completion_tokens"] > 0, "completion_tokens should be non-zero"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_non_streaming_prompt_tokens_reflect_transformation(mocker) -> None:
    """Prompt tokens account for the post-transformation Amplify request.

    The request includes system prompt injection for tools, so prompt
    tokens should be substantially larger than raw user message alone.
    """
    resp = _make_json_response(mocker, {"success": True, "data": "ok"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    # Request with tools -- transformation injects tool protocol into system message
    response_with_tools = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
    })
    assert response_with_tools.status_code == 200
    with_tools_prompt = response_with_tools.json()["usage"]["prompt_tokens"]

    # Request without tools
    response_no_tools = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    assert response_no_tools.status_code == 200
    no_tools_prompt = response_no_tools.json()["usage"]["prompt_tokens"]

    assert with_tools_prompt > no_tools_prompt, (
        "Tool injection should increase prompt token count"
    )


def test_non_streaming_completion_tokens_proportional(mocker) -> None:
    """Completion tokens scale with response length."""
    short_resp = _make_json_response(mocker, {"success": True, "data": "ok"})
    long_resp = _make_json_response(mocker, {"success": True, "data": "A" * 400})

    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, short_resp),
    )
    r_short = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, long_resp),
    )
    r_long = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    short_completion = r_short.json()["usage"]["completion_tokens"]
    long_completion = r_long.json()["usage"]["completion_tokens"]

    assert long_completion > short_completion, (
        "Longer response should have more completion tokens"
    )


# ---------------------------------------------------------------------------
# Integration tests: streaming usage accounting
# ---------------------------------------------------------------------------


def test_streaming_usage_chunk_has_nonzero_values(mocker) -> None:
    """Streaming usage chunk has non-zero token counts when include_usage is true."""
    lines = [
        'data: {"data":"' + "A" * 80 + '"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.streaming.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello world"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert response.status_code == 200

    body = response.text
    usage_chunk = None
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunk = json.loads(line[6:])
            if chunk.get("choices") == [] and "usage" in chunk:
                usage_chunk = chunk

    assert usage_chunk is not None, "No usage chunk found"
    usage = usage_chunk["usage"]
    assert usage["prompt_tokens"] > 0, "prompt_tokens should be non-zero"
    assert usage["completion_tokens"] > 0, "completion_tokens should be non-zero"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_streaming_no_usage_chunk_without_include_usage(mocker) -> None:
    """No usage chunk is emitted when stream_options.include_usage is not set."""
    lines = [
        'data: {"data":"Hello"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.streaming.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert response.status_code == 200

    body = response.text
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunk = json.loads(line[6:])
            assert "usage" not in chunk, "Usage chunk should not appear without include_usage"


# ---------------------------------------------------------------------------
# Unit tests for calculate_cost
# ---------------------------------------------------------------------------


def test_calculate_cost_basic() -> None:
    """Cost is computed from prompt and completion tokens with given pricing."""
    # 1000 prompt tokens at $3/M input, 500 completion tokens at $15/M output
    # cost = 1000*3/1_000_000 + 500*15/1_000_000 = 0.003 + 0.0075 = 0.0105
    result = calculate_cost(1000, 500, 3.0, 15.0)
    assert result is not None
    assert abs(result - 0.0105) < 1e-9


def test_calculate_cost_zero_tokens() -> None:
    """Zero tokens yields zero cost."""
    result = calculate_cost(0, 0, 3.0, 15.0)
    assert result == 0.0


def test_calculate_cost_missing_input_price() -> None:
    """Returns None when input pricing is absent."""
    assert calculate_cost(100, 50, None, 15.0) is None


def test_calculate_cost_missing_output_price() -> None:
    """Returns None when output pricing is absent."""
    assert calculate_cost(100, 50, 3.0, None) is None


def test_calculate_cost_both_prices_missing() -> None:
    """Returns None when both pricing values are absent."""
    assert calculate_cost(100, 50, None, None) is None


# ---------------------------------------------------------------------------
# Integration tests: prompt_tokens_details and cost in non-streaming response
# ---------------------------------------------------------------------------


def test_non_streaming_usage_has_prompt_tokens_details(mocker) -> None:
    """Non-streaming response includes prompt_tokens_details with cached_tokens."""
    resp = _make_json_response(mocker, {"success": True, "data": "Hello world"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello world"}],
    })
    assert response.status_code == 200
    data = response.json()

    usage = data["usage"]
    assert "prompt_tokens_details" in usage, "prompt_tokens_details must be present"
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0


def test_non_streaming_usage_cost_absent_without_model_pricing(mocker) -> None:
    """Cost field is absent when model metadata cannot be fetched."""
    resp = _make_json_response(mocker, {"success": True, "data": "Hello world"})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )
    # Patch model metadata to return None (no pricing available)
    mocker.patch(
        "open_amplify_ai.routers.chat.get_model_metadata",
        return_value=None,
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert "cost" not in usage, "cost should be absent when pricing is unavailable"


def test_non_streaming_usage_cost_present_with_model_pricing(mocker) -> None:
    """Cost field is present and non-negative when model metadata provides pricing."""
    resp = _make_json_response(mocker, {"success": True, "data": "A" * 80})
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )
    mocker.patch(
        "open_amplify_ai.routers.chat.get_model_metadata",
        return_value={
            "inputTokenCost": 3.0,
            "outputTokenCost": 15.0,
        },
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello world"}],
    })
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert "cost" in usage, "cost should be present when pricing is available"
    assert usage["cost"] >= 0.0


# ---------------------------------------------------------------------------
# Integration tests: prompt_tokens_details and cost in streaming response
# ---------------------------------------------------------------------------


def test_streaming_usage_chunk_has_prompt_tokens_details(mocker) -> None:
    """Streaming usage chunk includes prompt_tokens_details.cached_tokens."""
    lines = [
        'data: {"data":"' + "A" * 80 + '"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.streaming.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )

    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello world"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert response.status_code == 200

    usage_chunk = None
    for line in response.text.strip().split("\n"):
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunk = json.loads(line[6:])
            if chunk.get("choices") == [] and "usage" in chunk:
                usage_chunk = chunk

    assert usage_chunk is not None, "No usage chunk found"
    usage = usage_chunk["usage"]
    assert "prompt_tokens_details" in usage
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
