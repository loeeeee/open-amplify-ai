"""Unit tests for chat endpoint response invariants.

Tests assertions that should always hold true for responses.
Covers the response invariants from the test refactor plan.
"""
import json
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app
import os

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_async_client(mocker, response):
    """Build an async httpx client mock."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_streaming_client(mocker, lines):
    """Build an async httpx streaming client mock."""
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
    mock_client.stream = mocker.Mock(return_value=mock_stream_cm)
    return mock_client


def test_non_streaming_object_type(mocker):
    """Test that non-streaming response has object='chat.completion'."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"


def test_non_streaming_has_non_empty_choices(mocker):
    """Test that non-streaming response has non-empty choices array."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    assert "choices" in data
    assert isinstance(data["choices"], list)
    assert len(data["choices"]) > 0


def test_non_streaming_usage_token_sum(mocker):
    """Test that usage.total_tokens == prompt_tokens + completion_tokens."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    if "usage" in data:
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        
        # Verify the sum
        expected_total = usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["total_tokens"] == expected_total


def test_non_streaming_choice_index_is_stable(mocker):
    """Test that choice index values are stable (0 for single choice)."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    for i, choice in enumerate(data["choices"]):
        assert "index" in choice
        assert choice["index"] == i


def test_non_streaming_returned_model_matches(mocker):
    """Test that returned model matches requested model."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    requested_model = "gpt-4o"
    response = client.post(
        "/v1/chat/completions",
        json={"model": requested_model, "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    assert "model" in data
    assert data["model"] == requested_model


def test_non_streaming_has_required_fields(mocker):
    """Test that non-streaming response has all required OpenAI fields."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    # Required fields per OpenAI spec
    assert "id" in data
    assert "object" in data
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert "usage" in data
    
    # ID should start with appropriate prefix
    assert data["id"].startswith("chatcmpl-")
    
    # Created should be a timestamp
    assert isinstance(data["created"], int)
    assert data["created"] > 0


def test_non_streaming_choice_has_required_fields(mocker):
    """Test that each choice has required fields."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    for choice in data["choices"]:
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        
        # Message must have role
        assert "role" in choice["message"]
        assert choice["message"]["role"] == "assistant"


def test_streaming_object_type_is_chunk(mocker):
    """Test that streaming chunks have object='chat.completion.chunk'."""
    lines = ['data: {"data":"Hello"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            assert chunk["object"] == "chat.completion.chunk"


def test_streaming_chunk_ids_constant(mocker):
    """Test that chunk IDs stay constant across one stream."""
    lines = [
        'data: {"data":"Hello"}',
        'data: {"data":" world"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    chunk_ids = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            if "id" in chunk:
                chunk_ids.append(chunk["id"])
    
    # All IDs should be the same
    if len(chunk_ids) > 1:
        assert all(chunk_id == chunk_ids[0] for chunk_id in chunk_ids)


def test_streaming_choice_index_stable(mocker):
    """Test that choice index is stable across streaming chunks."""
    lines = [
        'data: {"data":"Hello"}',
        'data: {"data":" world"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    indices = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            if "choices" in chunk and len(chunk["choices"]) > 0:
                indices.append(chunk["choices"][0]["index"])
    
    # All indices should be 0
    assert all(idx == 0 for idx in indices)


def test_streaming_exactly_one_done(mocker):
    """Test that stream contains exactly one [DONE] marker."""
    lines = ['data: {"data":"Hello"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    done_count = body.count("data: [DONE]")
    assert done_count == 1


def test_streaming_no_data_after_done(mocker):
    """Test that no data appears after [DONE] marker."""
    lines = ['data: {"data":"Hello"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    lines_list = body.strip().split("\n")
    
    done_index = -1
    for i, line in enumerate(lines_list):
        if line == "data: [DONE]":
            done_index = i
            break
    
    if done_index != -1:
        # Check no data lines after [DONE]
        for i in range(done_index + 1, len(lines_list)):
            line = lines_list[i]
            if line.strip() and line.startswith("data: "):
                assert False, f"Found data after [DONE]: {line}"


def test_finish_reason_has_valid_value(mocker):
    """Test that finish_reason is one of the valid values."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    valid_reasons = ["stop", "length", "tool_calls", "content_filter", "function_call"]
    
    for choice in data["choices"]:
        assert choice["finish_reason"] in valid_reasons


def test_system_fingerprint_present(mocker):
    """Test that system_fingerprint is present in response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Hello"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    assert "system_fingerprint" in data
