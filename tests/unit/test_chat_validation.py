"""Unit tests for chat endpoint request validation.

Tests comprehensive validation of malformed or boundary requests.
Covers the request validation matrix from the test refactor plan.
"""
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app
import os

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


@pytest.mark.parametrize(
    "request_body,expected_status,error_type",
    [
        # Missing messages
        (
            {"model": "gpt-4o"},
            422,
            "validation_error",
        ),
        # messages is not a list
        (
            {"model": "gpt-4o", "messages": "not a list"},
            400,
            "invalid_request_error",
        ),
        # Empty messages list
        (
            {"model": "gpt-4o", "messages": []},
            400,
            "invalid_request_error",
        ),
        # Message missing role
        (
            {"model": "gpt-4o", "messages": [{"content": "Hello"}]},
            400,
            "invalid_request_error",
        ),
        # Message with unsupported role
        (
            {"model": "gpt-4o", "messages": [{"role": "developer", "content": "Hello"}]},
            400,
            "invalid_request_error",
        ),
        # Message with content=None (should be allowed for assistant with tool_calls)
        (
            {"model": "gpt-4o", "messages": [{"role": "user", "content": None}]},
            400,
            "invalid_request_error",
        ),
        # Message with empty string content (should be accepted)
        # This test expects success - will be handled differently
        # Message with content=[] (empty array)
        (
            {"model": "gpt-4o", "messages": [{"role": "user", "content": []}]},
            400,
            "invalid_request_error",
        ),
        # Message object not a dict (it's a string)
        (
            {"model": "gpt-4o", "messages": ["not a dict"]},
            400,
            "invalid_request_error",
        ),
        # max_tokens is zero
        (
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 0},
            400,
            "invalid_request_error",
        ),
        # max_tokens is negative
        (
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": -1},
            400,
            "invalid_request_error",
        ),
        # max_tokens is not an integer
        (
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": "invalid"},
            422,
            "validation_error",
        ),
        # temperature outside expected range (too high)
        (
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "temperature": 3.0},
            400,
            "invalid_request_error",
        ),
        # temperature outside expected range (negative)
        (
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "temperature": -1.0},
            400,
            "invalid_request_error",
        ),
        # stream_options provided while stream=False
        (
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "stream_options": {"include_usage": True},
            },
            400,
            "invalid_request_error",
        ),
        # tools wrong shape (not a list)
        (
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": "not a list",
            },
            422,
            "validation_error",
        ),
        # tools list with invalid tool shape
        (
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"invalid": "tool"}],
            },
            400,
            "invalid_request_error",
        ),
    ],
)
def test_chat_validation_errors(request_body, expected_status, error_type):
    """Test that malformed requests are properly rejected with appropriate error codes."""
    response = client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == expected_status
    
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == error_type
    assert "message" in data["error"]


def test_chat_validation_empty_string_content_accepted(mocker):
    """Test that empty string content is accepted (some clients send this)."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": ""}]},
    )
    # Should be accepted - empty string is valid content
    assert response.status_code == 200


def test_chat_validation_unknown_top_level_fields_ignored(mocker):
    """Test that unknown top-level fields are ignored rather than rejected."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "unknown_field": "should be ignored",
            "another_unknown": 123,
        },
    )
    # Should succeed - unknown fields are ignored
    assert response.status_code == 200


def test_chat_validation_content_part_missing_type(mocker):
    """Test that content part without type field is rejected."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],  # Missing 'type' field
        },
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_chat_validation_content_part_unsupported_type(mocker):
    """Test that content part with unsupported type is rejected."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": [{"type": "audio", "data": "base64data"}]}
            ],
        },
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_chat_validation_temperature_boundary_values(mocker):
    """Test that valid boundary temperature values are accepted."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    # Test temperature=0.0 (minimum valid)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0.0},
    )
    assert response.status_code == 200
    
    # Test temperature=2.0 (maximum valid for most APIs)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "temperature": 2.0},
    )
    assert response.status_code == 200


def test_chat_validation_max_tokens_boundary_values(mocker):
    """Test that valid boundary max_tokens values are accepted."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    # Test max_tokens=1 (minimum valid)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
    )
    assert response.status_code == 200
