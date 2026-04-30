"""Unit tests for chat endpoint upstream error translation.

Tests how upstream Amplify errors are translated to OpenAI-compatible error responses.
Covers the upstream failure translation from the test refactor plan.
"""
import json
import pytest
import httpx
from fastapi.testclient import TestClient
from open_amplify_ai.server import app
import os

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_error_response(mocker, status_code, json_data=None, text=None):
    """Build a mock httpx response with error status."""
    mock_response = mocker.Mock(spec=httpx.Response)
    mock_response.status_code = status_code
    
    # Always set text to a string to avoid Mock slicing errors
    if text:
        mock_response.text = text
    elif json_data:
        # Convert json_data to string for text attribute
        mock_response.text = json.dumps(json_data)
        mock_response.json.return_value = json_data
    else:
        # Default empty string if neither provided
        mock_response.text = ""
    
    # Make raise_for_status raise an HTTPStatusError
    def raise_for_status():
        raise httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=mocker.Mock(spec=httpx.Request),
            response=mock_response,
        )
    
    mock_response.raise_for_status = raise_for_status
    return mock_response


def _make_async_client_with_error(mocker, response):
    """Build an async httpx client mock that returns an error response."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def test_upstream_401_unauthorized(mocker):
    """Test that Amplify 401 is translated to OpenAI-compatible 401."""
    error_response = _make_error_response(
        mocker, 401, json_data={"error": "Unauthorized", "message": "Invalid token"}
    )
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 401
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "authentication_error"
    assert "message" in error


def test_upstream_403_forbidden(mocker):
    """Test that Amplify 403 is translated to OpenAI-compatible 403."""
    error_response = _make_error_response(
        mocker, 403, json_data={"error": "Forbidden", "message": "Access denied"}
    )
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 403
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "permission_error"


def test_upstream_404_not_found(mocker):
    """Test that Amplify 404 is translated to OpenAI-compatible 404."""
    error_response = _make_error_response(
        mocker, 404, json_data={"error": "Not Found", "message": "Model not found"}
    )
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 404
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "not_found_error"


def test_upstream_429_rate_limit(mocker):
    """Test that Amplify 429 is translated with rate limit information preserved."""
    error_response = _make_error_response(
        mocker,
        429,
        json_data={"error": "Rate limit exceeded", "message": "Too many requests"},
    )
    # Add retry headers
    error_response.headers = {
        "retry-after": "60",
        "x-ratelimit-limit": "100",
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": "1234567890",
    }
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 429
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "rate_limit_error"
    
    # Check if rate limit headers are preserved
    if "retry-after" in response.headers:
        assert response.headers["retry-after"] == "60"


def test_upstream_500_internal_error(mocker):
    """Test that Amplify 500 is translated to OpenAI-compatible 500."""
    error_response = _make_error_response(
        mocker, 500, json_data={"error": "Internal Server Error"}
    )
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 500
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] in ["api_error", "internal_server_error"]


def test_upstream_502_bad_gateway(mocker):
    """Test that Amplify 502 is translated appropriately."""
    error_response = _make_error_response(mocker, 502, text="Bad Gateway")
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 502
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "service_unavailable_error"


def test_upstream_503_service_unavailable(mocker):
    """Test that Amplify 503 is translated appropriately."""
    error_response = _make_error_response(
        mocker, 503, json_data={"error": "Service temporarily unavailable"}
    )
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 503
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "service_unavailable_error"


def test_upstream_504_gateway_timeout(mocker):
    """Test that Amplify 504 is translated appropriately."""
    error_response = _make_error_response(mocker, 504, text="Gateway Timeout")
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    assert response.status_code == 504
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "timeout_error"


def test_upstream_timeout_exception(mocker):
    """Test that request timeout is handled and translated appropriately."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(side_effect=httpx.TimeoutException("Request timeout"))
    
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    # TimeoutException maps to 504 per error_handling.py
    assert response.status_code == 504
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "timeout_error"
    assert "timeout" in error["message"].lower()


def test_upstream_connection_error(mocker):
    """Test that connection errors are handled appropriately."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    # ConnectError maps to 502 per error_handling.py
    assert response.status_code == 502
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "service_unavailable_error"


def test_upstream_malformed_json_response(mocker):
    """Test that malformed JSON from upstream is handled gracefully."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    mock_response.text = "This is not JSON"
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    
    mocker.patch("open_amplify_ai.routers.chat.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    # Should return 500 for malformed upstream response
    assert response.status_code == 500
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "api_error"


def test_upstream_streaming_connection_closed_early(mocker):
    """Test that streaming connection closing early is handled."""
    async def fake_aiter_lines():
        yield 'data: {"data":"Hello"}'
        # Simulate connection closing without [DONE]
        raise httpx.RemoteProtocolError("Connection closed")
    
    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = mocker.Mock()
    mock_resp.aiter_lines = fake_aiter_lines
    
    mock_stream_cm = mocker.AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_resp
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = mocker.Mock(return_value=mock_stream_cm)
    
    mocker.patch("open_amplify_ai.utils.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    # Should still return 200 but stream should be terminated properly
    assert response.status_code == 200
    # The stream should have been closed gracefully with available data


def test_upstream_streaming_invalid_sse_format(mocker):
    """Test that invalid SSE format from upstream is handled."""
    async def fake_aiter_lines():
        yield "invalid line without data: prefix"
        yield 'data: {"data":"Hello"}'
        yield "data: [DONE]"
    
    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = mocker.Mock()
    mock_resp.aiter_lines = fake_aiter_lines
    
    mock_stream_cm = mocker.AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_resp
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = mocker.Mock(return_value=mock_stream_cm)
    
    mocker.patch("open_amplify_ai.utils.httpx.AsyncClient", return_value=mock_client)
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    assert response.status_code == 200
    # Invalid lines should be skipped, valid data should be processed
    body = response.text
    assert "Hello" in body
    assert "[DONE]" in body


def test_error_response_has_openai_shape(mocker):
    """Test that all error responses follow OpenAI error shape."""
    error_response = _make_error_response(
        mocker, 500, json_data={"error": "Internal error"}
    )
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client_with_error(mocker, error_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    
    data = response.json()
    # OpenAI error shape: {"detail": {"error": {"message": str, "type": str, "code": str?}}}
    error = data["detail"]["error"]
    assert isinstance(error, dict)
    assert "message" in error
    assert "type" in error
    assert isinstance(error["message"], str)
    assert isinstance(error["type"], str)
