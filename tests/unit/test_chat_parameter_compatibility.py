"""Unit tests for chat endpoint parameter compatibility.

Tests documented behavior for each supported OpenAI parameter.
Covers the broader parameter compatibility from the test refactor plan.
"""
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


def test_top_p_parameter_handling(mocker):
    """Test that top_p parameter is rejected (not supported by Amplify)."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "top_p": 0.9,
        },
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    error = data["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "top_p" in error["message"].lower()


def test_n_parameter_handling(mocker):
    """Test that n parameter (number of completions) is handled."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "Response"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 2,
        },
    )
    
    # Should accept n parameter (may not implement multiple completions)
    assert response.status_code in [200, 400]


def test_stop_parameter_handling(mocker):
    """Test that stop parameter is rejected (not supported by Amplify)."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["\n", "END"],
        },
    )
    
    assert response.status_code == 400
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "stop" in error["message"].lower()


def test_presence_penalty_parameter_handling(mocker):
    """Test that presence_penalty parameter is rejected (not supported by Amplify)."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "presence_penalty": 0.5,
        },
    )
    
    assert response.status_code == 400
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "presence_penalty" in error["message"].lower()


def test_frequency_penalty_parameter_handling(mocker):
    """Test that frequency_penalty parameter is rejected (not supported by Amplify)."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "frequency_penalty": 0.3,
        },
    )
    
    assert response.status_code == 400
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "frequency_penalty" in error["message"].lower()


def test_user_parameter_handling(mocker):
    """Test that user parameter is accepted."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "user-123",
        },
    )
    
    assert response.status_code == 200


def test_seed_parameter_handling(mocker):
    """Test that seed parameter is rejected (determinism not supported by Amplify)."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "seed": 42,
        },
    )
    
    assert response.status_code == 400
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "seed" in error["message"].lower()


def test_response_format_parameter_handling(mocker):
    """Test that response_format parameter is rejected (structured output not fully supported by Amplify)."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
        },
    )
    
    assert response.status_code == 400
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "response_format" in error["message"].lower()


def test_tool_choice_parameter_handling(mocker):
    """Test that tool_choice parameter is handled."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": '{"tool":"read_file","parameters":{"path":"/tmp/test.txt"}}',
    }
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Read file"}],
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
            "tool_choice": "auto",
        },
    )
    
    assert response.status_code == 200


def test_parallel_tool_calls_parameter_handling(mocker):
    """Test that parallel_tool_calls parameter is handled."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            "parallel_tool_calls": True,
        },
    )
    
    assert response.status_code == 200


def test_logit_bias_parameter_handling(mocker):
    """Test that logit_bias parameter is handled."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "logit_bias": {"50256": -100},
        },
    )
    
    # May accept or reject logit_bias
    assert response.status_code in [200, 400]


def test_multiple_parameters_together(mocker):
    """Test that rejected parameters cause failure even when mixed with valid ones."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "user": "test-user",
        },
    )
    
    # Should reject due to unsupported parameters
    assert response.status_code == 400
    data = response.json()
    error = data["detail"]["error"]
    assert error["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "param_name,param_value,should_accept",
    [
        ("temperature", 0.7, True),
        ("max_tokens", 100, True),
        ("top_p", 0.9, False),  # Rejected by Amplify
        ("presence_penalty", 0.5, False),  # Rejected by Amplify
        ("frequency_penalty", 0.5, False),  # Rejected by Amplify
        ("user", "user-123", True),
        ("seed", 42, False),  # Rejected by Amplify
        ("stop", ["END"], False),  # Rejected by Amplify
    ],
)
def test_parameter_acceptance_matrix(mocker, param_name, param_value, should_accept):
    """Test parameter acceptance using a matrix of values."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": "ok"}
    
    mocker.patch(
        "open_amplify_ai.routers.chat.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )
    
    request_json = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        param_name: param_value,
    }
    
    response = client.post("/v1/chat/completions", json=request_json)
    
    if should_accept:
        assert response.status_code == 200
    else:
        assert response.status_code == 400
        data = response.json()
        error = data["detail"]["error"]
        assert error["type"] == "invalid_request_error"
