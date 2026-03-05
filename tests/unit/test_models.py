import io
import pytest
import os
import requests
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# Set up dummy environment variable for tests to bypass token validation failure
os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)



def test_get_models_success(mocker):
    """GET /v1/models returns OpenAI list format from Amplify /available_models."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o", "provider": "Azure"}
            ]
        },
    }
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "gpt-4o"
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "amplify-ai"
    assert "created" in data["data"][0]


def test_get_models_amplify_failure(mocker):
    """GET /v1/models returns 500 when Amplify signals failure."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": False, "error": "Some error"}
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to fetch models from Amplify AI"


def test_get_model_by_id_success(mocker):
    """GET /v1/models/{model} returns a single model when found."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "claude-3", "name": "Claude 3"},
            ]
        },
    }
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models/gpt-4o")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "gpt-4o"
    assert data["object"] == "model"
    assert "created" in data
    assert data["owned_by"] == "amplify-ai"


def test_get_model_by_id_not_found(mocker):
    """GET /v1/models/{model} returns 404 when model is absent."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": {"models": [{"id": "gpt-4o"}]},
    }
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models/nonexistent-model")
    assert response.status_code == 404


def test_delete_model_not_allowed():
    """DELETE /v1/models/{model} always returns 405."""
    response = client.delete("/v1/models/gpt-4o")
    assert response.status_code == 405

