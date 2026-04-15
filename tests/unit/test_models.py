import io
import os
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_async_client(mocker, response):
    """Build an async httpx client mock that returns the given response."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = mocker.AsyncMock(return_value=response)
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_amplify_model_full():
    """Return a realistic Amplify model dict with all metadata fields."""
    return {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "description": "A capable multimodal model.",
        "provider": "Azure",
        "inputContextWindow": 128000,
        "outputTokenLimit": 16384,
        "inputTokenCost": 2.50,
        "outputTokenCost": 10.00,
        "cachedInputTokenCost": 1.25,
        "cachedOutputTokenCost": None,
        "supportsImages": True,
        "supportsSystemPrompts": True,
        "systemPrompt": "You are a helpful assistant.",
    }


def test_get_models_success(mocker):
    """GET /v1/models returns OpenAI list format from Amplify /available_models."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o", "provider": "Azure"}
            ]
        },
    }
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

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
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": False, "error": "Some error"}
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/models")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to fetch models from Amplify AI"


def test_get_model_by_id_success(mocker):
    """GET /v1/models/{model} returns a single model when found."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "claude-3", "name": "Claude 3"},
            ]
        },
    }
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

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
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {"models": [{"id": "gpt-4o"}]},
    }
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/models/nonexistent-model")
    assert response.status_code == 404


def test_delete_model_not_allowed():
    """DELETE /v1/models/{model} always returns 405."""
    response = client.delete("/v1/models/gpt-4o")
    assert response.status_code == 405


def test_get_models_includes_kilo_cost_and_limit(mocker):
    """GET /v1/models returns cost and limit fields for Kilo consumption."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [_make_amplify_model_full()]
        },
    }
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/models")
    assert response.status_code == 200
    model = response.json()["data"][0]

    # Kilo-consumable cost fields
    assert "cost" in model
    assert model["cost"]["input"] == 2.50
    assert model["cost"]["output"] == 10.00
    assert model["cost"]["cache_read"] == 1.25
    assert "cache_write" not in model["cost"]  # None values omitted

    # Kilo-consumable limit fields
    assert "limit" in model
    assert model["limit"]["context"] == 128000
    assert model["limit"]["output"] == 16384

    # Capabilities
    assert "capabilities" in model
    assert model["capabilities"]["images"] is True
    assert model["capabilities"]["system_prompt"] is True
    assert model["capabilities"]["description"] == "A capable multimodal model."

    # Display name
    assert model["display_name"] == "GPT-4o"

    # Legacy fields still present
    assert model["context_length"] == 128000
    assert model["max_output_tokens"] == 16384
    assert model["max_model_len"] == 128000 + 16384


def test_get_model_by_id_includes_kilo_fields(mocker):
    """GET /v1/models/{model} includes cost and limit for a single model."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [_make_amplify_model_full()]
        },
    }
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/models/gpt-4o")
    assert response.status_code == 200
    model = response.json()
    assert model["cost"]["input"] == 2.50
    assert model["limit"]["context"] == 128000
    assert model["limit"]["output"] == 16384


def test_get_models_no_pricing_omits_cost(mocker):
    """Models without pricing data omit the cost field entirely."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {
                    "id": "basic-model",
                    "name": "Basic",
                    "inputContextWindow": 4096,
                    "outputTokenLimit": 1024,
                }
            ]
        },
    }
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/models")
    assert response.status_code == 200
    model = response.json()["data"][0]
    assert "cost" not in model  # No pricing -> no cost key
    assert model["limit"]["context"] == 4096
    assert model["limit"]["output"] == 1024


def test_get_models_filters_alias_entries(mocker):
    """Alias models (default, advanced, cheapest, documentCaching) are filtered out."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "default", "name": "Default"},
                {"id": "advanced", "name": "Advanced"},
                {"id": "cheapest", "name": "Cheapest"},
                {"id": "documentCaching", "name": "Doc Cache"},
            ]
        },
    }
    mocker.patch(
        "open_amplify_ai.routers.models.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    ids = [m["id"] for m in data["data"]]
    assert ids == ["gpt-4o"]
    assert "default" not in ids
    assert "advanced" not in ids
    assert "cheapest" not in ids
    assert "documentCaching" not in ids
