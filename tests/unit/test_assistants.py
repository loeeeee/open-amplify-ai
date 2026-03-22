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


def test_list_assistants_success(mocker):
    """GET /v1/assistants returns OpenAI assistant list from Amplify /assistant/list."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": [
            {
                "assistantId": "astp/abc123",
                "name": "Test Assistant",
                "instructions": "Be helpful",
                "createdAt": "2024-01-01T00:00:00",
                "dataSources": [],
            }
        ],
    }
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/assistants")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "astp/abc123"
    assert data["data"][0]["object"] == "assistant"
    assert data["has_more"] is False
    assert data["first_id"] == "astp/abc123"
    assert data["last_id"] == "astp/abc123"


def test_list_assistants_empty(mocker):
    """GET /v1/assistants returns an empty list when no assistants exist."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": []}
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/assistants")
    assert response.status_code == 200
    data = response.json()
    assert data["data"] == []
    assert data["first_id"] is None
    assert data["last_id"] is None


def test_create_assistant_success(mocker):
    """POST /v1/assistants returns an OpenAI assistant object from Amplify /assistant/create."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "message": "Assistant created successfully",
        "data": {
            "assistantId": "astp/new123",
            "id": "ast/new456",
            "version": 1,
        },
    }
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    req_body = {
        "model": "gpt-4o",
        "name": "My Assistant",
        "instructions": "Be concise.",
    }
    response = client.post("/v1/assistants", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/new123"
    assert data["object"] == "assistant"
    assert data["name"] == "My Assistant"
    assert data["instructions"] == "Be concise."


def test_retrieve_assistant_success(mocker):
    """GET /v1/assistants/{id} finds the matching assistant from the full list."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": [
            {
                "assistantId": "astp/abc123",
                "name": "Test Assistant",
                "instructions": "Be helpful",
                "createdAt": "2024-01-01T00:00:00",
                "dataSources": [],
            },
            {
                "assistantId": "astp/xyz789",
                "name": "Other Assistant",
                "instructions": "Other",
                "createdAt": "2024-01-02T00:00:00",
                "dataSources": [],
            },
        ],
    }
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/assistants/astp/abc123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["name"] == "Test Assistant"
    assert data["object"] == "assistant"


def test_retrieve_assistant_not_found(mocker):
    """GET /v1/assistants/{id} returns 404 when assistant is absent."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": []}
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.get("/v1/assistants/astp/nonexistent")
    assert response.status_code == 404


def test_modify_assistant_success(mocker):
    """POST /v1/assistants/{id} sends an upsert to Amplify with the assistantId."""
    captured = {}

    async def fake_post(url, **kwargs):
        captured["payload"] = kwargs.get("json", {})
        m = mocker.Mock()
        m.raise_for_status = mocker.Mock()
        m.json.return_value = {"success": True, "data": {"assistantId": "astp/abc123"}}
        return m

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient", return_value=mock_client
    )

    req_body = {"name": "Updated Name", "instructions": "New instructions"}
    response = client.post("/v1/assistants/astp/abc123", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["name"] == "Updated Name"
    assert captured["payload"]["data"]["assistantId"] == "astp/abc123"


def test_delete_assistant_success(mocker):
    """DELETE /v1/assistants/{id} returns OpenAI DeletedObject from Amplify /assistant/delete."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "message": "Assistant deleted successfully.",
    }
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.delete("/v1/assistants/astp/abc123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["object"] == "assistant.deleted"
    assert data["deleted"] is True


def test_delete_assistant_amplify_failure(mocker):
    """DELETE /v1/assistants/{id} returns deleted=False when Amplify reports failure."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": False,
        "message": "You are not authorized to delete this assistant.",
    }
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.delete("/v1/assistants/astp/other123")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is False
