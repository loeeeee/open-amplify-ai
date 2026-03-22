"""Comprehensive integration tests simulating cline/kilo/openclaw usage patterns.

These tests exercise the full FastAPI request/response cycle with mocked
Amplify upstream calls. No live AMPLIFY_AI_TOKEN is needed.

To run:
    nix-shell --run "uv run pytest tests/integration/mocked/test_assistants.py -v"
"""
import io
import json
import os
import pytest
from typing import Any, Dict, List
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_json_response(mocker: Any, json_data: Any) -> Any:
    """Build a generic sync mock response with a json() method."""
    mock = mocker.Mock()
    mock.raise_for_status = mocker.Mock()
    mock.json.return_value = json_data
    return mock


def _make_async_client(mocker: Any, response: Any) -> Any:
    """Build an async httpx client mock returning the same response for all methods."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = mocker.AsyncMock(return_value=response)
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


# ===========================================================================
# ASSISTANTS - openclaw pattern
# ===========================================================================


def test_client_creates_assistant(mocker: Any) -> None:
    """Openclaw creates an assistant via Amplify POST /assistant/create."""
    captured: Dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> Any:
        captured["payload"] = kwargs.get("json", {})
        return _make_json_response(mocker, {
            "success": True,
            "data": {"assistantId": "astp/new123", "id": "ast/new456"},
        })

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch("open_amplify_ai.routers.assistants.httpx.AsyncClient", return_value=mock_client)

    response = client.post("/v1/assistants", json={
        "model": "gpt-4o",
        "name": "My Test Assistant",
        "instructions": "Be concise and helpful.",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "astp/new123"
    assert data["object"] == "assistant"
    assert data["name"] == "My Test Assistant"
    assert data["instructions"] == "Be concise and helpful."
    assert data["model"] == "gpt-4o"

    assert captured["payload"]["data"]["name"] == "My Test Assistant"
    assert captured["payload"]["data"]["instructions"] == "Be concise and helpful."


def test_client_lists_assistants(mocker: Any) -> None:
    """Openclaw lists all assistants."""
    resp = _make_json_response(mocker, {
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
                "instructions": "Be brief",
                "createdAt": "2024-01-02T00:00:00",
                "dataSources": [],
            },
        ],
    })
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.get("/v1/assistants")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == "astp/abc123"
    assert data["data"][0]["object"] == "assistant"
    assert data["has_more"] is False
    assert data["first_id"] == "astp/abc123"
    assert data["last_id"] == "astp/xyz789"


def test_client_lists_assistants_empty(mocker: Any) -> None:
    """Openclaw gets empty assistant list."""
    resp = _make_json_response(mocker, {"success": True, "data": []})
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.get("/v1/assistants")
    assert response.status_code == 200

    data = response.json()
    assert data["data"] == []
    assert data["first_id"] is None
    assert data["last_id"] is None


def test_client_retrieves_assistant(mocker: Any) -> None:
    """Openclaw retrieves a single assistant by ID (filters from full list)."""
    resp = _make_json_response(mocker, {
        "success": True,
        "data": [
            {
                "assistantId": "astp/abc123",
                "name": "Test Assistant",
                "instructions": "Be helpful",
                "createdAt": "2024-01-01T00:00:00",
                "dataSources": [],
            },
        ],
    })
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.get("/v1/assistants/astp/abc123")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["object"] == "assistant"
    assert data["name"] == "Test Assistant"


def test_client_retrieves_assistant_not_found(mocker: Any) -> None:
    """Openclaw gets 404 when assistant is not found."""
    resp = _make_json_response(mocker, {"success": True, "data": []})
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.get("/v1/assistants/astp/nonexistent")
    assert response.status_code == 404


def test_client_modifies_assistant(mocker: Any) -> None:
    """Openclaw modifies an existing assistant via upsert with assistantId."""
    captured: Dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> Any:
        captured["payload"] = kwargs.get("json", {})
        return _make_json_response(mocker, {
            "success": True,
            "data": {"assistantId": "astp/abc123"},
        })

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch("open_amplify_ai.routers.assistants.httpx.AsyncClient", return_value=mock_client)

    response = client.post("/v1/assistants/astp/abc123", json={
        "name": "Updated Name",
        "instructions": "New instructions",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["name"] == "Updated Name"
    assert captured["payload"]["data"]["assistantId"] == "astp/abc123"


def test_client_deletes_assistant(mocker: Any) -> None:
    """Openclaw deletes an assistant."""
    resp = _make_json_response(mocker, {"success": True, "message": "Deleted"})
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.delete("/v1/assistants/astp/abc123")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["object"] == "assistant.deleted"
    assert data["deleted"] is True


def test_client_deletes_assistant_unauthorized(mocker: Any) -> None:
    """Openclaw gets deleted=False when Amplify denies deletion."""
    resp = _make_json_response(mocker, {"success": False, "message": "Not authorized"})
    mocker.patch(
        "open_amplify_ai.routers.assistants.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.delete("/v1/assistants/astp/other123")
    assert response.status_code == 200
    assert response.json()["deleted"] is False
