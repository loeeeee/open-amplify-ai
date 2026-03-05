"""Comprehensive integration tests simulating cline/kilo/openclaw usage patterns.

These tests exercise the full FastAPI request/response cycle with mocked
Amplify upstream calls. No live AMPLIFY_AI_TOKEN is needed.

Covers all endpoint groups:
  1. Model discovery and retrieval
  2. Chat completions (simple, streaming, tool calls, multi-turn)
  3. File operations (upload, list, retrieve, delete, content download)
  4. Assistant lifecycle (create, list, retrieve, modify, delete)
  5. Thread deletion
  6. Vector store lifecycle (create, retrieve, delete, files)
  7. Edge cases and error handling
  8. Unsupported endpoint stubs (501)

To run:
    nix-shell --run "uv run pytest src/open_amplify_ai/test_chat_client_integration.py -v"
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




# ===========================================================================
# ASSISTANTS - openclaw pattern
# ===========================================================================


def test_client_creates_assistant(mocker: Any) -> None:
    """Openclaw creates an assistant via Amplify POST /assistant/create."""
    captured: Dict[str, Any] = {}

    def _capture_post(url: str, headers: Any, json: Any, timeout: int) -> Any:
        """Intercept Amplify POST to verify payload mapping."""
        captured["payload"] = json
        return type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {
                "success": True,
                "data": {"assistantId": "astp/new123", "id": "ast/new456"},
            },
        })()

    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.post",
        side_effect=_capture_post,
    )

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

    # Verify payload was correctly mapped
    assert captured["payload"]["data"]["name"] == "My Test Assistant"
    assert captured["payload"]["data"]["instructions"] == "Be concise and helpful."


def test_client_lists_assistants(mocker: Any) -> None:
    """Openclaw lists all assistants."""
    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.get",
        return_value=type("MockResponse", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {
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
            },
        })(),
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
    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.get",
        return_value=type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True, "data": []},
        })(),
    )

    response = client.get("/v1/assistants")
    assert response.status_code == 200

    data = response.json()
    assert data["data"] == []
    assert data["first_id"] is None
    assert data["last_id"] is None


def test_client_retrieves_assistant(mocker: Any) -> None:
    """Openclaw retrieves a single assistant by ID (filters from full list)."""
    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.get",
        return_value=type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {
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
            },
        })(),
    )

    response = client.get("/v1/assistants/astp/abc123")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["object"] == "assistant"
    assert data["name"] == "Test Assistant"


def test_client_retrieves_assistant_not_found(mocker: Any) -> None:
    """Openclaw gets 404 when assistant is not found."""
    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.get",
        return_value=type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True, "data": []},
        })(),
    )

    response = client.get("/v1/assistants/astp/nonexistent")
    assert response.status_code == 404


def test_client_modifies_assistant(mocker: Any) -> None:
    """Openclaw modifies an existing assistant via upsert with assistantId."""
    captured: Dict[str, Any] = {}

    def _capture_post(url: str, headers: Any, json: Any, timeout: int) -> Any:
        """Intercept Amplify POST to verify assistantId is set."""
        captured["payload"] = json
        return type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True, "data": {"assistantId": "astp/abc123"}},
        })()

    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.post",
        side_effect=_capture_post,
    )

    response = client.post("/v1/assistants/astp/abc123", json={
        "name": "Updated Name",
        "instructions": "New instructions",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["name"] == "Updated Name"

    # Verify assistantId was included for upsert
    assert captured["payload"]["data"]["assistantId"] == "astp/abc123"


def test_client_deletes_assistant(mocker: Any) -> None:
    """Openclaw deletes an assistant."""
    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.post",
        return_value=type("MockResponse", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True, "message": "Deleted"},
        })(),
    )

    response = client.delete("/v1/assistants/astp/abc123")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["object"] == "assistant.deleted"
    assert data["deleted"] is True


def test_client_deletes_assistant_unauthorized(mocker: Any) -> None:
    """Openclaw gets deleted=False when Amplify denies deletion."""
    mocker.patch(
        "open_amplify_ai.routers.assistants.requests.post",
        return_value=type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": False, "message": "Not authorized"},
        })(),
    )

    response = client.delete("/v1/assistants/astp/other123")
    assert response.status_code == 200
    assert response.json()["deleted"] is False
