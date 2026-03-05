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

def _make_amplify_files_query_response(items: List[Dict[str, Any]]) -> Any:
    """Build a mock response for Amplify POST /files/query."""
    mock = type("MockResponse", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: {
            "success": True,
            "data": {"items": items, "pageKey": None},
        },
    })()
    return mock


# ===========================================================================
# VECTOR STORES
# ===========================================================================


def test_client_creates_vector_store(mocker: Any) -> None:
    """Client creates a vector store backed by an Amplify tag."""
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.requests.post",
        return_value=type("MockResponse", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True, "message": "Tags added"},
        })(),
    )

    response = client.post("/v1/vector_stores", json={"name": "my-store"})
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store"
    assert data["status"] == "completed"
    assert data["file_counts"]["total"] == 0


def test_client_retrieves_vector_store(mocker: Any) -> None:
    """Client retrieves a vector store with file counts."""
    tags_mock = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {
            "success": True,
            "data": {"tags": ["my-store", "other-tag"]},
        },
    })()

    files_mock = _make_amplify_files_query_response([
        {"id": "file1", "totalTokens": 100},
        {"id": "file2", "totalTokens": 200},
    ])

    mocker.patch("open_amplify_ai.routers.vector_stores.requests.get", return_value=tags_mock)
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=files_mock)

    response = client.get("/v1/vector_stores/my-store")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store"
    assert data["file_counts"]["total"] == 2
    assert data["file_counts"]["completed"] == 2


def test_client_retrieves_vector_store_not_found(mocker: Any) -> None:
    """Client gets 404 when the backing tag does not exist."""
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.requests.get",
        return_value=type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {
                "success": True,
                "data": {"tags": ["other-tag"]},
            },
        })(),
    )

    response = client.get("/v1/vector_stores/nonexistent-store")
    assert response.status_code == 404


def test_client_deletes_vector_store(mocker: Any) -> None:
    """Client deletes a vector store (removes backing Amplify tag)."""
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.requests.post",
        return_value=type("MockResponse", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True, "message": "Tag deleted"},
        })(),
    )

    response = client.delete("/v1/vector_stores/my-store")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store.deleted"
    assert data["deleted"] is True


def test_client_lists_vector_store_files(mocker: Any) -> None:
    """Client lists all files in a vector store."""
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_files_query_response([
            {"id": "file-a", "totalTokens": 100},
            {"id": "file-b", "totalTokens": 200},
        ]),
    )

    response = client.get("/v1/vector_stores/my-store/files")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "vector_store.file"
    assert data["data"][0]["vector_store_id"] == "my-store"
    assert data["data"][0]["status"] == "completed"


def test_client_adds_file_to_vector_store(mocker: Any) -> None:
    """Client adds a file to a vector store by tagging it."""
    captured: Dict[str, Any] = {}

    def _capture_post(url: str, headers: Any, json: Any, timeout: int) -> Any:
        """Intercept Amplify POST to verify tag payload."""
        captured["payload"] = json
        return type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True},
        })()

    mocker.patch(
        "open_amplify_ai.routers.vector_stores.requests.post",
        side_effect=_capture_post,
    )

    response = client.post("/v1/vector_stores/my-store/files", json={
        "file_id": "user@vu.edu/2024-01-01/doc.json",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "user@vu.edu/2024-01-01/doc.json"
    assert data["object"] == "vector_store.file"
    assert data["vector_store_id"] == "my-store"
    assert data["status"] == "completed"

    # Verify the tag was set correctly
    assert captured["payload"]["data"]["tags"] == ["my-store"]


def test_client_vector_store_modify_not_implemented() -> None:
    """Modifying a vector store returns 501 (Amplify has no tag rename)."""
    response = client.post("/v1/vector_stores/my-store", json={"name": "renamed"})
    assert response.status_code == 501
