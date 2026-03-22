"""Comprehensive integration tests simulating cline/kilo/openclaw usage patterns.

These tests exercise the full FastAPI request/response cycle with mocked
Amplify upstream calls. No live AMPLIFY_AI_TOKEN is needed.

To run:
    nix-shell --run "uv run pytest tests/integration/mocked/test_vector_stores.py -v"
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


def _make_files_query_client(mocker: Any, items: List[Dict[str, Any]]) -> Any:
    """Build an async client returning a /files/query response (targets utils module)."""
    resp = _make_json_response(mocker, {
        "success": True,
        "data": {"items": items, "pageKey": None},
    })
    return _make_async_client(mocker, resp)


# ===========================================================================
# VECTOR STORES
# ===========================================================================


def test_client_creates_vector_store(mocker: Any) -> None:
    """Client creates a vector store backed by an Amplify tag."""
    resp = _make_json_response(mocker, {"success": True, "message": "Tags added"})
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
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
    tags_resp = _make_json_response(mocker, {
        "success": True,
        "data": {"tags": ["my-store", "other-tag"]},
    })
    router_client = mocker.AsyncMock()
    router_client.__aenter__.return_value = router_client
    router_client.get = mocker.AsyncMock(return_value=tags_resp)

    files_client = _make_files_query_client(mocker, [
        {"id": "file1", "totalTokens": 100},
        {"id": "file2", "totalTokens": 200},
    ])

    # Both vector_stores.py and utils.py import the same httpx module; use side_effect
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        side_effect=[router_client, files_client],
    )

    response = client.get("/v1/vector_stores/my-store")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store"
    assert data["file_counts"]["total"] == 2
    assert data["file_counts"]["completed"] == 2


def test_client_retrieves_vector_store_not_found(mocker: Any) -> None:
    """Client gets 404 when the backing tag does not exist."""
    tags_resp = _make_json_response(mocker, {
        "success": True,
        "data": {"tags": ["other-tag"]},
    })
    router_client = mocker.AsyncMock()
    router_client.__aenter__.return_value = router_client
    router_client.get = mocker.AsyncMock(return_value=tags_resp)
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        return_value=router_client,
    )

    response = client.get("/v1/vector_stores/nonexistent-store")
    assert response.status_code == 404


def test_client_deletes_vector_store(mocker: Any) -> None:
    """Client deletes a vector store (removes backing Amplify tag)."""
    resp = _make_json_response(mocker, {"success": True, "message": "Tag deleted"})
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
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
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_files_query_client(mocker, [
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

    async def fake_post(url: str, **kwargs: Any) -> Any:
        captured["payload"] = kwargs.get("json", {})
        return _make_json_response(mocker, {"success": True})

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient", return_value=mock_client
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
    assert captured["payload"]["data"]["tags"] == ["my-store"]


def test_client_vector_store_modify_not_implemented() -> None:
    """Modifying a vector store returns 501 (Amplify has no tag rename)."""
    response = client.post("/v1/vector_stores/my-store", json={"name": "renamed"})
    assert response.status_code == 501
