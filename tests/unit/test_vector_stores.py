import io
import os
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_async_client(mocker, response):
    """Build an async httpx client mock that returns the given response for all methods."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = mocker.AsyncMock(return_value=response)
    mock_client.post = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_files_query_response(mocker, items, page_key=None):
    """Build a mock response for Amplify POST /files/query."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {"items": items, "pageKey": page_key},
    }
    return mock_response


def test_create_vector_store_success(mocker):
    """POST /v1/vector_stores creates a tag in Amplify and returns a synthetic VectorStore."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "message": "Tags added successfully"}
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.post("/v1/vector_stores", json={"name": "my-store"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store"
    assert data["status"] == "completed"
    assert "file_counts" in data
    assert data["file_counts"]["total"] == 0


def test_retrieve_vector_store_success(mocker):
    """GET /v1/vector_stores/{id} returns a VectorStore with file counts."""
    tags_resp = mocker.Mock()
    tags_resp.raise_for_status = mocker.Mock()
    tags_resp.json.return_value = {
        "success": True,
        "data": {"tags": ["my-store", "other-tag"]},
    }
    # First AsyncClient call: router GETs /files/tags/list
    router_client = mocker.AsyncMock()
    router_client.__aenter__.return_value = router_client
    router_client.get = mocker.AsyncMock(return_value=tags_resp)

    # Second AsyncClient call: query_amplify_files in utils POSTs /files/query
    files_items = [
        {"id": "file1", "totalTokens": 100},
        {"id": "file2", "totalTokens": 200},
    ]
    utils_client = _make_async_client(
        mocker, _make_files_query_response(mocker, files_items)
    )

    # Both vector_stores.py and utils.py import the same httpx module; use side_effect
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        side_effect=[router_client, utils_client],
    )

    response = client.get("/v1/vector_stores/my-store")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store"
    assert data["file_counts"]["total"] == 2
    assert data["file_counts"]["completed"] == 2


def test_retrieve_vector_store_not_found(mocker):
    """GET /v1/vector_stores/{id} returns 404 when the backing tag does not exist."""
    tags_resp = mocker.Mock()
    tags_resp.raise_for_status = mocker.Mock()
    tags_resp.json.return_value = {
        "success": True,
        "data": {"tags": ["other-tag"]},
    }
    router_client = mocker.AsyncMock()
    router_client.__aenter__.return_value = router_client
    router_client.get = mocker.AsyncMock(return_value=tags_resp)
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        return_value=router_client,
    )

    response = client.get("/v1/vector_stores/nonexistent-store")
    assert response.status_code == 404


def test_delete_vector_store_success(mocker):
    """DELETE /v1/vector_stores/{id} deletes the backing Amplify tag."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "message": "Tag deleted successfully"}
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.delete("/v1/vector_stores/my-store")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store.deleted"
    assert data["deleted"] is True


def test_list_vector_store_files_success(mocker):
    """GET /v1/vector_stores/{id}/files returns files tagged with the store ID."""
    files_items = [
        {"id": "file-a", "totalTokens": 50},
        {"id": "file-b", "totalTokens": 80},
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_async_client(
            mocker, _make_files_query_response(mocker, files_items)
        ),
    )

    response = client.get("/v1/vector_stores/my-store/files")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "vector_store.file"
    assert data["data"][0]["vector_store_id"] == "my-store"
    assert data["data"][0]["status"] == "completed"


def test_add_file_to_vector_store_success(mocker):
    """POST /v1/vector_stores/{id}/files associates a file with the store via set_tags."""
    captured = {}

    async def fake_post(url, **kwargs):
        captured["payload"] = kwargs.get("json", {})
        m = mocker.Mock()
        m.raise_for_status = mocker.Mock()
        m.json.return_value = {"success": True, "message": "Tags updated and added to user"}
        return m

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = fake_post
    mocker.patch(
        "open_amplify_ai.routers.vector_stores.httpx.AsyncClient", return_value=mock_client
    )

    req_body = {"file_id": "user@vu.edu/2024-01-01/abc.json"}
    response = client.post("/v1/vector_stores/my-store/files", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["object"] == "vector_store.file"
    assert data["vector_store_id"] == "my-store"
    assert captured["payload"]["data"]["tags"] == ["my-store"]


def test_modify_vector_store_not_implemented():
    """POST /v1/vector_stores/{id} (update) returns 501."""
    response = client.post("/v1/vector_stores/my-store", json={"name": "new-name"})
    assert response.status_code == 501


def test_delete_vector_store_file_not_implemented():
    """DELETE /v1/vector_stores/{id}/files/{file_id} returns 501."""
    response = client.delete("/v1/vector_stores/my-store/files/file-abc")
    assert response.status_code == 501
