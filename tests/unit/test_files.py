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
    mock_client.put = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_files_query_response(mocker, items, page_key=None):
    """Build a mock response for Amplify POST /files/query."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {"items": items, "pageKey": page_key},
    }
    return mock_response


def test_list_files_success(mocker):
    """GET /v1/files returns OpenAI file list mapped from Amplify /files/query."""
    items = [
        {
            "id": "user@vu.edu/2024-01-01/abc.json",
            "name": "test.pdf",
            "createdAt": "2024-01-01T00:00:00",
            "totalTokens": 100,
            "type": "application/pdf",
        }
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_async_client(mocker, _make_files_query_response(mocker, items)),
    )

    response = client.get("/v1/files")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "file"
    assert data["data"][0]["filename"] == "test.pdf"
    assert data["data"][0]["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["data"][0]["purpose"] == "assistants"


def test_list_files_empty(mocker):
    """GET /v1/files returns an empty list when no files exist."""
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_async_client(mocker, _make_files_query_response(mocker, [])),
    )

    response = client.get("/v1/files")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"] == []


def test_retrieve_file_success(mocker):
    """GET /v1/files/{file_id} returns a single file when found."""
    items = [
        {
            "id": "user@vu.edu/2024-01-01/abc.json",
            "name": "test.pdf",
            "createdAt": "2024-01-01T00:00:00",
            "totalTokens": 200,
            "type": "application/pdf",
        }
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_async_client(mocker, _make_files_query_response(mocker, items)),
    )

    response = client.get("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["filename"] == "test.pdf"


def test_retrieve_file_not_found(mocker):
    """GET /v1/files/{file_id} returns 404 when file is not in the list."""
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_async_client(mocker, _make_files_query_response(mocker, [])),
    )

    response = client.get("/v1/files/nonexistent-file-id")
    assert response.status_code == 404


def test_upload_file_success(mocker):
    """POST /v1/files uploads a file via two-step Amplify + S3 PUT and returns a File object."""
    init_resp = mocker.Mock()
    init_resp.raise_for_status = mocker.Mock()
    init_resp.json.return_value = {
        "success": True,
        "uploadUrl": "https://s3.example.com/upload?signed=true",
        "key": "user@vu.edu/2024-01-01/new-file.json",
        "statusUrl": "",
        "contentUrl": "",
        "metadataUrl": "",
    }

    s3_resp = mocker.Mock()
    s3_resp.raise_for_status = mocker.Mock()

    # upload_file uses two separate AsyncClient blocks: one for POST, one for PUT
    init_client = mocker.AsyncMock()
    init_client.__aenter__.return_value = init_client
    init_client.post = mocker.AsyncMock(return_value=init_resp)

    s3_client = mocker.AsyncMock()
    s3_client.__aenter__.return_value = s3_client
    s3_client.put = mocker.AsyncMock(return_value=s3_resp)

    mocker.patch(
        "open_amplify_ai.routers.files.httpx.AsyncClient",
        side_effect=[init_client, s3_client],
    )

    response = client.post(
        "/v1/files",
        files={"file": ("sample.pdf", io.BytesIO(b"fake pdf content"), "application/pdf")},
        data={"purpose": "assistants"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/new-file.json"
    assert data["filename"] == "sample.pdf"
    assert data["purpose"] == "assistants"
    assert data["bytes"] == len(b"fake pdf content")
    init_client.post.assert_called_once()
    s3_client.put.assert_called_once()


def test_upload_file_init_failure(mocker):
    """POST /v1/files returns 500 when Amplify upload init fails."""
    init_resp = mocker.Mock()
    init_resp.raise_for_status = mocker.Mock()
    init_resp.json.return_value = {"success": False, "error": "storage full"}

    init_client = mocker.AsyncMock()
    init_client.__aenter__.return_value = init_client
    init_client.post = mocker.AsyncMock(return_value=init_resp)

    mocker.patch(
        "open_amplify_ai.routers.files.httpx.AsyncClient", return_value=init_client
    )

    response = client.post(
        "/v1/files",
        files={"file": ("file.txt", io.BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == 500


def test_delete_file_success(mocker):
    """DELETE /v1/files/{file_id} calls Amplify POST /files op=/delete."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True}
    mocker.patch(
        "open_amplify_ai.routers.files.httpx.AsyncClient",
        return_value=_make_async_client(mocker, mock_response),
    )

    response = client.delete("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["deleted"] is True


def test_retrieve_file_content_success(mocker):
    """GET /v1/files/{file_id}/content returns binary via Amplify code interpreter download."""
    api_resp = mocker.Mock()
    api_resp.raise_for_status = mocker.Mock()
    api_resp.json.return_value = {
        "success": True,
        "downloadUrl": "https://s3.example.com/file.png",
    }

    content_resp = mocker.Mock()
    content_resp.raise_for_status = mocker.Mock()
    content_resp.content = b"\x89PNG\r\n"
    content_resp.headers = {"Content-Type": "image/png"}

    # retrieve_file_content uses two separate AsyncClient blocks: POST then GET
    api_client = mocker.AsyncMock()
    api_client.__aenter__.return_value = api_client
    api_client.post = mocker.AsyncMock(return_value=api_resp)

    s3_client = mocker.AsyncMock()
    s3_client.__aenter__.return_value = s3_client
    s3_client.get = mocker.AsyncMock(return_value=content_resp)

    mocker.patch(
        "open_amplify_ai.routers.files.httpx.AsyncClient",
        side_effect=[api_client, s3_client],
    )

    response = client.get("/v1/files/user@vu.edu/ast/file.png/content")
    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n"


def test_retrieve_file_content_not_found(mocker):
    """GET /v1/files/{file_id}/content returns 404 when downloadUrl is absent."""
    api_resp = mocker.Mock()
    api_resp.raise_for_status = mocker.Mock()
    api_resp.json.return_value = {"success": False, "message": "File not found"}

    api_client = mocker.AsyncMock()
    api_client.__aenter__.return_value = api_client
    api_client.post = mocker.AsyncMock(return_value=api_resp)

    mocker.patch(
        "open_amplify_ai.routers.files.httpx.AsyncClient", return_value=api_client
    )

    response = client.get("/v1/files/nonexistent/content")
    assert response.status_code == 404
