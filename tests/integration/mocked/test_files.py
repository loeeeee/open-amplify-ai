"""Comprehensive integration tests simulating cline/kilo/openclaw usage patterns.

These tests exercise the full FastAPI request/response cycle with mocked
Amplify upstream calls. No live AMPLIFY_AI_TOKEN is needed.

To run:
    nix-shell --run "uv run pytest tests/integration/mocked/test_files.py -v"
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
    mock_client.put = mocker.AsyncMock(return_value=response)
    return mock_client


def _make_files_query_client(mocker: Any, items: List[Dict[str, Any]]) -> Any:
    """Build an async client returning a /files/query response (targets utils module)."""
    resp = _make_json_response(mocker, {
        "success": True,
        "data": {"items": items, "pageKey": None},
    })
    return _make_async_client(mocker, resp)


# ===========================================================================
# FILES - openclaw pattern
# ===========================================================================


def test_client_uploads_file(mocker: Any) -> None:
    """Openclaw uploads a file via two-step Amplify init + S3 PUT."""
    init_resp = _make_json_response(mocker, {
        "success": True,
        "uploadUrl": "https://s3.example.com/upload?signed=true",
        "key": "user@vu.edu/2024-01-01/new-file.json",
        "statusUrl": "",
        "contentUrl": "",
        "metadataUrl": "",
    })

    s3_resp = mocker.Mock()
    s3_resp.raise_for_status = mocker.Mock()

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


def test_client_lists_files(mocker: Any) -> None:
    """Openclaw lists all uploaded files."""
    items = [
        {
            "id": "user@vu.edu/2024-01-01/abc.json",
            "name": "test.pdf",
            "createdAt": "2024-01-01T00:00:00",
            "totalTokens": 100,
            "type": "application/pdf",
        },
        {
            "id": "user@vu.edu/2024-01-02/def.json",
            "name": "readme.md",
            "createdAt": "2024-01-02T00:00:00",
            "totalTokens": 50,
            "type": "text/markdown",
        },
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_files_query_client(mocker, items),
    )

    response = client.get("/v1/files")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "file"
    assert data["data"][0]["filename"] == "test.pdf"
    assert data["data"][1]["filename"] == "readme.md"
    assert data["data"][0]["purpose"] == "assistants"


def test_client_lists_files_empty(mocker: Any) -> None:
    """Openclaw gets empty list when no files exist."""
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_files_query_client(mocker, []),
    )

    response = client.get("/v1/files")
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_client_retrieves_file(mocker: Any) -> None:
    """Openclaw retrieves a single file by ID."""
    items = [
        {
            "id": "user@vu.edu/2024-01-01/abc.json",
            "name": "test.pdf",
            "createdAt": "2024-01-01T00:00:00",
            "totalTokens": 200,
            "type": "application/pdf",
        },
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_files_query_client(mocker, items),
    )

    response = client.get("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["filename"] == "test.pdf"


def test_client_retrieves_file_not_found(mocker: Any) -> None:
    """Openclaw gets 404 when file is not in the list."""
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_files_query_client(mocker, []),
    )

    response = client.get("/v1/files/nonexistent-file-id")
    assert response.status_code == 404


def test_client_deletes_file(mocker: Any) -> None:
    """Openclaw deletes a file (base64-encoded Amplify dispatch)."""
    resp = _make_json_response(mocker, {"success": True})
    mocker.patch(
        "open_amplify_ai.routers.files.httpx.AsyncClient",
        return_value=_make_async_client(mocker, resp),
    )

    response = client.delete("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["deleted"] is True


def test_client_downloads_file_content(mocker: Any) -> None:
    """Openclaw downloads file content via Code Interpreter proxy."""
    api_resp = _make_json_response(mocker, {
        "success": True,
        "downloadUrl": "https://s3.example.com/file.png",
    })

    content_resp = mocker.Mock()
    content_resp.raise_for_status = mocker.Mock()
    content_resp.content = b"\x89PNG\r\n"
    content_resp.headers = {"Content-Type": "image/png"}

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


def test_client_downloads_file_content_not_found(mocker: Any) -> None:
    """Openclaw gets 404 when file content download URL is absent."""
    resp = _make_json_response(mocker, {"success": False, "message": "File not found"})

    api_client = mocker.AsyncMock()
    api_client.__aenter__.return_value = api_client
    api_client.post = mocker.AsyncMock(return_value=resp)

    mocker.patch(
        "open_amplify_ai.routers.files.httpx.AsyncClient", return_value=api_client
    )

    response = client.get("/v1/files/nonexistent/content")
    assert response.status_code == 404


def test_client_upload_file_init_failure(mocker: Any) -> None:
    """Server returns 500 when Amplify upload init fails."""
    init_resp = _make_json_response(mocker, {"success": False, "error": "storage full"})

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
