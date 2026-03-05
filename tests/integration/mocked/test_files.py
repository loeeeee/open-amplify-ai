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


def _make_amplify_success_response(data: Any = None) -> Any:
    """Build a generic success mock response."""
    resp_data = {"success": True}
    if data is not None:
        resp_data["data"] = data
    mock = type("MockResponse", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: resp_data,
    })()
    return mock


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


def test_client_uploads_file(mocker: Any) -> None:
    """Openclaw uploads a file via two-step Amplify init + S3 PUT."""
    init_mock = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {
            "success": True,
            "uploadUrl": "https://s3.example.com/upload?signed=true",
            "key": "user@vu.edu/2024-01-01/new-file.json",
            "statusUrl": "",
            "contentUrl": "",
            "metadataUrl": "",
        },
    })()

    s3_mock = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
    })()

    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=init_mock)
    mocker.patch("open_amplify_ai.routers.files.requests.put", return_value=s3_mock)

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
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_files_query_response([
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
        ]),
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
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_files_query_response([]),
    )

    response = client.get("/v1/files")
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_client_retrieves_file(mocker: Any) -> None:
    """Openclaw retrieves a single file by ID."""
    mocker.patch(
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_files_query_response([
            {
                "id": "user@vu.edu/2024-01-01/abc.json",
                "name": "test.pdf",
                "createdAt": "2024-01-01T00:00:00",
                "totalTokens": 200,
                "type": "application/pdf",
            },
        ]),
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
        "open_amplify_ai.utils.requests.post",
        return_value=_make_amplify_files_query_response([]),
    )

    response = client.get("/v1/files/nonexistent-file-id")
    assert response.status_code == 404


def test_client_deletes_file(mocker: Any) -> None:
    """Openclaw deletes a file (base64-encoded Amplify dispatch)."""
    mocker.patch(
        "open_amplify_ai.routers.files.requests.post",
        return_value=_make_amplify_success_response(),
    )

    response = client.delete("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["deleted"] is True


def test_client_downloads_file_content(mocker: Any) -> None:
    """Openclaw downloads file content via Code Interpreter proxy."""
    api_mock = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {
            "success": True,
            "downloadUrl": "https://s3.example.com/file.png",
        },
    })()
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=api_mock)

    content_mock = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
        "content": b"\x89PNG\r\n",
        "headers": {"Content-Type": "image/png"},
    })()
    mocker.patch("open_amplify_ai.routers.files.requests.get", return_value=content_mock)

    response = client.get("/v1/files/user@vu.edu/ast/file.png/content")
    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n"


def test_client_downloads_file_content_not_found(mocker: Any) -> None:
    """Openclaw gets 404 when file content download URL is absent."""
    api_mock = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {"success": False, "message": "File not found"},
    })()
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=api_mock)

    response = client.get("/v1/files/nonexistent/content")
    assert response.status_code == 404


def test_client_upload_file_init_failure(mocker: Any) -> None:
    """Server returns 500 when Amplify upload init fails."""
    init_mock = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {"success": False, "error": "storage full"},
    })()
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=init_mock)

    response = client.post(
        "/v1/files",
        files={"file": ("file.txt", io.BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == 500
