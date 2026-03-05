import io
import pytest
import os
import requests
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# Set up dummy environment variable for tests to bypass token validation failure
os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)



def test_list_files_success(mocker):
    """GET /v1/files returns OpenAI file list mapped from Amplify /files/query."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "items": [
                {
                    "id": "user@vu.edu/2024-01-01/abc.json",
                    "name": "test.pdf",
                    "createdAt": "2024-01-01T00:00:00",
                    "totalTokens": 100,
                    "type": "application/pdf",
                }
            ],
            "pageKey": None,
        },
    }
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_response)

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
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {"items": [], "pageKey": None},
    }
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_response)

    response = client.get("/v1/files")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"] == []


def test_retrieve_file_success(mocker):
    """GET /v1/files/{file_id} returns a single file when found."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "items": [
                {
                    "id": "user@vu.edu/2024-01-01/abc.json",
                    "name": "test.pdf",
                    "createdAt": "2024-01-01T00:00:00",
                    "totalTokens": 200,
                    "type": "application/pdf",
                }
            ],
            "pageKey": None,
        },
    }
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_response)

    response = client.get("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["filename"] == "test.pdf"


def test_retrieve_file_not_found(mocker):
    """GET /v1/files/{file_id} returns 404 when file is not in the list."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {"items": [], "pageKey": None},
    }
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_response)

    response = client.get("/v1/files/nonexistent-file-id")
    assert response.status_code == 404


def test_upload_file_success(mocker):
    """POST /v1/files uploads a file via two-step Amplify + S3 PUT and returns a File object."""
    init_mock = mocker.Mock()
    init_mock.raise_for_status = mocker.Mock()
    init_mock.json.return_value = {
        "success": True,
        "uploadUrl": "https://s3.example.com/upload?signed=true",
        "key": "user@vu.edu/2024-01-01/new-file.json",
        "statusUrl": "",
        "contentUrl": "",
        "metadataUrl": "",
    }

    s3_mock = mocker.Mock()
    s3_mock.raise_for_status = mocker.Mock()

    post_mock = mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=init_mock)
    put_mock = mocker.patch("open_amplify_ai.routers.files.requests.put", return_value=s3_mock)

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
    post_mock.assert_called_once()
    put_mock.assert_called_once()


def test_upload_file_init_failure(mocker):
    """POST /v1/files returns 500 when Amplify upload init fails."""
    init_mock = mocker.Mock()
    init_mock.raise_for_status = mocker.Mock()
    init_mock.json.return_value = {"success": False, "error": "storage full"}
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=init_mock)

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
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=mock_response)

    response = client.delete("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "file"
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["deleted"] is True


def test_retrieve_file_content_success(mocker):
    """GET /v1/files/{file_id}/content returns binary via Amplify code interpreter download."""
    api_mock = mocker.Mock()
    api_mock.raise_for_status = mocker.Mock()
    api_mock.json.return_value = {
        "success": True,
        "downloadUrl": "https://s3.example.com/file.png",
    }
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=api_mock)

    content_mock = mocker.Mock()
    content_mock.raise_for_status = mocker.Mock()
    content_mock.content = b"\x89PNG\r\n"
    content_mock.headers = {"Content-Type": "image/png"}
    mocker.patch("open_amplify_ai.routers.files.requests.get", return_value=content_mock)

    response = client.get("/v1/files/user@vu.edu/ast/file.png/content")
    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n"


def test_retrieve_file_content_not_found(mocker):
    """GET /v1/files/{file_id}/content returns 404 when downloadUrl is absent."""
    api_mock = mocker.Mock()
    api_mock.raise_for_status = mocker.Mock()
    api_mock.json.return_value = {"success": False, "message": "File not found"}
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=api_mock)

    response = client.get("/v1/files/nonexistent/content")
    assert response.status_code == 404

