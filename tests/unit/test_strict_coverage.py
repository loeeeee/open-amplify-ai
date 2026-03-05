import io
import pytest
import os
import requests
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# Set up dummy environment variable for tests to bypass token validation failure
os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)



def test_missing_auth_token():
    """Endpoints should return 401 if AMPLIFY_AI_TOKEN is missing."""
    # Temporarily remove token
    original_token = os.environ.get("AMPLIFY_AI_TOKEN")
    if "AMPLIFY_AI_TOKEN" in os.environ:
        del os.environ["AMPLIFY_AI_TOKEN"]

    try:
        response = client.get("/v1/models")
        assert response.status_code == 401
        assert response.json()["detail"] == "Amplify AI token not configured"
    finally:
        # Restore token
        if original_token is not None:
            os.environ["AMPLIFY_AI_TOKEN"] = original_token


def test_upload_file_s3_put_failure(mocker):
    """POST /v1/files returns 500 when S3 PUT fails after initial Amplify POST succeeds."""
    init_mock = mocker.Mock()
    init_mock.raise_for_status = mocker.Mock()
    init_mock.json.return_value = {
        "success": True,
        "uploadUrl": "https://s3.example.com/upload?signed=true",
        "key": "user/123.pdf",
    }
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=init_mock)

    s3_mock = mocker.Mock()
    s3_mock.raise_for_status = mocker.Mock(side_effect=requests.exceptions.RequestException("S3 Error"))
    mocker.patch("open_amplify_ai.routers.files.requests.put", return_value=s3_mock)

    response = client.post(
        "/v1/files",
        files={"file": ("file.txt", io.BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == 500
    assert "Error communicating with Amplify AI" in response.json()["detail"]


def test_delete_file_amplify_failure(mocker):
    """DELETE /v1/files/{file_id} returns deleted=False when Amplify returns success=False."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": False, "error": "Not found"}
    mocker.patch("open_amplify_ai.routers.files.requests.post", return_value=mock_response)

    response = client.delete("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is False


def test_chat_completions_json_invalid(mocker):
    """POST /v1/chat/completions yields raw string on invalid JSON in SSE."""
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__ = mocker.Mock(return_value=mock_cm)
    mock_cm.__exit__ = mocker.Mock(return_value=False)
    mock_cm.status_code = 200
    mock_cm.raise_for_status = mocker.Mock()
    mock_cm.iter_lines = mocker.Mock(
        return_value=[b'data: {invalid json snippet']
    )
    mocker.patch("open_amplify_ai.routers.chat.requests.post", return_value=mock_cm)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert "{invalid json snippet" in response.text


# ---------------------------------------------------------------------------
# 501 stubs - parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        # Embeddings
        ("POST", "/v1/embeddings"),
        # Audio
        ("POST", "/v1/audio/speech"),
        ("POST", "/v1/audio/transcriptions"),
        ("POST", "/v1/audio/translations"),
        # Images
        ("POST", "/v1/images/generations"),
        ("POST", "/v1/images/edits"),
        ("POST", "/v1/images/variations"),
        # Fine-tuning
        ("POST", "/v1/fine_tuning/jobs"),
        ("GET", "/v1/fine_tuning/jobs"),
        ("GET", "/v1/fine_tuning/jobs/job-123"),
        ("POST", "/v1/fine_tuning/jobs/job-123/cancel"),
        ("GET", "/v1/fine_tuning/jobs/job-123/events"),
        # Moderations
        ("POST", "/v1/moderations"),
        # Batch
        ("POST", "/v1/batches"),
        ("GET", "/v1/batches"),
        ("GET", "/v1/batches/batch-123"),
        ("POST", "/v1/batches/batch-123/cancel"),
        # Threads (create, get, modify)
        ("POST", "/v1/threads"),
        ("GET", "/v1/threads/thread-123"),
        ("POST", "/v1/threads/thread-123"),
        # Messages
        ("POST", "/v1/threads/thread-123/messages"),
        ("GET", "/v1/threads/thread-123/messages"),
        ("GET", "/v1/threads/thread-123/messages/msg-123"),
        # Runs
        ("POST", "/v1/threads/thread-123/runs"),
        ("GET", "/v1/threads/thread-123/runs"),
        ("GET", "/v1/threads/thread-123/runs/run-123"),
        ("POST", "/v1/threads/thread-123/runs/run-123/cancel"),
        ("POST", "/v1/threads/thread-123/runs/run-123/submit_tool_outputs"),
        ("POST", "/v1/threads/runs"),
        # Run steps
        ("GET", "/v1/threads/thread-123/runs/run-123/steps"),
        ("GET", "/v1/threads/thread-123/runs/run-123/steps/step-123"),
        # Vector store: modify, file batches
        ("POST", "/v1/vector_stores/store-123"),
        ("DELETE", "/v1/vector_stores/store-123/files/file-abc"),
        ("POST", "/v1/vector_stores/store-123/file_batches"),
        ("GET", "/v1/vector_stores/store-123/file_batches/batch-123"),
        ("POST", "/v1/vector_stores/store-123/file_batches/batch-123/cancel"),
        ("GET", "/v1/vector_stores/store-123/file_batches/batch-123/files"),
    ],
)
def test_unsupported_endpoints_return_501(method, path):
    """All unsupported endpoints must return 501 Not Implemented."""
    response = client.request(method, path, json={})
    assert response.status_code == 501, (
        f"Expected 501 for {method} {path}, got {response.status_code}: {response.text}"
    )

