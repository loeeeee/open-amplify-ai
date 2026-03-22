import io
import os
import pytest
import httpx
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def test_missing_auth_token():
    """Endpoints should return 401 if AMPLIFY_AI_TOKEN is missing."""
    original_token = os.environ.get("AMPLIFY_AI_TOKEN")
    if "AMPLIFY_AI_TOKEN" in os.environ:
        del os.environ["AMPLIFY_AI_TOKEN"]

    try:
        response = client.get("/v1/models")
        assert response.status_code == 401
        assert response.json()["detail"] == "Amplify AI token not configured"
    finally:
        if original_token is not None:
            os.environ["AMPLIFY_AI_TOKEN"] = original_token


def test_upload_file_s3_put_failure(mocker):
    """POST /v1/files returns 500 when S3 PUT fails after initial Amplify POST succeeds."""
    init_resp = mocker.Mock()
    init_resp.raise_for_status = mocker.Mock()
    init_resp.json.return_value = {
        "success": True,
        "uploadUrl": "https://s3.example.com/upload?signed=true",
        "key": "user/123.pdf",
    }

    s3_resp = mocker.Mock()
    s3_resp.raise_for_status = mocker.Mock(
        side_effect=httpx.HTTPStatusError(
            "S3 Error",
            request=mocker.Mock(),
            response=mocker.Mock(text="S3 Error"),
        )
    )

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
        files={"file": ("file.txt", io.BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == 500
    assert "Error communicating with Amplify AI" in response.json()["detail"]


def test_delete_file_amplify_failure(mocker):
    """DELETE /v1/files/{file_id} returns deleted=False when Amplify returns success=False."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": False, "error": "Not found"}

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("open_amplify_ai.routers.files.httpx.AsyncClient", return_value=mock_client)

    response = client.delete("/v1/files/user@vu.edu/2024-01-01/abc.json")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is False


def test_chat_completions_json_invalid(mocker):
    """POST /v1/chat/completions yields raw string on invalid JSON in SSE."""
    async def fake_aiter_lines():
        yield "data: {invalid json snippet"

    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = mocker.Mock()
    mock_resp.aiter_lines = fake_aiter_lines

    mock_stream_cm = mocker.AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_resp

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = mocker.Mock(return_value=mock_stream_cm)
    mocker.patch("open_amplify_ai.utils.httpx.AsyncClient", return_value=mock_client)

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
