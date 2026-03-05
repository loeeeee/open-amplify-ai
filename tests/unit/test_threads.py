import io
import pytest
import os
import requests
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# Set up dummy environment variable for tests to bypass token validation failure
os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)



def test_delete_thread_success(mocker):
    """DELETE /v1/threads/{id} maps to Amplify /assistant/openai/thread/delete."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "message": "Thread deleted successfully"}
    mocker.patch("open_amplify_ai.routers.threads.requests.delete", return_value=mock_response)

    response = client.delete("/v1/threads/thread-abc123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "thread-abc123"
    assert data["object"] == "thread.deleted"
    assert data["deleted"] is True


def test_delete_thread_with_slash_id(mocker):
    """DELETE /v1/threads/{id} handles email-style thread IDs containing slashes."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True}
    mocker.patch("open_amplify_ai.routers.threads.requests.delete", return_value=mock_response)

    response = client.delete("/v1/threads/user@vu.edu/thr/abc-123")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is True


