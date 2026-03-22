"""Comprehensive integration tests simulating cline/kilo/openclaw usage patterns.

These tests exercise the full FastAPI request/response cycle with mocked
Amplify upstream calls. No live AMPLIFY_AI_TOKEN is needed.

To run:
    nix-shell --run "uv run pytest tests/integration/mocked/test_threads.py -v"
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


# ===========================================================================
# THREADS
# ===========================================================================


def test_client_deletes_thread(mocker: Any) -> None:
    """Client deletes a thread via Amplify DELETE with query param."""
    resp = _make_json_response(mocker, {"success": True, "message": "Thread deleted"})

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.delete = mocker.AsyncMock(return_value=resp)
    mocker.patch(
        "open_amplify_ai.routers.threads.httpx.AsyncClient", return_value=mock_client
    )

    response = client.delete("/v1/threads/thread-abc123")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "thread-abc123"
    assert data["object"] == "thread.deleted"
    assert data["deleted"] is True


def test_client_deletes_thread_with_slash_id(mocker: Any) -> None:
    """Thread IDs may contain slashes (email-style); server must handle."""
    resp = _make_json_response(mocker, {"success": True})

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.delete = mocker.AsyncMock(return_value=resp)
    mocker.patch(
        "open_amplify_ai.routers.threads.httpx.AsyncClient", return_value=mock_client
    )

    response = client.delete("/v1/threads/user@vu.edu/thr/abc-123")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_client_thread_stubs_return_501() -> None:
    """Thread create/retrieve/modify all return 501 (not implemented)."""
    create_resp = client.post("/v1/threads", json={})
    assert create_resp.status_code == 501

    retrieve_resp = client.get("/v1/threads/thread-123")
    assert retrieve_resp.status_code == 501

    modify_resp = client.post("/v1/threads/thread-123", json={})
    assert modify_resp.status_code == 501
