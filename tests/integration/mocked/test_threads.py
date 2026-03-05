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




# ===========================================================================
# THREADS
# ===========================================================================


def test_client_deletes_thread(mocker: Any) -> None:
    """Client deletes a thread via Amplify DELETE with query param."""
    mocker.patch(
        "open_amplify_ai.routers.threads.requests.delete",
        return_value=type("MockResponse", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True, "message": "Thread deleted"},
        })(),
    )

    response = client.delete("/v1/threads/thread-abc123")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "thread-abc123"
    assert data["object"] == "thread.deleted"
    assert data["deleted"] is True


def test_client_deletes_thread_with_slash_id(mocker: Any) -> None:
    """Thread IDs may contain slashes (email-style); server must handle."""
    mocker.patch(
        "open_amplify_ai.routers.threads.requests.delete",
        return_value=type("MockResponse", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"success": True},
        })(),
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
