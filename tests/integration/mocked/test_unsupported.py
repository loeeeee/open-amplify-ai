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
# UNSUPPORTED ENDPOINTS - All return 501
# ===========================================================================


@pytest.mark.parametrize("method,path", [
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
    # Batches
    ("POST", "/v1/batches"),
    ("GET", "/v1/batches"),
    ("GET", "/v1/batches/batch-123"),
    ("POST", "/v1/batches/batch-123/cancel"),
    # Thread messages
    ("POST", "/v1/threads/thread-123/messages"),
    ("GET", "/v1/threads/thread-123/messages"),
    ("GET", "/v1/threads/thread-123/messages/msg-123"),
    # Thread runs
    ("POST", "/v1/threads/thread-123/runs"),
    ("GET", "/v1/threads/thread-123/runs"),
    ("GET", "/v1/threads/thread-123/runs/run-123"),
    ("POST", "/v1/threads/thread-123/runs/run-123/cancel"),
    ("POST", "/v1/threads/thread-123/runs/run-123/submit_tool_outputs"),
    ("POST", "/v1/threads/runs"),
    # Run steps
    ("GET", "/v1/threads/thread-123/runs/run-123/steps"),
    ("GET", "/v1/threads/thread-123/runs/run-123/steps/step-123"),
    # Vector store stubs
    ("DELETE", "/v1/vector_stores/store-123/files/file-abc"),
    ("POST", "/v1/vector_stores/store-123/file_batches"),
    ("GET", "/v1/vector_stores/store-123/file_batches/batch-123"),
    ("POST", "/v1/vector_stores/store-123/file_batches/batch-123/cancel"),
    ("GET", "/v1/vector_stores/store-123/file_batches/batch-123/files"),
])
def test_unsupported_endpoint_returns_501(method: str, path: str) -> None:
    """All unsupported OpenAI endpoints return 501 Not Implemented."""
    if method == "GET":
        response = client.get(path)
    elif method == "POST":
        response = client.post(path, json={})
    elif method == "DELETE":
        response = client.delete(path)
    else:
        pytest.fail(f"Unknown method: {method}")

    assert response.status_code == 501, (
        f"{method} {path} returned {response.status_code}, expected 501"
    )
