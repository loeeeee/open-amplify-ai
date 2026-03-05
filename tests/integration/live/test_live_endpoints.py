import io
import json
import os
import time
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# ---------------------------------------------------------------------------
# Integration Tests - kilo/cline and openclaw client usage patterns
#
# These tests run against the ACTUAL Amplify API using a real token.
# They exercise the specific request shapes that kilo, cline, and openclaw
# emit when pointed at this server, not just generic endpoint smoke-tests.
#
# To run:
#   AMPLIFY_AI_TOKEN="..." uv run pytest tests/integration/live/test_live_endpoints.py -v -s
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a live TestClient after verifying a real token is present."""
    token = os.environ.get("AMPLIFY_AI_TOKEN")
    if not token or token == "test-token-123":
        pytest.skip("AMPLIFY_AI_TOKEN is required for live integration tests.")

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Model Discovery
# kilo and cline call GET /v1/models on start-up to populate the model picker.
# ---------------------------------------------------------------------------


def test_kilo_cline_model_discovery(client: TestClient) -> None:
    """kilo/cline enumerate available models on start-up and expect OpenAI list shape."""
    response = client.get("/v1/models")
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["object"] == "list"
    assert len(data["data"]) > 0

    first = data["data"][0]
    assert first["object"] == "model"
    assert "id" in first
    assert first["owned_by"] == "amplify-ai"
    assert "created" in first


# ---------------------------------------------------------------------------
# Chat - system prompt + user message
# cline always prepends a system message before the user turn.
# ---------------------------------------------------------------------------


def test_kilo_cline_system_prompt_chat(client: TestClient) -> None:
    """cline sends a system role message followed by the user prompt."""
    req_body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. Be concise.",
            },
            {
                "role": "user",
                "content": "What is 2+2? Reply with just the number.",
            },
        ],
        "temperature": 0.0,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "4" in data["choices"][0]["message"]["content"]
    assert "system_fingerprint" in data


# ---------------------------------------------------------------------------
# Chat - streaming with include_usage
# kilo and cline request stream=True together with stream_options.include_usage=True.
# The server must emit a final usage chunk before [DONE].
# ---------------------------------------------------------------------------


def test_kilo_cline_streaming_with_usage(client: TestClient) -> None:
    """kilo/cline use stream=True + stream_options.include_usage=True and expect a usage chunk."""
    req_body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Say the word 'hello' only."},
        ],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200, response.text
    assert "text/event-stream" in response.headers.get("content-type", "")

    has_content_chunk = False
    has_usage_chunk = False
    has_done = False

    for raw_line in response.iter_lines():
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        if not raw_line:
            continue
        if not raw_line.startswith("data: "):
            continue

        payload = raw_line[6:]
        if payload == "[DONE]":
            has_done = True
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if choices and choices[0].get("delta", {}).get("content"):
            has_content_chunk = True

        # Usage chunk has empty choices list and a usage block
        if chunk.get("choices") == [] and "usage" in chunk:
            has_usage_chunk = True

    assert has_content_chunk, "No content delta chunks received"
    assert has_usage_chunk, "Missing final usage chunk (stream_options.include_usage=True)"
    assert has_done, "Stream did not terminate with [DONE]"


# ---------------------------------------------------------------------------
# Chat - list-typed content
# cline passes content as a list of {type, text} dicts for multi-part messages.
# The server must flatten them into a plain string before forwarding.
# ---------------------------------------------------------------------------


def test_kilo_cline_list_content_messages(client: TestClient) -> None:
    """cline passes content as a list of typed parts; server must handle without error."""
    req_body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is "},
                    {"type": "text", "text": "2+2? Reply with just the number."},
                ],
            }
        ],
        "temperature": 0.0,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["object"] == "chat.completion"
    assert "4" in data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Chat - extra message fields
# kilo includes a `name` field on messages. The server must not reject this.
# ---------------------------------------------------------------------------


def test_kilo_cline_extra_message_fields(client: TestClient) -> None:
    """kilo sends messages with extra fields like 'name'; server must not return 400."""
    req_body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2? Reply with just the number.",
                "name": "kilo-agent",
            }
        ],
        "temperature": 0.0,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["object"] == "chat.completion"


# ---------------------------------------------------------------------------
# Chat - tool call response parsing
# kilo expects the server to detect when the model emits a JSON tool call
# ({"command": ..., "parameters": ...}) and convert it to a proper tool_calls
# block with finish_reason=tool_calls.
# This test uses a normal prompt - if the model returns plain text that is fine.
# The server is expected to pass through plain text as a regular message.
# We verify at minimum that the response is well-formed regardless of branch.
# ---------------------------------------------------------------------------


def test_kilo_cline_tool_call_response(client: TestClient) -> None:
    """Server correctly handles any response shape - tool call or plain text."""
    req_body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "List the files in the current directory. Use a tool call if available.",
            }
        ],
        "temperature": 0.0,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]

    finish_reason = choice["finish_reason"]
    assert finish_reason in ("stop", "tool_calls"), (
        f"Unexpected finish_reason: {finish_reason}"
    )

    if finish_reason == "tool_calls":
        tool_calls = choice["message"].get("tool_calls", [])
        assert len(tool_calls) > 0
        tc = tool_calls[0]
        assert tc["type"] == "function"
        assert "name" in tc["function"]
        # arguments must be valid JSON
        json.loads(tc["function"]["arguments"])
    else:
        assert choice["message"].get("content") is not None


# ---------------------------------------------------------------------------
# File upload and use (openclaw pattern)
# openclaw uploads a document then confirms it is visible in the file list.
# ---------------------------------------------------------------------------


def test_openclaw_file_upload_and_use(client: TestClient) -> None:
    """openclaw uploads a file and verifies it appears in the file list."""
    file_content = b"Integration test document for openclaw."
    upload_resp = client.post(
        "/v1/files",
        files={
            "file": (
                "openclaw_test.txt",
                io.BytesIO(file_content),
                "text/plain",
            )
        },
    )
    assert upload_resp.status_code == 200, upload_resp.text
    file_data = upload_resp.json()
    file_id = file_data["id"]

    assert file_data["object"] == "file"
    assert file_data["filename"] == "openclaw_test.txt"
    assert file_data["bytes"] == len(file_content)

    # Allow S3 indexing time before querying
    time.sleep(1.5)

    list_resp = client.get("/v1/files")
    assert list_resp.status_code == 200, list_resp.text
    files = list_resp.json()["data"]
    assert any(f["id"] == file_id for f in files), (
        f"Uploaded file {file_id} not found in file list"
    )

    # Attempt deletion; upstream may return 403 - treat as a known limitation
    del_resp = client.delete(f"/v1/files/{file_id}")
    if del_resp.status_code == 500 and "Forbidden" in del_resp.text:
        pytest.xfail("Amplify file delete returns 403 Forbidden upstream")
    assert del_resp.status_code == 200, del_resp.text
    assert del_resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Assistant with file attachment (openclaw pattern)
# openclaw creates an assistant, uploads a file, and attaches it for RAG.
# Marked xfail because the upstream assistant create endpoint is unstable.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Amplify assistant create endpoint intermittently returns 502 Bad Gateway"
)
def test_openclaw_assistant_with_file(client: TestClient) -> None:
    """openclaw creates an assistant and attaches an uploaded file for grounding."""
    # Step 1: upload file
    file_content = b"Context document for the openclaw assistant test."
    upload_resp = client.post(
        "/v1/files",
        files={
            "file": (
                "openclaw_ctx.txt",
                io.BytesIO(file_content),
                "text/plain",
            )
        },
    )
    assert upload_resp.status_code == 200, upload_resp.text
    file_id = upload_resp.json()["id"]

    # Step 2: create assistant - will raise if upstream is down
    create_body = {
        "name": "openclaw Integration Test Assistant",
        "model": "gpt-4o",
        "instructions": "Answer questions using the attached context document.",
    }
    create_resp = client.post("/v1/assistants", json=create_body)
    assert create_resp.status_code == 200, create_resp.text
    assistant_data = create_resp.json()
    assistant_id = assistant_data["id"]

    assert assistant_id.startswith("astp/")
    assert assistant_data["name"] == "openclaw Integration Test Assistant"

    try:
        time.sleep(1.0)
        # Step 3: retrieve to confirm creation persisted
        get_resp = client.get(f"/v1/assistants/{assistant_id}")
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["id"] == assistant_id
    finally:
        # Step 4: cleanup
        del_resp = client.delete(f"/v1/assistants/{assistant_id}")
        assert del_resp.status_code == 200, del_resp.text
        assert del_resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Vector Store Lifecycle (existing smoke test retained)
# ---------------------------------------------------------------------------


def test_live_vector_store_lifecycle(client: TestClient) -> None:
    """Create, retrieve, and delete a vector store (Amplify tag)."""
    create_resp = client.post(
        "/v1/vector_stores", json={"name": "integration-test-store"}
    )
    assert create_resp.status_code == 200, create_resp.text
    vs_id = create_resp.json()["id"]

    try:
        time.sleep(0.5)
        get_resp = client.get(f"/v1/vector_stores/{vs_id}")
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["id"] == vs_id
        assert get_resp.json()["file_counts"]["total"] == 0
    finally:
        del_resp = client.delete(f"/v1/vector_stores/{vs_id}")
        assert del_resp.status_code == 200, del_resp.text
        assert del_resp.json()["deleted"] is True
