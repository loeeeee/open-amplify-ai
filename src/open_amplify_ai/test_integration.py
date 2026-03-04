import os
import time
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# ---------------------------------------------------------------------------
# Integration Tests for Amplify AI Compatible Server
#
# These tests run against the ACTUAL Amplify API using your provided token.
# To run these tests:
#   AMPLIFY_AI_TOKEN="..." uv run pytest src/open_amplify_ai/test_integration.py -v -s
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    # Verify we have a real token
    token = os.environ.get("AMPLIFY_AI_TOKEN")
    if not token or token == "test-token-123":
        pytest.skip("AMPLIFY_AI_TOKEN is required for live integration tests.")
    
    with TestClient(app) as c:
        yield c

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_live_models_list(client):
    """Fetch real available models from Amplify and verify OpenAI shape."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    
    # Check first model
    first = data["data"][0]
    assert first["object"] == "model"
    assert "id" in first
    assert first["owned_by"] == "amplify-ai"
    assert "created" in first


def test_live_models_retrieve(client):
    """Retrieve a single real model by ID (picking first from list)."""
    # Get the list first to pick a valid model ID (e.g. gpt-4o)
    list_resp = client.get("/v1/models")
    assert list_resp.status_code == 200
    first_model_id = list_resp.json()["data"][0]["id"]
    
    response = client.get(f"/v1/models/{first_model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == first_model_id
    assert data["object"] == "model"


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------

def test_live_chat_completion(client):
    """Run a real standard chat completion."""
    req_body = {
        "model": "gpt-4o",  # Usually a safe bet, adjust if Vanderbilt changes defaults
        "messages": [
            {"role": "user", "content": "What is 2+2? Say exactly '4'."}
        ],
        "temperature": 0.0,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["finish_reason"] == "stop"
    content = data["choices"][0]["message"]["content"]
    assert "4" in content


def test_live_chat_completion_streaming(client):
    """Run a real streaming chat completion."""
    req_body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Count from 1 to 3 rapidly."}
        ],
        "temperature": 0.5,
        "stream": True
    }
    
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    
    chunks = response.iter_lines()
    has_data = False
    has_done = False
    
    for chunk_str in chunks:
        if isinstance(chunk_str, bytes):
            chunk_str = chunk_str.decode("utf-8")
        if not chunk_str: continue
        
        if chunk_str.startswith("data: "):
            if "[DONE]" in chunk_str:
                has_done = True
                break
            has_data = True
            # Could parse json here to verify fields if needed

    assert has_data is True
    assert has_done is True


# ---------------------------------------------------------------------------
# Assistants Lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Amplify Assistant API is currently unstable and often returns 502 Bad Gateway")
@pytest.mark.xfail(reason="Amplify Assistant API is currently unstable and often returns 502 Bad Gateway")
def test_live_assistant_lifecycle(client):
    """Create, retrieve, modify, and delete a live assistant."""
    # 1. Create
    create_body = {
        "name": "Integration Test Assistant",
        "model": "gpt-4o",
        "instructions": "You are a test."
    }
    create_resp = client.post("/v1/assistants", json=create_body)
    assert create_resp.status_code == 200, create_resp.text
    create_data = create_resp.json()
    ast_id = create_data["id"]
    
    assert ast_id.startswith("astp/")
    assert create_data["name"] == "Integration Test Assistant"

    try:
        # Give remote systems a tiny moment
        time.sleep(1.0)
        
        # 2. Retrieve
        get_resp = client.get(f"/v1/assistants/{ast_id}")
        assert get_resp.status_code == 200, get_resp.text
        get_data = get_resp.json()
        assert get_data["id"] == ast_id
        assert get_data["name"] == "Integration Test Assistant"
        
        # 3. Modify
        update_body = {
            "name": "Integration Test Assistant Updated",
            "instructions": "You are a modified test."
        }
        mod_resp = client.post(f"/v1/assistants/{ast_id}", json=update_body)
        assert mod_resp.status_code == 200, mod_resp.text
        mod_data = mod_resp.json()
        assert mod_data["name"] == "Integration Test Assistant Updated"
        
    finally:
        # 4. Delete
        del_resp = client.delete(f"/v1/assistants/{ast_id}")
        assert del_resp.status_code == 200, del_resp.text
        assert del_resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Vector Store Lifecycle
# ---------------------------------------------------------------------------

def test_live_vector_store_lifecycle(client):
    """Create, retrieve, and delete a vector store (Amplify Tag)."""
    # 1. Create
    create_body = {
        "name": "integration-test-store"
    }
    create_resp = client.post("/v1/vector_stores", json=create_body)
    assert create_resp.status_code == 200, create_resp.text
    vs_id = create_resp.json()["id"]

    try:
        time.sleep(0.5)
        # 2. Retrieve
        get_resp = client.get(f"/v1/vector_stores/{vs_id}")
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["id"] == vs_id
        assert get_resp.json()["file_counts"]["total"] == 0
    finally:
        # 3. Delete
        del_resp = client.delete(f"/v1/vector_stores/{vs_id}")
        # Could be 200 or failing depending on if the tag was fully commited 
        # inside Amplify's async tag creation. We assert it attempted.
        assert del_resp.status_code == 200, del_resp.text
        assert del_resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Files Lifecycle
# ---------------------------------------------------------------------------

def test_live_files_lifecycle(client):
    """Upload, list, and delete a real file in Amplify."""
    import io
    
    file_content = b"Hello, integration test world!"
    upload_resp = client.post(
        "/v1/files",
        files={"file": ("integration_test_file.txt", io.BytesIO(file_content), "text/plain")}
    )
    assert upload_resp.status_code == 200, upload_resp.text
    file_data = upload_resp.json()
    file_id = file_data["id"]
    
    try:
        # Give remote s3 uploads a moment
        time.sleep(1.0)
        
        # 1. List files, verify ours is there
        list_resp = client.get("/v1/files")
        assert list_resp.status_code == 200, list_resp.text
        files = list_resp.json()["data"]
        assert any(f["id"] == file_id for f in files)
        
    finally:
        # 2. Delete file
        del_resp = client.delete(f"/v1/files/{file_id}")
        
        # We know file delete currently returns 403 Forbidden (wrapped in 500 internally)
        if del_resp.status_code == 500 and "Forbidden" in del_resp.text:
            pytest.xfail("Amplify File Delete API is currently returning 403 Forbidden")
            
        assert del_resp.status_code == 200, del_resp.text
        assert del_resp.json()["deleted"] is True
