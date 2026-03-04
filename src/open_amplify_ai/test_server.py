import io
import pytest
import os
import requests
from fastapi.testclient import TestClient
from open_amplify_ai.server import app

# Set up dummy environment variable for tests to bypass token validation failure
os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_get_models_success(mocker):
    """GET /v1/models returns OpenAI list format from Amplify /available_models."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o", "provider": "Azure"}
            ]
        },
    }
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "gpt-4o"
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "amplify-ai"
    assert "created" in data["data"][0]


def test_get_models_amplify_failure(mocker):
    """GET /v1/models returns 500 when Amplify signals failure."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": False, "error": "Some error"}
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to fetch models from Amplify AI"


def test_get_model_by_id_success(mocker):
    """GET /v1/models/{model} returns a single model when found."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": {
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "claude-3", "name": "Claude 3"},
            ]
        },
    }
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models/gpt-4o")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "gpt-4o"
    assert data["object"] == "model"
    assert "created" in data
    assert data["owned_by"] == "amplify-ai"


def test_get_model_by_id_not_found(mocker):
    """GET /v1/models/{model} returns 404 when model is absent."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": {"models": [{"id": "gpt-4o"}]},
    }
    mocker.patch("open_amplify_ai.routers.models.requests.get", return_value=mock_response)

    response = client.get("/v1/models/nonexistent-model")
    assert response.status_code == 404


def test_delete_model_not_allowed():
    """DELETE /v1/models/{model} always returns 405."""
    response = client.delete("/v1/models/gpt-4o")
    assert response.status_code == 405


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def test_chat_completions_success(mocker):
    """POST /v1/chat/completions returns OpenAI format from Amplify /chat."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": "This is a mocked response from Amplify.",
    }
    mocker.patch("open_amplify_ai.routers.chat.requests.post", return_value=mock_response)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4o"
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "This is a mocked response from Amplify."
    assert "usage" in data
    assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_invalid_request():
    """POST /v1/chat/completions returns 400 when messages are malformed."""
    req_body = {"messages": "this is not a list"}
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 400


def test_chat_completions_extra_fields(mocker):
    """POST /v1/chat/completions ignores extra fields like 'name' in messages without failing."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": "ok",
    }
    mocker.patch("open_amplify_ai.routers.chat.requests.post", return_value=mock_response)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello", "name": "Cline"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_list_content(mocker):
    """POST /v1/chat/completions extracts text if content is a list of dicts."""
    captured = {}
    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        mock = mocker.Mock()
        mock.raise_for_status = mocker.Mock()
        mock.json.return_value = {"success": True, "data": "ok"}
        return mock
    mocker.patch("open_amplify_ai.routers.chat.requests.post", side_effect=fake_post)

    req_body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is "},
                    {"type": "text", "text": "Linux?"}
                ]
            }
        ]
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert captured["payload"]["data"]["messages"][0]["content"] == "What is Linux?"


def test_chat_completions_stream_options(mocker):
    """POST /v1/chat/completions with stream_options and include_usage emits a usage chunk."""
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__ = mocker.Mock(return_value=mock_cm)
    mock_cm.__exit__ = mocker.Mock(return_value=False)
    mock_cm.status_code = 200
    mock_cm.raise_for_status = mocker.Mock()
    mock_cm.iter_lines = mocker.Mock(
        return_value=[b"data: Hello"]
    )
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_cm)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    body = response.text
    # Should contain a chunk with an empty choices list and a usage block
    assert '"choices": []' in body
    assert '"usage":' in body


def test_chat_completions_tool_call_parsing(mocker):
    """POST /v1/chat/completions parses a JSON string command into structured tool_calls."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": '{"command":"list_files","parameters":{"path":"","recursive":true}}',
    }
    mocker.patch("open_amplify_ai.routers.chat.requests.post", return_value=mock_response)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "List files in the dir"}],
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert "tool_calls" in choice["message"]
    assert "content" not in choice["message"] or choice["message"]["content"] is None
    
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "list_files"
    import json
    args = json.loads(tool_call["function"]["arguments"])
    assert args["recursive"] is True


def test_chat_completions_streaming(mocker):
    """POST /v1/chat/completions with stream=True returns text/event-stream SSE."""
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__ = mocker.Mock(return_value=mock_cm)
    mock_cm.__exit__ = mocker.Mock(return_value=False)
    mock_cm.status_code = 200
    mock_cm.raise_for_status = mocker.Mock()
    mock_cm.iter_lines = mocker.Mock(
        return_value=[
            b'data: {"data":"Hello"}',
            b'data: {"data":"{\\"command\\":\\"foo\\",\\"parameters\\":{}}"}',
            b"data: [DONE]"
        ]
    )
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=mock_cm)

    req_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    
    body = response.text
    # Response body must contain SSE data lines and terminator
    assert "data:" in body
    assert "[DONE]" in body
    
    # We should see content parsed out
    assert 'Hello' in body
    
    # We should see tool call parsed out
    assert 'tool_calls' in body
    assert 'foo' in body


def test_chat_completions_custom_params(mocker):
    """POST /v1/chat/completions forwards temperature and max_tokens to Amplify."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        mock = mocker.Mock()
        mock.raise_for_status = mocker.Mock()
        mock.json.return_value = {"success": True, "data": "ok"}
        return mock

    mocker.patch("open_amplify_ai.routers.chat.requests.post", side_effect=fake_post)

    req_body = {
        "model": "claude-3",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    response = client.post("/v1/chat/completions", json=req_body)
    assert response.status_code == 200
    assert captured["payload"]["data"]["temperature"] == 0.2
    assert captured["payload"]["data"]["max_tokens"] == 512
    assert captured["payload"]["data"]["options"]["model"]["id"] == "claude-3"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Assistants
# ---------------------------------------------------------------------------


def test_list_assistants_success(mocker):
    """GET /v1/assistants returns OpenAI assistant list from Amplify /assistant/list."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": [
            {
                "assistantId": "astp/abc123",
                "name": "Test Assistant",
                "instructions": "Be helpful",
                "createdAt": "2024-01-01T00:00:00",
                "dataSources": [],
            }
        ],
    }
    mocker.patch("open_amplify_ai.routers.assistants.requests.get", return_value=mock_response)

    response = client.get("/v1/assistants")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "astp/abc123"
    assert data["data"][0]["object"] == "assistant"
    assert data["has_more"] is False
    assert data["first_id"] == "astp/abc123"
    assert data["last_id"] == "astp/abc123"


def test_list_assistants_empty(mocker):
    """GET /v1/assistants returns an empty list when no assistants exist."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": []}
    mocker.patch("open_amplify_ai.routers.assistants.requests.get", return_value=mock_response)

    response = client.get("/v1/assistants")
    assert response.status_code == 200
    data = response.json()
    assert data["data"] == []
    assert data["first_id"] is None
    assert data["last_id"] is None


def test_create_assistant_success(mocker):
    """POST /v1/assistants returns an OpenAI assistant object from Amplify /assistant/create."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "message": "Assistant created successfully",
        "data": {
            "assistantId": "astp/new123",
            "id": "ast/new456",
            "version": 1,
        },
    }
    mocker.patch("open_amplify_ai.routers.assistants.requests.post", return_value=mock_response)

    req_body = {
        "model": "gpt-4o",
        "name": "My Assistant",
        "instructions": "Be concise.",
    }
    response = client.post("/v1/assistants", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/new123"
    assert data["object"] == "assistant"
    assert data["name"] == "My Assistant"
    assert data["instructions"] == "Be concise."


def test_retrieve_assistant_success(mocker):
    """GET /v1/assistants/{id} finds the matching assistant from the full list."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": [
            {
                "assistantId": "astp/abc123",
                "name": "Test Assistant",
                "instructions": "Be helpful",
                "createdAt": "2024-01-01T00:00:00",
                "dataSources": [],
            },
            {
                "assistantId": "astp/xyz789",
                "name": "Other Assistant",
                "instructions": "Other",
                "createdAt": "2024-01-02T00:00:00",
                "dataSources": [],
            },
        ],
    }
    mocker.patch("open_amplify_ai.routers.assistants.requests.get", return_value=mock_response)

    response = client.get("/v1/assistants/astp/abc123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["name"] == "Test Assistant"
    assert data["object"] == "assistant"


def test_retrieve_assistant_not_found(mocker):
    """GET /v1/assistants/{id} returns 404 when assistant is absent."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "data": []}
    mocker.patch("open_amplify_ai.routers.assistants.requests.get", return_value=mock_response)

    response = client.get("/v1/assistants/astp/nonexistent")
    assert response.status_code == 404


def test_modify_assistant_success(mocker):
    """POST /v1/assistants/{id} sends an upsert to Amplify with the assistantId."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        m = mocker.Mock()
        m.raise_for_status = mocker.Mock()
        m.json.return_value = {"success": True, "data": {"assistantId": "astp/abc123"}}
        return m

    mocker.patch("open_amplify_ai.routers.assistants.requests.post", side_effect=fake_post)

    req_body = {"name": "Updated Name", "instructions": "New instructions"}
    response = client.post("/v1/assistants/astp/abc123", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["name"] == "Updated Name"
    assert captured["payload"]["data"]["assistantId"] == "astp/abc123"


def test_delete_assistant_success(mocker):
    """DELETE /v1/assistants/{id} returns OpenAI DeletedObject from Amplify /assistant/delete."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": True,
        "message": "Assistant deleted successfully.",
    }
    mocker.patch("open_amplify_ai.routers.assistants.requests.post", return_value=mock_response)

    response = client.delete("/v1/assistants/astp/abc123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "astp/abc123"
    assert data["object"] == "assistant.deleted"
    assert data["deleted"] is True


def test_delete_assistant_amplify_failure(mocker):
    """DELETE /v1/assistants/{id} returns deleted=False when Amplify reports failure."""
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {
        "success": False,
        "message": "You are not authorized to delete this assistant.",
    }
    mocker.patch("open_amplify_ai.routers.assistants.requests.post", return_value=mock_response)

    response = client.delete("/v1/assistants/astp/other123")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is False


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------


def test_create_vector_store_success(mocker):
    """POST /v1/vector_stores creates a tag in Amplify and returns a synthetic VectorStore."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "message": "Tags added successfully"}
    mocker.patch("open_amplify_ai.routers.vector_stores.requests.post", return_value=mock_response)

    response = client.post("/v1/vector_stores", json={"name": "my-store"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store"
    assert data["status"] == "completed"
    assert "file_counts" in data
    assert data["file_counts"]["total"] == 0


def test_retrieve_vector_store_success(mocker):
    """GET /v1/vector_stores/{id} returns a VectorStore with file counts."""
    tags_mock = mocker.Mock()
    tags_mock.raise_for_status = mocker.Mock()
    tags_mock.json.return_value = {
        "success": True,
        "data": {"tags": ["my-store", "other-tag"]},
    }

    files_mock = mocker.Mock()
    files_mock.raise_for_status = mocker.Mock()
    files_mock.json.return_value = {
        "success": True,
        "data": {
            "items": [
                {"id": "file1", "totalTokens": 100},
                {"id": "file2", "totalTokens": 200},
            ],
            "pageKey": None,
        },
    }

    mocker.patch("open_amplify_ai.routers.vector_stores.requests.get", return_value=tags_mock)
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=files_mock)

    response = client.get("/v1/vector_stores/my-store")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store"
    assert data["file_counts"]["total"] == 2
    assert data["file_counts"]["completed"] == 2


def test_retrieve_vector_store_not_found(mocker):
    """GET /v1/vector_stores/{id} returns 404 when the backing tag does not exist."""
    tags_mock = mocker.Mock()
    tags_mock.raise_for_status = mocker.Mock()
    tags_mock.json.return_value = {
        "success": True,
        "data": {"tags": ["other-tag"]},
    }
    mocker.patch("open_amplify_ai.routers.vector_stores.requests.get", return_value=tags_mock)

    response = client.get("/v1/vector_stores/nonexistent-store")
    assert response.status_code == 404


def test_delete_vector_store_success(mocker):
    """DELETE /v1/vector_stores/{id} deletes the backing Amplify tag."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json.return_value = {"success": True, "message": "Tag deleted successfully"}
    mocker.patch("open_amplify_ai.routers.vector_stores.requests.post", return_value=mock_response)

    response = client.delete("/v1/vector_stores/my-store")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "my-store"
    assert data["object"] == "vector_store.deleted"
    assert data["deleted"] is True


def test_list_vector_store_files_success(mocker):
    """GET /v1/vector_stores/{id}/files returns files tagged with the store ID."""
    files_mock = mocker.Mock()
    files_mock.raise_for_status = mocker.Mock()
    files_mock.json.return_value = {
        "success": True,
        "data": {
            "items": [
                {"id": "file-a", "totalTokens": 50},
                {"id": "file-b", "totalTokens": 80},
            ],
            "pageKey": None,
        },
    }
    mocker.patch("open_amplify_ai.utils.requests.post", return_value=files_mock)

    response = client.get("/v1/vector_stores/my-store/files")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "vector_store.file"
    assert data["data"][0]["vector_store_id"] == "my-store"
    assert data["data"][0]["status"] == "completed"


def test_add_file_to_vector_store_success(mocker):
    """POST /v1/vector_stores/{id}/files associates a file with the store via set_tags."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        m = mocker.Mock()
        m.raise_for_status = mocker.Mock()
        m.json.return_value = {"success": True, "message": "Tags updated and added to user"}
        return m

    mocker.patch("open_amplify_ai.routers.vector_stores.requests.post", side_effect=fake_post)

    req_body = {"file_id": "user@vu.edu/2024-01-01/abc.json"}
    response = client.post("/v1/vector_stores/my-store/files", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "user@vu.edu/2024-01-01/abc.json"
    assert data["object"] == "vector_store.file"
    assert data["vector_store_id"] == "my-store"
    assert captured["payload"]["data"]["tags"] == ["my-store"]


def test_modify_vector_store_not_implemented():
    """POST /v1/vector_stores/{id} (update) returns 501."""
    response = client.post("/v1/vector_stores/my-store", json={"name": "new-name"})
    assert response.status_code == 501


def test_delete_vector_store_file_not_implemented():
    """DELETE /v1/vector_stores/{id}/files/{file_id} returns 501."""
    response = client.delete("/v1/vector_stores/my-store/files/file-abc")
    assert response.status_code == 501


# ---------------------------------------------------------------------------
# Strict Coverage / Edge Cases
# ---------------------------------------------------------------------------


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
