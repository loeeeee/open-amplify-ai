"""Unit tests for chat endpoint streaming protocol correctness.

Tests detailed streaming behavior beyond basic functionality.
Covers the streaming protocol correctness from the test refactor plan.
"""
import json
import pytest
from fastapi.testclient import TestClient
from open_amplify_ai.server import app
import os

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


def _make_streaming_client(mocker, lines):
    """Build an async httpx streaming client mock."""
    async def fake_aiter_lines():
        for line in lines:
            yield line
    
    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = mocker.Mock()
    mock_resp.aiter_lines = fake_aiter_lines
    
    mock_stream_cm = mocker.AsyncMock()
    mock_stream_cm.__aenter__.return_value = mock_resp
    
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = mocker.Mock(return_value=mock_stream_cm)
    return mock_client


def test_streaming_first_delta_contains_role(mocker):
    """Test that the first delta in streaming contains role='assistant'."""
    lines = ['data: {"data":"Hello"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    assert response.status_code == 200
    body = response.text
    chunks = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    
    # First chunk should contain role
    assert len(chunks) > 0
    first_delta = chunks[0]["choices"][0]["delta"]
    assert "role" in first_delta
    assert first_delta["role"] == "assistant"


def test_streaming_subsequent_chunks_omit_role(mocker):
    """Test that subsequent chunks omit role unless needed."""
    lines = [
        'data: {"data":"Hello"}',
        'data: {"data":" world"}',
        'data: {"data":"!"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    chunks = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    
    # First chunk has role
    assert "role" in chunks[0]["choices"][0]["delta"]
    
    # Subsequent content chunks should not have role
    for chunk in chunks[1:]:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            assert "role" not in delta or delta.get("role") is None


def test_streaming_content_deltas_preserve_order(mocker):
    """Test that content deltas preserve order across chunks."""
    lines = [
        'data: {"data":"First"}',
        'data: {"data":" second"}',
        'data: {"data":" third"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    content_parts = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                content_parts.append(delta["content"])
    
    # Content should appear in order
    full_content = "".join(content_parts)
    assert full_content == "First second third"


def test_streaming_final_chunk_carries_finish_reason(mocker):
    """Test that the final chunk contains finish_reason."""
    lines = ['data: {"data":"Done"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    chunks = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    
    # At least one chunk should have finish_reason
    has_finish_reason = False
    for chunk in chunks:
        choice = chunk["choices"][0]
        if "finish_reason" in choice and choice["finish_reason"] is not None:
            has_finish_reason = True
            assert choice["finish_reason"] in ["stop", "length", "tool_calls", "content_filter"]
    
    assert has_finish_reason, "No chunk contained finish_reason"


def test_streaming_usage_chunk_only_when_requested(mocker):
    """Test that usage chunk appears only when stream_options.include_usage=True."""
    lines = ['data: {"data":"Hello"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    # Without stream_options
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    has_usage = False
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            if "usage" in chunk:
                has_usage = True
    
    assert not has_usage, "Usage chunk appeared without stream_options.include_usage"
    
    # With stream_options.include_usage=True
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    
    body = response.text
    has_usage = False
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            if "usage" in chunk:
                has_usage = True
                # Usage chunk should have empty choices
                assert chunk["choices"] == []
    
    assert has_usage, "Usage chunk did not appear with stream_options.include_usage=True"


def test_streaming_usage_chunk_before_done(mocker):
    """Test that usage chunk comes before [DONE] marker."""
    lines = ['data: {"data":"Hello"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    
    body = response.text
    lines_list = body.strip().split("\n")
    
    usage_index = -1
    done_index = -1
    
    for i, line in enumerate(lines_list):
        if line.startswith("data: "):
            if line == "data: [DONE]":
                done_index = i
            else:
                chunk = json.loads(line[6:])
                if "usage" in chunk:
                    usage_index = i
    
    if usage_index != -1 and done_index != -1:
        assert usage_index < done_index, "Usage chunk must come before [DONE]"


def test_streaming_empty_delta_chunks_tolerated(mocker):
    """Test that empty delta chunks are handled gracefully."""
    lines = [
        'data: {"data":""}',  # Empty delta
        'data: {"data":"Hello"}',
        'data: {"data":""}',  # Another empty delta
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    assert response.status_code == 200
    body = response.text
    # Should handle empty deltas without errors
    assert "[DONE]" in body


def test_streaming_multiline_content_preserved(mocker):
    """Test that multiline content with newlines is preserved."""
    lines = [
        'data: {"data":"Line 1\\n"}',
        'data: {"data":"Line 2\\n"}',
        'data: {"data":"Line 3"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    content_parts = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                content_parts.append(delta["content"])
    
    full_content = "".join(content_parts)
    # Newlines should be preserved
    assert "\n" in full_content


def test_streaming_zero_generated_tokens(mocker):
    """Test streaming with response that generates zero tokens (empty response)."""
    lines = ["data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    assert response.status_code == 200
    body = response.text
    # Should handle gracefully even with zero content
    assert "[DONE]" in body


def test_streaming_tool_call_chunks_accumulate(mocker):
    """Test that tool call chunks accumulate arguments over time."""
    # Simulate tool call being sent in parts
    tool_json_part1 = '{"tool":"read_file"'
    tool_json_part2 = ',"parameters":{"path":"/tmp/test.txt"}}'
    
    lines = [
        f'data: {{"data":"{tool_json_part1}"}}',
        f'data: {{"data":"{tool_json_part2}"}}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Read file"}],
            "stream": True,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }],
        },
    )
    
    assert response.status_code == 200
    body = response.text
    # Should eventually parse complete tool call
    assert "tool_calls" in body
    assert "read_file" in body


def test_streaming_chunk_ids_stay_constant(mocker):
    """Test that chunk IDs remain constant across one stream."""
    lines = [
        'data: {"data":"Hello"}',
        'data: {"data":" world"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    chunk_ids = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            if "id" in chunk:
                chunk_ids.append(chunk["id"])
    
    # All chunks should have the same ID
    if len(chunk_ids) > 1:
        assert all(chunk_id == chunk_ids[0] for chunk_id in chunk_ids), "Chunk IDs must be constant"


def test_streaming_choice_index_stable(mocker):
    """Test that choice index remains stable across chunks."""
    lines = [
        'data: {"data":"Hello"}',
        'data: {"data":" world"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    indices = []
    
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            if "choices" in chunk and len(chunk["choices"]) > 0:
                indices.append(chunk["choices"][0]["index"])
    
    # All choices should have index 0 (for single choice response)
    assert all(idx == 0 for idx in indices), "Choice index must be stable"


def test_streaming_exactly_one_done_marker(mocker):
    """Test that stream contains exactly one [DONE] marker."""
    lines = [
        'data: {"data":"Hello"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    done_count = body.count("data: [DONE]")
    
    assert done_count == 1, f"Expected exactly 1 [DONE] marker, got {done_count}"


def test_streaming_no_data_after_done(mocker):
    """Test that no data appears after [DONE] marker."""
    lines = [
        'data: {"data":"Hello"}',
        "data: [DONE]",
    ]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    lines_list = body.strip().split("\n")
    
    done_index = -1
    for i, line in enumerate(lines_list):
        if line == "data: [DONE]":
            done_index = i
            break
    
    if done_index != -1:
        # Check no data lines after [DONE]
        for i in range(done_index + 1, len(lines_list)):
            line = lines_list[i]
            if line.strip():  # Non-empty line
                assert not line.startswith("data: "), f"Found data after [DONE]: {line}"


def test_streaming_object_type_is_chunk(mocker):
    """Test that streaming chunks have object='chat.completion.chunk'."""
    lines = ['data: {"data":"Hello"}', "data: [DONE]"]
    mocker.patch(
        "open_amplify_ai.utils.httpx.AsyncClient",
        return_value=_make_streaming_client(mocker, lines),
    )
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    
    body = response.text
    for line in body.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            assert chunk["object"] == "chat.completion.chunk"
