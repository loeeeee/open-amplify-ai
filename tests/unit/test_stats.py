"""Unit tests for the token usage statistics module (stats.py).

Tests cover:
  - estimate_tokens heuristic
  - extract_prompt_tokens from request body bytes
  - extract_completion_tokens for non-streaming and streaming (SSE) responses
  - write_token_stats CSV persistence (header creation, row append)
  - build_record timestamp and totals
"""
import csv
import json
import os
import tempfile
from datetime import timezone

import pytest

from open_amplify_ai.stats import (
    TokenStatsRecord,
    build_record,
    estimate_tokens,
    extract_completion_tokens,
    extract_prompt_tokens,
    write_token_stats,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty() -> None:
    """Empty string yields zero tokens."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_four_chars() -> None:
    """Four characters equal exactly one token."""
    assert estimate_tokens("abcd") == 1


def test_estimate_tokens_rounding_down() -> None:
    """Token count is rounded down (integer division)."""
    assert estimate_tokens("abc") == 0
    assert estimate_tokens("abcde") == 1


def test_estimate_tokens_longer_text() -> None:
    """Longer text is estimated proportionally."""
    text = "a" * 400
    assert estimate_tokens(text) == 100


# ---------------------------------------------------------------------------
# extract_prompt_tokens
# ---------------------------------------------------------------------------


def test_extract_prompt_tokens_simple() -> None:
    """Counts characters across all message content fields."""
    payload = {
        "messages": [
            {"role": "system", "content": "A" * 40},
            {"role": "user", "content": "B" * 40},
        ]
    }
    body = json.dumps(payload).encode()
    # 80 chars total -> 20 tokens
    assert extract_prompt_tokens(body) == 20


def test_extract_prompt_tokens_list_content() -> None:
    """Handles list-typed content parts (OpenAI multipart format)."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world!"},
                ],
            }
        ]
    }
    body = json.dumps(payload).encode()
    # "Hello world!" = 12 chars -> 3 tokens
    assert extract_prompt_tokens(body) == 3


def test_extract_prompt_tokens_invalid_json() -> None:
    """Returns 0 for unparseable body instead of raising."""
    assert extract_prompt_tokens(b"not json at all") == 0


def test_extract_prompt_tokens_missing_messages() -> None:
    """Returns 0 when the body has no messages key."""
    body = json.dumps({"model": "gpt-4o"}).encode()
    assert extract_prompt_tokens(body) == 0


def test_extract_prompt_tokens_empty_content() -> None:
    """Empty string content contributes zero characters."""
    payload = {"messages": [{"role": "user", "content": ""}]}
    body = json.dumps(payload).encode()
    assert extract_prompt_tokens(body) == 0


# ---------------------------------------------------------------------------
# extract_completion_tokens — non-streaming
# ---------------------------------------------------------------------------


def test_extract_completion_tokens_non_streaming() -> None:
    """Parses message content from a non-streaming JSON response."""
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "A" * 80}}
        ]
    }
    body = json.dumps(response).encode()
    assert extract_completion_tokens(body, is_streaming=False) == 20


def test_extract_completion_tokens_non_streaming_null_content() -> None:
    """Handles None content (tool call responses) without raising."""
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": None}}
        ]
    }
    body = json.dumps(response).encode()
    assert extract_completion_tokens(body, is_streaming=False) == 0


def test_extract_completion_tokens_non_streaming_invalid_json() -> None:
    """Returns 0 for malformed non-streaming response body."""
    assert extract_completion_tokens(b"broken", is_streaming=False) == 0


# ---------------------------------------------------------------------------
# extract_completion_tokens — streaming (SSE)
# ---------------------------------------------------------------------------


def _build_sse_body(delta_contents: list[str]) -> bytes:
    """Build a fake SSE response body from a list of delta content strings."""
    lines = []
    for content in delta_contents:
        chunk = {
            "choices": [{"delta": {"content": content}, "finish_reason": None}]
        }
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return "\n".join(lines).encode()


def test_extract_completion_tokens_streaming() -> None:
    """Accumulates delta content lengths across all SSE chunks."""
    body = _build_sse_body(["A" * 20, "B" * 20, "C" * 20])
    # 60 chars -> 15 tokens
    assert extract_completion_tokens(body, is_streaming=True) == 15


def test_extract_completion_tokens_streaming_skips_done() -> None:
    """[DONE] sentinel line does not contribute tokens."""
    body = _build_sse_body(["Hello"])
    assert extract_completion_tokens(body, is_streaming=True) == 1


def test_extract_completion_tokens_streaming_null_delta_content() -> None:
    """Null or absent delta content in a chunk is treated as zero chars."""
    chunk = {"choices": [{"delta": {"content": None}, "finish_reason": None}]}
    body = f"data: {json.dumps(chunk)}\ndata: [DONE]".encode()
    assert extract_completion_tokens(body, is_streaming=True) == 0


def test_extract_completion_tokens_streaming_empty_body() -> None:
    """Empty streaming response body returns zero tokens."""
    assert extract_completion_tokens(b"", is_streaming=True) == 0


# ---------------------------------------------------------------------------
# build_record
# ---------------------------------------------------------------------------


def test_build_record_totals() -> None:
    """total_tokens equals prompt_tokens + completion_tokens."""
    record = build_record(
        ip_address="1.2.3.4",
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        prompt_tokens=10,
        completion_tokens=5,
    )
    assert record.total_tokens == 15
    assert record.error == ""


def test_build_record_timestamp_utc() -> None:
    """Timestamp is a non-empty ISO 8601 string with UTC offset."""
    record = build_record(
        ip_address="127.0.0.1",
        method="GET",
        path="/v1/models",
        status_code=200,
        prompt_tokens=0,
        completion_tokens=0,
    )
    assert record.timestamp
    assert "+00:00" in record.timestamp or record.timestamp.endswith("Z")


def test_build_record_error_field() -> None:
    """Error field is stored verbatim."""
    record = build_record(
        ip_address="10.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        status_code=500,
        prompt_tokens=0,
        completion_tokens=0,
        error="Connection refused",
    )
    assert record.error == "Connection refused"


# ---------------------------------------------------------------------------
# write_token_stats
# ---------------------------------------------------------------------------


def _make_record(**kwargs) -> TokenStatsRecord:
    """Helper to create a minimal TokenStatsRecord for CSV tests."""
    defaults = dict(
        ip_address="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        prompt_tokens=5,
        completion_tokens=3,
    )
    defaults.update(kwargs)
    return build_record(**defaults)


def test_write_token_stats_creates_file_with_header(tmp_path) -> None:
    """Creates a new CSV file with a header row on first write."""
    csv_path = str(tmp_path / "token_stats.csv")
    record = _make_record()
    write_token_stats(record, csv_path)

    assert os.path.isfile(csv_path)
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert set(reader.fieldnames) == {
            "timestamp",
            "ip_address",
            "method",
            "path",
            "status_code",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "error",
        }
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["ip_address"] == "127.0.0.1"
    assert rows[0]["prompt_tokens"] == "5"
    assert rows[0]["completion_tokens"] == "3"
    assert rows[0]["total_tokens"] == "8"


def test_write_token_stats_appends_rows(tmp_path) -> None:
    """Appends additional rows without duplicating the header."""
    csv_path = str(tmp_path / "token_stats.csv")
    write_token_stats(_make_record(prompt_tokens=10), csv_path)
    write_token_stats(_make_record(prompt_tokens=20), csv_path)
    write_token_stats(_make_record(prompt_tokens=30), csv_path)

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3
    assert rows[0]["prompt_tokens"] == "10"
    assert rows[1]["prompt_tokens"] == "20"
    assert rows[2]["prompt_tokens"] == "30"


def test_write_token_stats_records_error(tmp_path) -> None:
    """Error field is written correctly to the CSV."""
    csv_path = str(tmp_path / "token_stats.csv")
    record = _make_record(error="HTTP 500")
    write_token_stats(record, csv_path)

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["error"] == "HTTP 500"
    assert rows[0]["status_code"] == "200"
