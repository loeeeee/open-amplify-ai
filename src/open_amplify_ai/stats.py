"""Token usage statistics module.

Provides helpers to estimate token counts from raw request/response bytes
and to persist per-request stats to a CSV file.
"""
import csv
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
logger = logging.getLogger(__name__)

_csv_lock = threading.Lock()


@dataclass
class TokenStatsRecord:
    """One row of token usage statistics."""

    timestamp: str
    ip_address: str
    method: str
    path: str
    status_code: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str = field(default="")
    model: str = field(default="")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a string using the 4-chars-per-token heuristic."""
    return len(text) // 4


def extract_prompt_tokens(body_bytes: bytes) -> int:
    """Parse a chat completion request body and return estimated prompt token count.

    Sums the character length of all message content fields and divides by 4.
    Returns 0 if the body cannot be parsed or does not contain messages.
    """
    try:
        payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
        messages = payload.get("messages", [])
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
                    elif isinstance(part, str):
                        total_chars += len(part)
        return total_chars // 4
    except Exception:
        return 0


def extract_model(body_bytes: bytes) -> str:
    """Return the top-level model id from a JSON request body, or empty string."""
    try:
        payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
        model = payload.get("model", "")
        return model if isinstance(model, str) else ""
    except Exception:
        return ""


def _completion_tokens_from_json(response_bytes: bytes) -> int:
    """Return estimated completion token count from a non-streaming JSON response."""
    try:
        payload = json.loads(response_bytes.decode("utf-8", errors="replace"))
        total_chars = 0
        for choice in payload.get("choices", []):
            content = (choice.get("message") or {}).get("content") or ""
            total_chars += len(content)
        return total_chars // 4
    except Exception:
        return 0


def _completion_tokens_from_sse(response_bytes: bytes) -> int:
    """Return estimated completion token count from a streaming SSE response body."""
    total_chars = 0
    try:
        text = response_bytes.decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str == "[DONE]":
                continue
            try:
                chunk = json.loads(payload_str)
                for choice in chunk.get("choices", []):
                    content = (choice.get("delta") or {}).get("content") or ""
                    total_chars += len(content)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return total_chars // 4


def extract_completion_tokens(response_bytes: bytes, is_streaming: bool) -> int:
    """Return estimated completion token count from the buffered response body.

    Uses SSE line parsing for streaming responses and JSON parsing for
    non-streaming responses.
    """
    if is_streaming:
        return _completion_tokens_from_sse(response_bytes)
    return _completion_tokens_from_json(response_bytes)


def build_record(
    *,
    ip_address: str,
    method: str,
    path: str,
    status_code: int,
    prompt_tokens: int,
    completion_tokens: int,
    error: str = "",
    model: str = "",
) -> TokenStatsRecord:
    """Construct a TokenStatsRecord with the current UTC timestamp."""
    return TokenStatsRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        ip_address=ip_address,
        method=method,
        path=path,
        status_code=status_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        error=error,
        model=model,
    )


def _csv_header_needs_model_migration(csv_path: str) -> bool:
    """Return True if the file exists and its header row lacks a model column."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            line = fh.readline()
        if not line.strip():
            return False
        header = next(csv.reader([line]))
    except OSError:
        return False
    return "model" not in header


def _rewrite_csv_with_model_column(csv_path: str, fieldnames: list[str]) -> None:
    """Rewrite csv_path so the header includes model; existing rows get model empty."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)


def write_token_stats(record: TokenStatsRecord, csv_path: str) -> None:
    """Append a single stats row to the CSV file at csv_path.

    Creates the file and writes a header row if the file does not yet exist.
    Uses a threading lock so concurrent async tasks do not corrupt the file.
    """
    fieldnames = [
        "timestamp",
        "ip_address",
        "method",
        "path",
        "status_code",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "error",
        "model",
    ]
    with _csv_lock:
        try:
            existed_before = os.path.isfile(csv_path)
            if existed_before and _csv_header_needs_model_migration(csv_path):
                _rewrite_csv_with_model_column(csv_path, fieldnames)
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                if not existed_before:
                    writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": record.timestamp,
                        "ip_address": record.ip_address,
                        "method": record.method,
                        "path": record.path,
                        "status_code": record.status_code,
                        "prompt_tokens": record.prompt_tokens,
                        "completion_tokens": record.completion_tokens,
                        "total_tokens": record.total_tokens,
                        "error": record.error,
                        "model": record.model,
                    }
                )
        except Exception as exc:
            logger.error("Failed to write token stats to %s: %s", csv_path, exc)
