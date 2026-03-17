"""Unit tests for the dashboard router (routers/dashboard.py).

Covers:
  - read_csv_records: missing file, valid CSV, malformed rows
  - aggregate_stats: empty list, single record, multiple records with errors
  - render_html: presence of summary values and table rows in output
  - GET /: no CSV present, populated CSV, AMPLIFY_STATS_CSV env override
"""
import csv
import os

import pytest
from fastapi.testclient import TestClient

from open_amplify_ai.routers.dashboard import (
    DashboardStats,
    aggregate_stats,
    read_csv_records,
    render_html,
)
from open_amplify_ai.server import app
from open_amplify_ai.stats import TokenStatsRecord, build_record

os.environ["AMPLIFY_AI_TOKEN"] = "test-token-123"

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: str, records: list[TokenStatsRecord]) -> None:
    """Write a list of TokenStatsRecord instances to a CSV file at path."""
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
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
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
                }
            )


def _make_record(**kwargs) -> TokenStatsRecord:
    """Return a minimal TokenStatsRecord via build_record with sensible defaults."""
    defaults = dict(
        ip_address="127.0.0.1",
        method="GET",
        path="/v1/models",
        status_code=200,
        prompt_tokens=0,
        completion_tokens=0,
    )
    defaults.update(kwargs)
    return build_record(**defaults)


# ---------------------------------------------------------------------------
# read_csv_records
# ---------------------------------------------------------------------------


def test_read_csv_records_missing_file(tmp_path: pytest.TempPathFactory) -> None:
    """Returns empty list when the CSV file does not exist."""
    result = read_csv_records(str(tmp_path / "nonexistent.csv"))
    assert result == []


def test_read_csv_records_valid_file(tmp_path: pytest.TempPathFactory) -> None:
    """Parses all rows from a valid CSV into TokenStatsRecord instances."""
    records = [
        _make_record(prompt_tokens=10, completion_tokens=5),
        _make_record(prompt_tokens=20, completion_tokens=15, method="POST", path="/v1/chat/completions"),
    ]
    csv_path = str(tmp_path / "stats.csv")
    _write_csv(csv_path, records)

    result = read_csv_records(csv_path)

    assert len(result) == 2
    assert result[0].prompt_tokens == 10
    assert result[0].completion_tokens == 5
    assert result[1].prompt_tokens == 20
    assert result[1].path == "/v1/chat/completions"


def test_read_csv_records_skips_malformed_rows(tmp_path: pytest.TempPathFactory) -> None:
    """Skips rows with non-integer token fields and returns the valid ones."""
    csv_path = str(tmp_path / "stats.csv")
    # Write a header-only file then append one broken row and one valid row
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(
            "timestamp,ip_address,method,path,status_code,"
            "prompt_tokens,completion_tokens,total_tokens,error\n"
        )
        fh.write("2026-01-01T00:00:00+00:00,127.0.0.1,GET,/v1/models,not-a-int,bad,bad,bad,\n")
        fh.write("2026-01-01T00:01:00+00:00,127.0.0.1,GET,/v1/models,200,4,2,6,\n")

    result = read_csv_records(csv_path)
    assert len(result) == 1
    assert result[0].prompt_tokens == 4


def test_read_csv_records_empty_file(tmp_path: pytest.TempPathFactory) -> None:
    """Returns empty list for a CSV that only contains a header row."""
    csv_path = str(tmp_path / "stats.csv")
    _write_csv(csv_path, [])

    result = read_csv_records(csv_path)
    assert result == []


# ---------------------------------------------------------------------------
# aggregate_stats
# ---------------------------------------------------------------------------


def test_aggregate_stats_empty() -> None:
    """Returns all-zero DashboardStats for an empty record list."""
    stats = aggregate_stats([])
    assert stats.total_requests == 0
    assert stats.prompt_tokens == 0
    assert stats.completion_tokens == 0
    assert stats.total_tokens == 0
    assert stats.error_count == 0


def test_aggregate_stats_single_record() -> None:
    """Sums token fields from a single record correctly."""
    record = _make_record(prompt_tokens=10, completion_tokens=5)
    stats = aggregate_stats([record])
    assert stats.total_requests == 1
    assert stats.prompt_tokens == 10
    assert stats.completion_tokens == 5
    assert stats.total_tokens == 15
    assert stats.error_count == 0


def test_aggregate_stats_multiple_records() -> None:
    """Accumulates totals across multiple records."""
    records = [
        _make_record(prompt_tokens=10, completion_tokens=5),
        _make_record(prompt_tokens=20, completion_tokens=10),
        _make_record(prompt_tokens=0, completion_tokens=0),
    ]
    stats = aggregate_stats(records)
    assert stats.total_requests == 3
    assert stats.prompt_tokens == 30
    assert stats.completion_tokens == 15
    assert stats.total_tokens == 45


def test_aggregate_stats_counts_errors() -> None:
    """Records with a non-empty error field increment error_count."""
    records = [
        _make_record(error="HTTP 500"),
        _make_record(error=""),
        _make_record(error="Connection refused"),
    ]
    stats = aggregate_stats(records)
    assert stats.error_count == 2


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------


def test_render_html_contains_summary_values() -> None:
    """Summary section shows the DashboardStats values."""
    stats = DashboardStats(
        total_requests=42,
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        error_count=3,
    )
    html = render_html(stats, [])
    assert "42" in html
    assert "100" in html
    assert "200" in html
    assert "300" in html
    assert "3" in html


def test_render_html_no_data_placeholder() -> None:
    """Shows a placeholder message when the recent list is empty."""
    stats = DashboardStats(0, 0, 0, 0, 0)
    html = render_html(stats, [])
    assert "No data recorded yet" in html


def test_render_html_shows_request_rows() -> None:
    """Each record in the recent list produces a table row with its fields."""
    stats = DashboardStats(1, 4, 2, 6, 0)
    record = _make_record(
        ip_address="10.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        prompt_tokens=4,
        completion_tokens=2,
    )
    html = render_html(stats, [record])
    assert "10.0.0.1" in html
    assert "POST" in html
    assert "/v1/chat/completions" in html


def test_render_html_highlights_error_status() -> None:
    """Rows with status >= 400 include a color style for the status cell."""
    stats = DashboardStats(1, 0, 0, 0, 1)
    record = _make_record(status_code=404, error="HTTP 404")
    html = render_html(stats, [record])
    assert "404" in html
    assert "HTTP 404" in html
    assert "#c00" in html


# ---------------------------------------------------------------------------
# GET / endpoint
# ---------------------------------------------------------------------------


def test_dashboard_endpoint_no_csv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """GET / returns 200 HTML even when the stats CSV does not exist."""
    monkeypatch.setenv("AMPLIFY_STATS_CSV", str(tmp_path / "missing.csv"))
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Amplify AI" in response.text
    assert "No data recorded yet" in response.text


def test_dashboard_endpoint_with_data(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """GET / returns 200 HTML with token totals from the CSV."""
    records = [
        _make_record(
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            prompt_tokens=40,
            completion_tokens=20,
        ),
    ]
    csv_path = str(tmp_path / "stats.csv")
    _write_csv(csv_path, records)
    monkeypatch.setenv("AMPLIFY_STATS_CSV", csv_path)

    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "40" in html
    assert "20" in html
    assert "/v1/chat/completions" in html


def test_dashboard_endpoint_respects_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """GET / uses AMPLIFY_STATS_CSV when set rather than the default path."""
    records = [_make_record(prompt_tokens=99)]
    csv_path = str(tmp_path / "custom_stats.csv")
    _write_csv(csv_path, records)
    monkeypatch.setenv("AMPLIFY_STATS_CSV", csv_path)

    response = client.get("/")
    assert response.status_code == 200
    assert "99" in response.text
