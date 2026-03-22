"""Unit tests for the dashboard router (routers/dashboard.py).

Covers:
  - read_csv_records: missing file, valid CSV, malformed rows
  - aggregate_stats: empty list, single record, multiple records with errors
  - parse_timestamp_iso, records_since, compute_rate_stats
  - render_html: period summaries, rates, table rows, meta refresh
  - GET /: no CSV present, populated CSV, AMPLIFY_STATS_CSV env override
"""
import csv
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from open_amplify_ai.routers.dashboard import (
    DashboardPageData,
    DashboardStats,
    RateStats,
    aggregate_stats,
    compute_rate_stats,
    parse_timestamp_iso,
    read_csv_records,
    records_since,
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


def _page(
    stats: DashboardStats,
    *,
    recent: list[TokenStatsRecord] | None = None,
    rates: RateStats | None = None,
) -> DashboardPageData:
    """Build a DashboardPageData with the same stats for each period."""
    r = rates if rates is not None else RateStats(0.0, 0.0)
    rec = recent if recent is not None else []
    return DashboardPageData(
        last_24_hours=stats,
        last_7_days=stats,
        lifetime=stats,
        rates=r,
        recent=rec,
    )


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
# parse_timestamp_iso, records_since, compute_rate_stats
# ---------------------------------------------------------------------------


def test_parse_timestamp_iso_z_suffix() -> None:
    """Parses timestamps ending with Z as UTC."""
    dt = parse_timestamp_iso("2026-01-15T10:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.year == 2026 and dt.month == 1 and dt.day == 15


def test_parse_timestamp_iso_offset() -> None:
    """Parses ISO strings with explicit offset."""
    dt = parse_timestamp_iso("2026-01-15T10:00:00+00:00")
    assert dt is not None
    assert dt.utcoffset() is not None


def test_parse_timestamp_iso_invalid() -> None:
    """Returns None for invalid input."""
    assert parse_timestamp_iso("") is None
    assert parse_timestamp_iso("not-a-date") is None


def test_records_since_filters_by_cutoff() -> None:
    """Keeps records on or after the cutoff; drops unparseable timestamps."""
    r1 = TokenStatsRecord(
        timestamp="2026-01-01T08:00:00+00:00",
        ip_address="127.0.0.1",
        method="GET",
        path="/a",
        status_code=200,
        prompt_tokens=1,
        completion_tokens=0,
        total_tokens=1,
        error="",
    )
    r2 = TokenStatsRecord(
        timestamp="2026-01-01T12:00:00+00:00",
        ip_address="127.0.0.1",
        method="GET",
        path="/b",
        status_code=200,
        prompt_tokens=2,
        completion_tokens=0,
        total_tokens=2,
        error="",
    )
    bad = TokenStatsRecord(
        timestamp="broken",
        ip_address="127.0.0.1",
        method="GET",
        path="/c",
        status_code=200,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        error="",
    )
    cutoff = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    out = records_since([r1, r2, bad], cutoff)
    assert len(out) == 1
    assert out[0].path == "/b"


def test_compute_rate_stats_window() -> None:
    """Averages requests and tokens over the window using UTC bounds."""
    now = datetime(2026, 6, 1, 15, 0, 0, tzinfo=timezone.utc)
    t_in = (now - timedelta(seconds=30)).isoformat()
    t_old = (now - timedelta(seconds=120)).isoformat()
    rec_in = TokenStatsRecord(
        timestamp=t_in,
        ip_address="127.0.0.1",
        method="GET",
        path="/x",
        status_code=200,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        error="",
    )
    rec_old = TokenStatsRecord(
        timestamp=t_old,
        ip_address="127.0.0.1",
        method="GET",
        path="/y",
        status_code=200,
        prompt_tokens=100,
        completion_tokens=100,
        total_tokens=200,
        error="",
    )
    rates = compute_rate_stats([rec_in, rec_old], now, window_seconds=60)
    assert rates.requests_per_second == pytest.approx(1.0 / 60.0)
    assert rates.tokens_per_second == pytest.approx(20.0 / 60.0)


def test_compute_rate_stats_zero_window() -> None:
    """Non-positive window returns zero rates."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r = _make_record()
    rates = compute_rate_stats([r], now, window_seconds=0)
    assert rates.requests_per_second == 0.0
    assert rates.tokens_per_second == 0.0


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------


def test_render_html_contains_summary_values() -> None:
    """Summary sections show the DashboardStats values."""
    stats = DashboardStats(
        total_requests=42,
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        error_count=3,
    )
    page = _page(stats, rates=RateStats(1.25, 99.5))
    html_out = render_html(page)
    assert html_out.count("42") >= 3
    assert html_out.count("100") >= 3
    assert html_out.count("200") >= 3
    assert html_out.count("300") >= 3
    assert html_out.count("3") >= 3
    assert "1.25" in html_out
    assert "99.50" in html_out
    assert 'http-equiv="refresh" content="5"' in html_out
    assert "Last 24 hours" in html_out
    assert "Last 7 days" in html_out
    assert "Lifetime" in html_out


def test_render_html_no_data_placeholder() -> None:
    """Shows a placeholder message when the recent list is empty."""
    stats = DashboardStats(0, 0, 0, 0, 0)
    page = _page(stats)
    html_out = render_html(page)
    assert "No data recorded yet" in html_out


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
    page = _page(stats, recent=[record])
    html_out = render_html(page)
    assert "10.0.0.1" in html_out
    assert "POST" in html_out
    assert "/v1/chat/completions" in html_out
    assert 'class="dashboard-ts"' in html_out


def test_render_html_highlights_error_status() -> None:
    """Rows with status >= 400 include a color style for the status cell."""
    stats = DashboardStats(1, 0, 0, 0, 1)
    record = _make_record(status_code=404, error="HTTP 404")
    page = _page(stats, recent=[record])
    html_out = render_html(page)
    assert "404" in html_out
    assert "HTTP 404" in html_out
    assert "#c00" in html_out


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
    assert 'http-equiv="refresh" content="5"' in response.text


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
    assert "Last 24 hours" in html


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
