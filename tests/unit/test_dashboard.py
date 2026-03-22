"""Unit tests for the dashboard router (routers/dashboard.py).

Covers:
  - read_csv_records: missing file, valid CSV, malformed rows, optional model column
  - aggregate_stats: empty list, single record, multiple records with errors
  - aggregate_stats_by_model: grouping and sort order
  - parse_timestamp_iso, records_since, compute_rate_stats
  - render_html: period summaries, usage-by-model tables, rates, table rows, meta refresh
  - GET /: no CSV present, populated CSV, AMPLIFY_STATS_CSV env override
  - aggregate_http_status_counts, GET /usage: empty CSV, window filtering, invalid seconds
"""
import csv
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from open_amplify_ai.routers.dashboard import (
    DashboardPageData,
    DashboardStats,
    HttpStatusCounts,
    ModelUsageStats,
    RateStats,
    aggregate_http_status_counts,
    aggregate_stats,
    aggregate_stats_by_model,
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
        "model",
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
                    "model": record.model,
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
    by_model_24h: list[ModelUsageStats] | None = None,
    by_model_7d: list[ModelUsageStats] | None = None,
    by_model_lifetime: list[ModelUsageStats] | None = None,
) -> DashboardPageData:
    """Build a DashboardPageData with the same stats for each period."""
    r = rates if rates is not None else RateStats(0.0, 0.0)
    rec = recent if recent is not None else []
    m24 = by_model_24h if by_model_24h is not None else []
    m7 = by_model_7d if by_model_7d is not None else []
    mlife = by_model_lifetime if by_model_lifetime is not None else []
    return DashboardPageData(
        last_24_hours=stats,
        last_7_days=stats,
        lifetime=stats,
        by_model_24h=m24,
        by_model_7d=m7,
        by_model_lifetime=mlife,
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


def test_read_csv_records_model_optional(tmp_path: pytest.TempPathFactory) -> None:
    """Rows without a model column get an empty model string."""
    csv_path = str(tmp_path / "stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(
            "timestamp,ip_address,method,path,status_code,"
            "prompt_tokens,completion_tokens,total_tokens,error\n"
        )
        fh.write(
            "2026-01-01T00:00:00+00:00,127.0.0.1,POST,/v1/chat/completions,200,4,2,6,\n"
        )

    result = read_csv_records(csv_path)
    assert len(result) == 1
    assert result[0].model == ""


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
# aggregate_stats_by_model
# ---------------------------------------------------------------------------


def test_aggregate_stats_by_model_groups_and_sorts() -> None:
    """Sums per model and sorts by total tokens descending."""
    records = [
        TokenStatsRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            ip_address="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            error="",
            model="small",
        ),
        TokenStatsRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            ip_address="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            error="",
            model="big",
        ),
        TokenStatsRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            ip_address="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            prompt_tokens=1,
            completion_tokens=0,
            total_tokens=1,
            error="",
            model="",
        ),
    ]
    rows = aggregate_stats_by_model(records)
    assert [r.model for r in rows] == ["big", "small", ""]
    assert rows[0].total_tokens == 150
    assert rows[0].total_requests == 1
    unknown = rows[2]
    assert unknown.prompt_tokens == 1
    assert unknown.total_requests == 1


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
    assert html_out.count("Errors</dt><dd>3</dd>") == 3
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
    record = build_record(
        ip_address="10.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        prompt_tokens=4,
        completion_tokens=2,
        model="gpt-4o",
    )
    page = _page(stats, recent=[record])
    html_out = render_html(page)
    assert "10.0.0.1" in html_out
    assert "POST" in html_out
    assert "/v1/chat/completions" in html_out
    assert "gpt-4o" in html_out
    assert "<th>Model</th>" in html_out
    assert 'class="dashboard-ts"' in html_out


def test_render_html_unknown_model_label() -> None:
    """Empty model id is shown as (unknown) in the recent table."""
    stats = DashboardStats(1, 0, 0, 0, 0)
    record = _make_record()
    page = _page(stats, recent=[record])
    html_out = render_html(page)
    assert "(unknown)" in html_out


def test_render_html_usage_by_model_tables() -> None:
    """Per-model sections render rows from ModelUsageStats."""
    stats = DashboardStats(0, 0, 0, 0, 0)
    m = ModelUsageStats(
        model="alpha",
        total_requests=2,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    page = _page(
        stats,
        by_model_24h=[m],
        by_model_7d=[m],
        by_model_lifetime=[m],
    )
    html_out = render_html(page)
    assert "Usage by model (last 24 hours)" in html_out
    assert "Usage by model (last 7 days)" in html_out
    assert "Usage by model (lifetime)" in html_out
    assert "alpha" in html_out
    assert html_out.count("15") >= 3


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


# ---------------------------------------------------------------------------
# aggregate_http_status_counts
# ---------------------------------------------------------------------------


def test_aggregate_http_status_counts_buckets() -> None:
    """Status codes are classified into 2xx, 3xx, 4xx, 5xx, and other."""
    records = [
        _make_record(status_code=200),
        _make_record(status_code=201),
        _make_record(status_code=302),
        _make_record(status_code=404),
        _make_record(status_code=500),
        _make_record(status_code=0),
    ]
    got = aggregate_http_status_counts(records)
    assert got == HttpStatusCounts(http_2xx=2, http_3xx=1, http_4xx=1, http_5xx=1, http_other=1)


# ---------------------------------------------------------------------------
# GET /usage endpoint
# ---------------------------------------------------------------------------


def test_usage_endpoint_no_csv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """GET /usage returns JSON with zeros when the stats CSV does not exist."""
    monkeypatch.setenv("AMPLIFY_STATS_CSV", str(tmp_path / "missing.csv"))
    response = client.get("/usage")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert data["total_requests"] == 0
    assert data["prompt_tokens"] == 0
    assert data["completion_tokens"] == 0
    assert data["total_tokens"] == 0
    assert data["error_count"] == 0
    assert data["http_2xx"] == 0
    assert data["by_model"] == []


def test_usage_endpoint_window_and_status_buckets(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """GET /usage aggregates only rows in the UTC window and counts HTTP classes."""
    now = datetime.now(timezone.utc)
    recent_ts = (now - timedelta(seconds=30)).isoformat()
    old_ts = (now - timedelta(seconds=400)).isoformat()
    records = [
        TokenStatsRecord(
            timestamp=recent_ts,
            ip_address="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            error="",
            model="m1",
        ),
        TokenStatsRecord(
            timestamp=old_ts,
            ip_address="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            status_code=404,
            prompt_tokens=1,
            completion_tokens=0,
            total_tokens=1,
            error="",
            model="m2",
        ),
    ]
    csv_path = str(tmp_path / "stats.csv")
    _write_csv(csv_path, records)
    monkeypatch.setenv("AMPLIFY_STATS_CSV", csv_path)

    r120 = client.get("/usage", params={"seconds": 120})
    assert r120.status_code == 200
    d120 = r120.json()
    assert d120["window_seconds"] == 120
    assert d120["total_requests"] == 1
    assert d120["prompt_tokens"] == 10
    assert d120["completion_tokens"] == 5
    assert d120["http_2xx"] == 1
    assert d120["http_4xx"] == 0
    assert len(d120["by_model"]) == 1
    assert d120["by_model"][0]["model"] == "m1"

    r600 = client.get("/usage", params={"seconds": 600})
    assert r600.status_code == 200
    d600 = r600.json()
    assert d600["total_requests"] == 2
    assert d600["http_2xx"] == 1
    assert d600["http_4xx"] == 1


def test_usage_endpoint_invalid_seconds() -> None:
    """GET /usage returns 422 when seconds is below the minimum."""
    response = client.get("/usage", params={"seconds": 0})
    assert response.status_code == 422
