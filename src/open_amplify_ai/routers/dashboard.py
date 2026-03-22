"""Dashboard endpoint for token usage statistics."""
import csv
import html
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from open_amplify_ai.stats import TokenStatsRecord

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dashboard"])


@dataclass
class DashboardStats:
    """Aggregate token usage totals across all recorded requests."""

    total_requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error_count: int


@dataclass
class RateStats:
    """Average request and token rates over a fixed time window."""

    requests_per_second: float
    tokens_per_second: float


@dataclass
class ModelUsageStats:
    """Token usage aggregated for one model id within a time window."""

    model: str
    total_requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class DashboardPageData:
    """All values needed to render the HTML dashboard."""

    last_24_hours: DashboardStats
    last_7_days: DashboardStats
    lifetime: DashboardStats
    by_model_24h: List[ModelUsageStats]
    by_model_7d: List[ModelUsageStats]
    by_model_lifetime: List[ModelUsageStats]
    rates: RateStats
    recent: List[TokenStatsRecord]


def read_csv_records(csv_path: str) -> List[TokenStatsRecord]:
    """Read all rows from the token stats CSV and return them as TokenStatsRecord instances.

    Returns an empty list if the file does not exist or cannot be parsed.
    """
    if not os.path.isfile(csv_path):
        logger.warning("Token stats CSV not found at %s", csv_path)
        return []
    records: List[TokenStatsRecord] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    records.append(
                        TokenStatsRecord(
                            timestamp=row.get("timestamp", ""),
                            ip_address=row.get("ip_address", ""),
                            method=row.get("method", ""),
                            path=row.get("path", ""),
                            status_code=int(row.get("status_code", 0)),
                            prompt_tokens=int(row.get("prompt_tokens", 0)),
                            completion_tokens=int(row.get("completion_tokens", 0)),
                            total_tokens=int(row.get("total_tokens", 0)),
                            error=row.get("error", ""),
                            model=row.get("model", ""),
                        )
                    )
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping malformed CSV row: %s", exc)
    except Exception as exc:
        logger.error("Failed to read token stats CSV at %s: %s", csv_path, exc)
    return records


def aggregate_stats(records: List[TokenStatsRecord]) -> DashboardStats:
    """Sum token counts and count errors across all records."""
    prompt = 0
    completion = 0
    total = 0
    errors = 0
    for record in records:
        prompt += record.prompt_tokens
        completion += record.completion_tokens
        total += record.total_tokens
        if record.error:
            errors += 1
    return DashboardStats(
        total_requests=len(records),
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        error_count=errors,
    )


def aggregate_stats_by_model(records: List[TokenStatsRecord]) -> List[ModelUsageStats]:
    """Group records by model id and sum token counts; sort by total tokens descending."""
    acc: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    for record in records:
        key = record.model or ""
        bucket = acc[key]
        bucket["total_requests"] += 1
        bucket["prompt_tokens"] += record.prompt_tokens
        bucket["completion_tokens"] += record.completion_tokens
        bucket["total_tokens"] += record.total_tokens
    rows = [
        ModelUsageStats(
            model=model,
            total_requests=b["total_requests"],
            prompt_tokens=b["prompt_tokens"],
            completion_tokens=b["completion_tokens"],
            total_tokens=b["total_tokens"],
        )
        for model, b in acc.items()
    ]
    rows.sort(key=lambda r: (-r.total_tokens, r.model))
    return rows


def parse_timestamp_iso(ts: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string to an aware datetime, or return None if invalid."""
    if not ts:
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def records_since(records: List[TokenStatsRecord], cutoff_utc: datetime) -> List[TokenStatsRecord]:
    """Return records whose timestamp is at or after cutoff_utc; skip rows with unparseable times."""
    out: List[TokenStatsRecord] = []
    for record in records:
        dt = parse_timestamp_iso(record.timestamp)
        if dt is None:
            continue
        if dt >= cutoff_utc:
            out.append(record)
    return out


def compute_rate_stats(
    records: List[TokenStatsRecord],
    now_utc: datetime,
    window_seconds: int = 60,
) -> RateStats:
    """Compute average requests/s and total tokens/s over the last window_seconds (UTC)."""
    if window_seconds <= 0:
        return RateStats(0.0, 0.0)
    window_start = now_utc - timedelta(seconds=window_seconds)
    in_window: List[TokenStatsRecord] = []
    for record in records:
        dt = parse_timestamp_iso(record.timestamp)
        if dt is None:
            continue
        if dt >= window_start:
            in_window.append(record)
    n = len(in_window)
    tokens = sum(r.total_tokens for r in in_window)
    w = float(window_seconds)
    return RateStats(requests_per_second=n / w, tokens_per_second=tokens / w)


def _summary_dl(stats: DashboardStats) -> str:
    """Return HTML for a definition list of aggregate stats."""
    return (
        "<dl>\n"
        f"  <dt>Total Requests</dt><dd>{stats.total_requests}</dd>\n"
        f"  <dt>Prompt Tokens</dt><dd>{stats.prompt_tokens}</dd>\n"
        f"  <dt>Completion Tokens</dt><dd>{stats.completion_tokens}</dd>\n"
        f"  <dt>Total Tokens</dt><dd>{stats.total_tokens}</dd>\n"
        f"  <dt>Errors</dt><dd>{stats.error_count}</dd>\n"
        "</dl>"
    )


def _model_label(model: str) -> str:
    """Human-readable model cell; empty ids show as (unknown)."""
    return "(unknown)" if not model else model


def _model_usage_table(rows: List[ModelUsageStats]) -> str:
    """Return an HTML table of per-model usage, or a no-data row."""
    if not rows:
        return '<table><thead><tr><th>Model</th><th>Requests</th><th>Prompt</th><th>Completion</th><th>Total</th></tr></thead><tbody><tr><td colspan="5">No data recorded yet.</td></tr></tbody></table>'
    body = ""
    for row in rows:
        label = html.escape(_model_label(row.model))
        body += (
            f"<tr>"
            f"<td>{label}</td>"
            f"<td>{row.total_requests}</td>"
            f"<td>{row.prompt_tokens}</td>"
            f"<td>{row.completion_tokens}</td>"
            f"<td>{row.total_tokens}</td>"
            f"</tr>\n"
        )
    return (
        "<table>\n"
        "<thead>\n"
        "<tr><th>Model</th><th>Requests</th><th>Prompt</th><th>Completion</th><th>Total</th></tr>\n"
        "</thead>\n"
        "<tbody>\n"
        f"{body}"
        "</tbody>\n"
        "</table>"
    )


def render_html(page: DashboardPageData) -> str:
    """Build an HTML page with usage by period, rates, and a recent-requests table.

    Uses inline styles, optional meta refresh, and a small script so timestamps display in the
    browser's local timezone.
    """
    rows_html = ""
    for record in page.recent:
        error_cell = f'<span style="color:#c00">{html.escape(record.error)}</span>' if record.error else ""
        status_style = 'style="color:#c00"' if record.status_code >= 400 else ""
        ts_attr = html.escape(record.timestamp, quote=True)
        ts_body = html.escape(record.timestamp)
        model_cell = html.escape(_model_label(record.model))
        rows_html += (
            f"<tr>"
            f'<td><time class="dashboard-ts" datetime="{ts_attr}">{ts_body}</time></td>'
            f"<td>{html.escape(record.ip_address)}</td>"
            f"<td>{html.escape(record.method)}</td>"
            f"<td>{html.escape(record.path)}</td>"
            f"<td>{model_cell}</td>"
            f'<td {status_style}>{record.status_code}</td>'
            f"<td>{record.prompt_tokens}</td>"
            f"<td>{record.completion_tokens}</td>"
            f"<td>{record.total_tokens}</td>"
            f"<td>{error_cell}</td>"
            f"</tr>\n"
        )

    if not rows_html:
        rows_html = '<tr><td colspan="10">No data recorded yet.</td></tr>'

    rps = page.rates.requests_per_second
    tps = page.rates.tokens_per_second

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="5">
<title>Amplify AI - Token Usage</title>
<style>
  body {{ font-family: monospace; margin: 2rem; color: #111; background: #fafafa; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 1.5rem; }}
  h2 {{ font-size: 1rem; margin-bottom: 0.5rem; margin-top: 1.5rem; }}
  dl {{ display: grid; grid-template-columns: max-content auto; gap: 0.25rem 1.5rem; margin: 0; }}
  dt {{ font-weight: bold; }}
  dd {{ margin: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; }}
  th {{ background: #eee; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>Amplify AI - Token Usage</h1>

<h2>Last 24 hours</h2>
{_summary_dl(page.last_24_hours)}
<h3>Usage by model (last 24 hours)</h3>
{_model_usage_table(page.by_model_24h)}

<h2>Last 7 days</h2>
{_summary_dl(page.last_7_days)}
<h3>Usage by model (last 7 days)</h3>
{_model_usage_table(page.by_model_7d)}

<h2>Lifetime</h2>
{_summary_dl(page.lifetime)}
<h3>Usage by model (lifetime)</h3>
{_model_usage_table(page.by_model_lifetime)}

<h2>Rates (last 60 seconds, UTC window)</h2>
<dl>
  <dt>Requests per second</dt><dd>{rps:.2f}</dd>
  <dt>Tokens per second</dt><dd>{tps:.2f}</dd>
</dl>

<h2>Recent Requests (last {len(page.recent)})</h2>
<table>
<thead>
<tr>
  <th>Timestamp</th>
  <th>IP</th>
  <th>Method</th>
  <th>Path</th>
  <th>Model</th>
  <th>Status</th>
  <th>Prompt</th>
  <th>Completion</th>
  <th>Total</th>
  <th>Error</th>
</tr>
</thead>
<tbody>
{rows_html}</tbody>
</table>
<script>
document.querySelectorAll("time.dashboard-ts").forEach(function (el) {{
  var v = el.getAttribute("datetime");
  if (v) {{ el.textContent = new Date(v).toLocaleString(); }}
}});
</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Render the token usage dashboard as HTML.

    Reads logs/token_stats.csv (or the path in AMPLIFY_STATS_CSV) and
    displays usage by period, short-window rates, and the most recent 100 requests.
    """
    csv_path = os.getenv("AMPLIFY_STATS_CSV", os.path.join("logs", "token_stats.csv"))
    records = read_csv_records(csv_path)
    now_utc = datetime.now(timezone.utc)
    rec_24 = records_since(records, now_utc - timedelta(hours=24))
    rec_7 = records_since(records, now_utc - timedelta(days=7))
    last_24_hours = aggregate_stats(rec_24)
    last_7_days = aggregate_stats(rec_7)
    lifetime = aggregate_stats(records)
    rates = compute_rate_stats(records, now_utc)
    recent = records[-100:][::-1]
    page = DashboardPageData(
        last_24_hours=last_24_hours,
        last_7_days=last_7_days,
        lifetime=lifetime,
        by_model_24h=aggregate_stats_by_model(rec_24),
        by_model_7d=aggregate_stats_by_model(rec_7),
        by_model_lifetime=aggregate_stats_by_model(records),
        rates=rates,
        recent=recent,
    )
    return render_html(page)
