"""Dashboard endpoint for token usage statistics."""
import csv
import logging
import os
from dataclasses import dataclass
from typing import List

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


def render_html(stats: DashboardStats, recent: List[TokenStatsRecord]) -> str:
    """Build a plain HTML page showing aggregate stats and a recent-requests table.

    The page uses only inline styles, no external resources, and no JavaScript.
    """
    rows_html = ""
    for record in recent:
        error_cell = f'<span style="color:#c00">{record.error}</span>' if record.error else ""
        status_style = 'style="color:#c00"' if record.status_code >= 400 else ""
        rows_html += (
            f"<tr>"
            f"<td>{record.timestamp}</td>"
            f"<td>{record.ip_address}</td>"
            f"<td>{record.method}</td>"
            f"<td>{record.path}</td>"
            f'<td {status_style}>{record.status_code}</td>'
            f"<td>{record.prompt_tokens}</td>"
            f"<td>{record.completion_tokens}</td>"
            f"<td>{record.total_tokens}</td>"
            f"<td>{error_cell}</td>"
            f"</tr>\n"
        )

    if not rows_html:
        rows_html = '<tr><td colspan="9">No data recorded yet.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
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

<h2>Summary</h2>
<dl>
  <dt>Total Requests</dt><dd>{stats.total_requests}</dd>
  <dt>Prompt Tokens</dt><dd>{stats.prompt_tokens}</dd>
  <dt>Completion Tokens</dt><dd>{stats.completion_tokens}</dd>
  <dt>Total Tokens</dt><dd>{stats.total_tokens}</dd>
  <dt>Errors</dt><dd>{stats.error_count}</dd>
</dl>

<h2>Recent Requests (last {len(recent)})</h2>
<table>
<thead>
<tr>
  <th>Timestamp</th>
  <th>IP</th>
  <th>Method</th>
  <th>Path</th>
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
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Render the token usage dashboard as a plain HTML page.

    Reads logs/token_stats.csv (or the path in AMPLIFY_STATS_CSV) and
    displays aggregate totals and the most recent 100 requests.
    """
    csv_path = os.getenv("AMPLIFY_STATS_CSV", os.path.join("logs", "token_stats.csv"))
    records = read_csv_records(csv_path)
    stats = aggregate_stats(records)
    recent = records[-100:][::-1]
    return render_html(stats, recent)
