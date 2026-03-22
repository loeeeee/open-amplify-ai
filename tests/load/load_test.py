import argparse
import concurrent.futures
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests
from tqdm import tqdm


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/load_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for a single load test run."""

    base_url: str
    token: str
    concurrency: int
    total_requests: int
    model: str
    prompt: str
    timeout_s: float


@dataclass
class RequestResult:
    """Outcome of a single HTTP request."""

    latency_s: float
    status_code: int
    success: bool
    error: Optional[str]


@dataclass
class LoadTestReport:
    """Aggregated statistics from a completed load test run."""

    total: int
    successful: int
    failed: int
    total_duration_s: float
    throughput_rps: float
    latency_min_s: float
    latency_max_s: float
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float


def parse_args() -> LoadTestConfig:
    """Parse CLI arguments and resolve the bearer token. Exits if token is absent."""
    parser = argparse.ArgumentParser(
        description="Load test the amplify-ai server by sending concurrent chat completion requests."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the running server (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for authentication. Falls back to AMPLIFY_AI_TOKEN env var.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent workers (default: 10)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=50,
        help="Total number of requests to send (default: 50)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model identifier to use for chat requests (default: gpt-4o)",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly one word: hello.",
        help="Prompt text to send in each request (default: short single-word reply prompt)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60.0)",
    )

    args = parser.parse_args()

    token = args.token or os.environ.get("AMPLIFY_AI_TOKEN")
    if not token:
        logger.error(
            "No bearer token provided. Use --token or set AMPLIFY_AI_TOKEN env var."
        )
        sys.exit(1)

    return LoadTestConfig(
        base_url=args.url.rstrip("/"),
        token=token,
        concurrency=args.concurrency,
        total_requests=args.total,
        model=args.model,
        prompt=args.prompt,
        timeout_s=args.timeout,
    )


def send_chat_request(config: LoadTestConfig) -> RequestResult:
    """Send a single non-streaming chat completion request and record the outcome.

    Returns a RequestResult with latency, HTTP status code, and any error text.
    """
    url = f"{config.base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": config.prompt}],
        "temperature": 0.0,
        "stream": False,
    }

    start = time.monotonic()
    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=config.timeout_s
        )
        latency = time.monotonic() - start
        success = response.status_code == 200
        error = None if success else response.text[:200]
        return RequestResult(
            latency_s=latency,
            status_code=response.status_code,
            success=success,
            error=error,
        )
    except requests.RequestException as exc:
        latency = time.monotonic() - start
        return RequestResult(
            latency_s=latency,
            status_code=0,
            success=False,
            error=str(exc),
        )


def compute_stats(results: list[RequestResult], total_duration_s: float) -> LoadTestReport:
    """Compute aggregated statistics from a list of request results.

    Uses statistics.quantiles() for percentile computation (Python 3.13+ stdlib).
    """
    total = len(results)
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    throughput = total / total_duration_s if total_duration_s > 0 else 0.0

    if not successful:
        return LoadTestReport(
            total=total,
            successful=0,
            failed=len(failed),
            total_duration_s=total_duration_s,
            throughput_rps=throughput,
            latency_min_s=0.0,
            latency_max_s=0.0,
            latency_mean_s=0.0,
            latency_p50_s=0.0,
            latency_p95_s=0.0,
            latency_p99_s=0.0,
        )

    latencies = sorted(r.latency_s for r in successful)
    quantiles_100 = statistics.quantiles(latencies, n=100)

    return LoadTestReport(
        total=total,
        successful=len(successful),
        failed=len(failed),
        total_duration_s=total_duration_s,
        throughput_rps=throughput,
        latency_min_s=latencies[0],
        latency_max_s=latencies[-1],
        latency_mean_s=statistics.mean(latencies),
        latency_p50_s=quantiles_100[49],
        latency_p95_s=quantiles_100[94],
        latency_p99_s=quantiles_100[98],
    )


def print_report(report: LoadTestReport) -> None:
    """Print a concise load test report to stdout."""
    separator = "-" * 40
    print()
    print("Load Test Report")
    print("=" * 40)
    print(f"Total requests  : {report.total}")
    print(f"Successful      : {report.successful}")
    print(f"Failed          : {report.failed}")
    print(f"Total duration  : {report.total_duration_s:.2f} s")
    print(f"Throughput      : {report.throughput_rps:.2f} req/s")
    print()
    print("Latency (successful requests)")
    print(separator)
    if report.successful == 0:
        print("No successful requests to report latency for.")
    else:
        print(f"Min             : {report.latency_min_s:.3f} s")
        print(f"Mean            : {report.latency_mean_s:.3f} s")
        print(f"Median (p50)    : {report.latency_p50_s:.3f} s")
        print(f"p95             : {report.latency_p95_s:.3f} s")
        print(f"p99             : {report.latency_p99_s:.3f} s")
        print(f"Max             : {report.latency_max_s:.3f} s")
    print()


def run_load_test(config: LoadTestConfig) -> LoadTestReport:
    """Orchestrate the load test: submit all requests concurrently and collect results.

    Uses ThreadPoolExecutor with tqdm progress tracking. Results are collected as
    futures complete (not in submission order) for accurate wall-clock timing.
    """
    logger.info(
        "Starting load test: url=%s concurrency=%d total=%d model=%s",
        config.base_url,
        config.concurrency,
        config.total_requests,
        config.model,
    )

    results: list[RequestResult] = []
    wall_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [
            executor.submit(send_chat_request, config)
            for _ in range(config.total_requests)
        ]
        with tqdm(
            total=config.total_requests,
            desc="Requests",
            unit="req",
        ) as progress:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                status_label = "ok" if result.success else f"err({result.status_code})"
                progress.set_postfix_str(status_label, refresh=False)
                progress.update(1)

    total_duration = time.monotonic() - wall_start
    report = compute_stats(results, total_duration)

    logger.info(
        "Load test complete: %d/%d successful in %.2fs (%.2f req/s)",
        report.successful,
        report.total,
        report.total_duration_s,
        report.throughput_rps,
    )
    return report


def main() -> None:
    """Entry point: parse configuration, run the load test, and print the report."""
    config = parse_args()
    report = run_load_test(config)
    print_report(report)

    if report.failed > 0:
        logger.warning("%d request(s) failed during the load test.", report.failed)


if __name__ == "__main__":
    main()
