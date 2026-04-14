# Amplify AI Compatibility Layer

![Build](https://github.com/loeeeee/amplify-ai/actions/workflows/build.yml/badge.svg)

An OpenAI-compatible HTTP layer in front of the Vanderbilt Amplify AI API, designed primarily for internal developers running local AI tools (cline, openclaw, kilo, etc.) and NixOS deployments.

External users are welcome, but this README assumes familiarity with NixOS and Amplify AI. For a concise, probed API reference, see the locally generated `docs/amplify_api_probed.md`.

## Quick Start (Local Server)

- **Requirements**
  - Python 3.13+
  - Nix with `nix-shell`
  - `.env` file with:
    - `AMPLIFY_AI_TOKEN` — API key (e.g., `amp-v1-...`)
    - `AMPLIFY_AI_EMAIL` — Vanderbilt email (used in upstream requests)

- **Start the server**

```bash
nix-shell
amplify server
```

By default the server binds to `0.0.0.0:8080`. You can override the port via CLI or environment:

```bash
amplify server --port 9090
```

or set:

- `AMPLIFY_SERVER_HOST` (default `0.0.0.0`)
- `AMPLIFY_SERVER_PORT` (default `8080`)

Enable verbose logging for request/response debugging with:

- CLI flag: `amplify server --debug`
- or environment: `AMPLIFY_DEBUG=1`

## Dashboard

A token usage dashboard is available at the root endpoint:

- `GET /` — plain HTML page sourced from `logs/token_stats.csv` (or `AMPLIFY_STATS_CSV`). It shows the same aggregate totals (requests, prompt, completion, and total tokens, plus errors) for the **last 24 hours**, **last 7 days**, and **lifetime**; **usage by model** (requests and token sums per requested model id) for each of those periods; **average requests per second** and **tokens per second** over the **last 60 seconds** (UTC window); and a table of the **100 most recent** requests including a **Model** column (requested model id from the JSON body; shown as `(unknown)` when empty). Timestamps in the table are shown in the **browser’s local timezone** (via a short inline script). The page **auto-refreshes every 5 seconds** (`meta refresh`).

- `GET /usage` — JSON summary of the same CSV stats for a configurable **UTC lookback window**. Query parameter **`seconds`** (integer, default `300`, minimum `1`, maximum 90 days) selects how far back to aggregate. The response includes `window_seconds`, `generated_at_utc`, `cutoff_utc`, token totals (`prompt_tokens`, `completion_tokens`, `total_tokens`), `total_requests`, `error_count`, `requests_per_second`, `tokens_per_second` (rates over that window), HTTP status bucket counts (`http_2xx`, `http_3xx`, `http_4xx`, `http_5xx`, `http_other`), and `by_model` (per-model aggregates). Use this endpoint for monitoring tools such as [Gatus](https://gatus.io/) (e.g. conditions on `[BODY].error_count` or `[BODY].total_requests`).

The CSV path can be overridden via the `AMPLIFY_STATS_CSV` environment variable.

## Concurrency Model

All route handlers are `async def` and use `httpx.AsyncClient` (non-blocking) for every upstream call. This allows Uvicorn's event loop to serve multiple in-flight requests concurrently even when upstream LLM calls are slow. The standalone API prober (`probe_api.py`) is exempt and remains synchronous.

## OpenAI-Compatible API

The FastAPI server exposes a subset of the OpenAI API under `/v1/*`, backed by Amplify AI (`https://prod-api.vanderbilt.ai`). It is compatible with AI coding tools that expect OpenAI’s `chat.completions` endpoint, including streaming and tool-call output.

### Models

- `GET /v1/models` — list available models
- `GET /v1/models/{model}` — retrieve a model by ID
- `DELETE /v1/models/{model}` — always returns `405` (Amplify does not support model deletion)

### Chat Completions

- `POST /v1/chat/completions`
  - Supports non-streaming and streaming responses (SSE `data:` lines)
  - Compatible with cline, openclaw, kilo, and similar tools
  - Refactored implementation available in `chat_refactored.py` with improved:
    - Strict request validation and explicit error handling
    - Deterministic tool calling with strong anchoring
    - Streaming state machine for clean mode transitions
    - Precise HTTP status code mapping
    - See `docs-vibe/72-chat-endpoint-refactor-complete.md` for details

### Files

- `GET /v1/files` — list uploaded files
- `POST /v1/files` — upload a file (Amplify pre-signed URL + S3 `PUT` under the hood)
- `GET /v1/files/{file_id}` — retrieve a file record
- `DELETE /v1/files/{file_id}` — delete a file
- `GET /v1/files/{file_id}/content` — download file contents (Code Interpreter files only)

### Assistants

- `GET /v1/assistants` — list assistants
- `POST /v1/assistants` — create an assistant
- `GET /v1/assistants/{assistant_id}` — retrieve an assistant
- `POST /v1/assistants/{assistant_id}` — modify an assistant
- `DELETE /v1/assistants/{assistant_id}` — delete an assistant

### Threads

- `DELETE /v1/threads/{thread_id}` — delete a thread
- All other thread, message, run, and run-step endpoints currently return `501 Not Implemented`

### Vector Stores

- `POST /v1/vector_stores` — create a virtual store (backed by Amplify tags)
- `GET /v1/vector_stores/{id}` — retrieve a vector store
- `DELETE /v1/vector_stores/{id}` — delete a vector store (removes backing tag only)
- `GET /v1/vector_stores/{id}/files` — list files in a store
- `POST /v1/vector_stores/{id}/files` — add a file to a store
- All other vector store batch endpoints currently return `501 Not Implemented`

### Unsupported (501)

The following OpenAI-style features are not implemented and return `501 Not Implemented` with a clear message:

- Embeddings
- Audio
- Images
- Fine-tuning
- Moderations
- Batch APIs
- Most thread / run / run-step primitives

## Development Environment

The project targets Python 3.13+ and uses `uv` for dependency and package management. On NixOS, the development environment is orchestrated via `shell.nix`.

To develop or run the application locally:

1. Enter the Nix shell:

   ```bash
   nix-shell
   ```

2. Inside the shell:
   - A `.venv` is created and activated automatically (if it does not exist).
   - A `start-server` helper is available for quick local runs.

3. Manage dependencies with `uv`:

   ```bash
   uv add <package>
   ```

4. Run Python scripts and tools (the virtual environment is already active).

5. Quickly start the local dev server:

   ```bash
   start-server
   ```

## Load Testing

The `tests/load/load_test.py` script fires concurrent HTTP requests at a running server instance to measure throughput and latency under parallel load. It targets `POST /v1/chat/completions`, which is the most compute-intensive endpoint.

**Prerequisites:** the server must already be running (see Quick Start above).

```bash
# Basic run: 50 requests, 10 concurrent workers
AMPLIFY_AI_TOKEN="..." python tests/load/load_test.py \
    --url http://localhost:8080 \
    --concurrency 10 \
    --total 50 \
    --model gpt-4o
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8080` | Base URL of the running server |
| `--token` | env `AMPLIFY_AI_TOKEN` | Bearer token (falls back to env var) |
| `--concurrency` | `10` | Number of parallel workers |
| `--total` | `50` | Total number of requests to send |
| `--model` | `gpt-4o` | Model identifier to use |
| `--prompt` | short single-word reply prompt | Prompt text sent in each request |
| `--timeout` | `60.0` | Per-request timeout in seconds |

The script prints a concise summary report on completion:

```
Load Test Report
========================================
Total requests  : 50
Successful      : 49
Failed          : 1
Total duration  : 42.31 s
Throughput      : 1.18 req/s

Latency (successful requests)
----------------------------------------
Min             : 0.841 s
Mean            : 3.612 s
Median (p50)    : 3.440 s
p95             : 6.120 s
p99             : 7.030 s
Max             : 7.450 s
```

Run logs are written to `logs/load_test.log`.

## Running Tests

All test commands are expected to run from within `nix-shell`.

- **Unit tests (no Amplify token required)**:

  ```bash
  uv run pytest tests/unit -v
  ```

- **Mocked integration tests (no Amplify token required)**  
  Exercise the full FastAPI stack with mocked Amplify upstream, including cline/kilo/openclaw usage patterns:

  ```bash
  uv run pytest tests/integration/mocked -v
  ```

- **Live integration tests (real Amplify API, token required)**  
  Run against the real Amplify API, mirroring cline/kilo/openclaw request patterns:

  ```bash
  AMPLIFY_AI_TOKEN="..." uv run pytest tests/integration/live/test_live_endpoints.py -v -s
  ```

If a `.env` file is present, `python-dotenv` (via `dotenv.load_dotenv`) loads `AMPLIFY_AI_TOKEN` automatically.

## API Prober

The CLI includes a prober that exercises all documented Amplify endpoints and records the results.

Run:

```bash
amplify probe
```

The prober:

- Reads `AMPLIFY_AI_TOKEN` and `AMPLIFY_AI_EMAIL` from `.env`
- Probes all relevant endpoints (including conflict variants for `/chat`, `/files/upload`, and `/files/tags/list`)
- Generates:
  - `docs-vibe/66_amplify_ai_probing_report.md` — full diagnostic report
  - `docs/amplify_api_probed.md` — concise, verified API reference

Email addresses in generated reports are redacted.

## NixOS Installation

The `nix/` directory provides a NixOS module that runs the server as a persistent systemd service. The module is designed to be imported directly from this repository (or a tarball) without a local checkout on the target system.

### Secrets Setup

Secrets must not live in the Nix store (world-readable). Create a secrets file on the target machine before running `nixos-rebuild`:

```bash
sudo install -m 400 -o root -g root /dev/null /run/secrets/amplify-ai.env
sudo tee /run/secrets/amplify-ai.env <<EOF
AMPLIFY_AI_TOKEN=amp-v1-...
AMPLIFY_AI_EMAIL=you@vanderbilt.edu
EOF
```

Consider using `agenix` or `sops-nix` to manage this file declaratively.

### Example `configuration.nix`

Import the module and configure the service:

```nix
{ config, pkgs, ... }:

let
  amplifyAiSrc = builtins.fetchTarball {
    # Pin to a specific commit SHA for reproducibility.
    # Replace <commit-sha> with the desired commit or use the branch tarball below.
    url    = "https://github.com/loeeeee/amplify-ai/archive/<commit-sha>.tar.gz";
    sha256 = "sha256:0000000000000000000000000000000000000000000000000000";
  };
in {
  imports = [ "${amplifyAiSrc}/nix/module.nix" ];

  services.amplify-ai = {
    enable          = true;
    environmentFile = /run/secrets/amplify-ai.env;

    # Optional overrides (defaults shown):
    # host         = "127.0.0.1";  # use "0.0.0.0" to expose on all interfaces
    # port         = 8080;
    # dataDir      = "/var/lib/amplify-ai";
    # openFirewall = false;        # set true to open the TCP port in the firewall
    # debug        = false;        # set true for verbose HTTP logging (avoid in production)
  };
}
```

To fetch the correct `sha256` for a given commit:

```bash
nix-prefetch-url --unpack \
  https://github.com/loeeeee/amplify-ai/archive/<commit-sha>.tar.gz
```

To quickly try the latest `main` branch before pinning (not reproducible):

```nix
url = "https://github.com/loeeeee/amplify-ai/archive/refs/heads/main.tar.gz";
# omit sha256 for a one-off test; always pin in production
```

Then apply and verify:

```bash
sudo nixos-rebuild switch
systemctl status amplify-ai
curl http://localhost:8080/v1/models
```

### Service Details

| Property       | Value                         |
|---             |---                            |
| Systemd unit   | `amplify-ai.service`          |
| Default bind   | `127.0.0.1:8080` (configurable via `services.amplify-ai.host` / `port`) |
| Debug logging  | `services.amplify-ai.debug` (default `false`; when `true`, sets `AMPLIFY_DEBUG=1`) |
| Log / state dir| `/var/lib/amplify-ai/` (default `dataDir`) |
| User           | Ephemeral (`DynamicUser = true`) |
| Restart policy | `on-failure`, 5 s back-off    |

## Token Usage Statistics

Every HTTP request is recorded to a CSV file for usage monitoring and debugging.

| Location      | Path                                   |
|---            |---                                     |
| Dev (local)   | `logs/token_stats.csv` (relative to CWD) |
| NixOS systemd | `/var/lib/amplify-ai/logs/token_stats.csv` |

### CSV Columns

`timestamp, ip_address, method, path, status_code, prompt_tokens, completion_tokens, total_tokens, error, model`

- `timestamp` — ISO 8601 UTC
- `ip_address` — client IP (`X-Forwarded-For` header, or direct connection IP)
- `prompt_tokens` / `completion_tokens` — estimated (4 characters per token); non-zero only for `POST /v1/chat/completions`
- `total_tokens` — `prompt_tokens + completion_tokens`
- `error` — empty on success; HTTP status or exception message on failure
- `model` — top-level `model` string from the JSON request body when present (empty for GET, multipart, or bodies without `model`)


