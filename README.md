# Amplify AI Compatibility Layer

![Build](https://github.com/loeeeee/open-amplify-ai/actions/workflows/build.yml/badge.svg)

A Python application that translates the Vanderbilt Amplify AI API into an
OpenAI-compatible HTTP API. It is designed for developers who want to use
standard OpenAI client libraries and tools (such as Cline, Kilo, or OpenClaw)
against Amplify AI without modifying those tools.

## Project Audience

This project targets developers at Vanderbilt University who:

- Have access to an Amplify AI token (`amp-v1-...`) and a Vanderbilt email
  address. [Get an API token](#get-api-tokens)
- Want to run AI coding tools (Cline, Kilo, OpenClaw, or any OpenAI-compatible
  client) against Amplify AI.
- Are running NixOS or any system with Python 3.13+ available.

The server speaks the OpenAI HTTP API on the client side and translates requests
to the Amplify AI upstream (`https://prod-api.vanderbilt.ai`) on the server
side. No changes to your client tool are required beyond pointing it at this
server's base URL.

## Features

### Models: `GET /v1/models`

Returns the list of models available on Amplify AI in OpenAI format.

- `GET /v1/models` — list all available models
- `GET /v1/models/{model_id}` — retrieve a single model by ID
- `DELETE /v1/models/{model_id}` — always returns `405`; Amplify does not
  support model deletion

Each model object includes extended metadata for downstream consumers such as
Kilo:

- `cost` — pricing in USD per million tokens (`input`, `output`; optionally
  `cache_read`, `cache_write`)
- `limit` — token limits (`context`, `output`)
- `capabilities` — feature flags (`images`, `system_prompt`, `description`)
- `display_name` — human-readable name from Amplify
- Legacy flat fields (`context_length`, `max_output_tokens`, `max_model_len`)
  are preserved for backward compatibility

Alias entries (`default`, `advanced`, `cheapest`, `documentCaching`) are
filtered from the list.

### Chat Completions: `POST /v1/chat/completions`

The primary endpoint. Translates OpenAI chat completion requests to Amplify AI
and returns OpenAI-format responses.

- Supports both non-streaming and streaming responses (SSE `data:` lines)
- Compatible with Cline, Kilo, OpenClaw, and any client using the OpenAI chat
  completions format
- Request validation:
  - Parameter checks: `max_tokens` (must be >0), `temperature` (0.0-2.0),
    `stream_options` (requires `stream=true`)
  - Message checks: content type, role, and empty content detection
  - Tool definition checks: structure and type validation
- Tool call handling:
  - Canonical embedded JSON tool calls are extracted, including multiple calls
    per message
  - Legacy `[Tool Call: name]` block format is supported, including text before
    the block
  - XML fallback format is supported but explicitly lossy
  - Deeply nested and partially parsed JSON is handled
- Streaming: a state machine manages clean mode transitions between text and
  tool call chunks
- Usage accounting: every response includes a `usage` object with
  `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens` (always 0; Amplify does not report cache
  hits), and `cost` in USD when model pricing is available. Token counts use a
  4-characters-per-token heuristic applied to the post-transformation request.
  For streaming, the final chunk includes the full usage block when
  `stream_options.include_usage` is true.
- HTTP status codes are mapped precisely from upstream Amplify errors.

## Quick Start: Nix-Enabled Machines

**Requirements:**

- Nix with `nix-shell` available
- An Amplify AI token and Vanderbilt email address

**Setup:**

Create a `.env` file in the project root:

```
AMPLIFY_AI_TOKEN=amp-v1-...
AMPLIFY_AI_EMAIL=you@vanderbilt.edu
```

**Start the server:**

```bash
nix-shell
amplify server
```

The server binds to `0.0.0.0:8080` by default. Override with:

```bash
amplify server --port 9090
```

or via environment variables:

- `AMPLIFY_SERVER_HOST` (default: `0.0.0.0`)
- `AMPLIFY_SERVER_PORT` (default: `8080`)

Enable verbose request/response logging:

```bash
amplify server --debug
# or
export AMPLIFY_DEBUG=1
```

Point your OpenAI-compatible tool at `http://localhost:8080` and use your
Amplify AI token as the API key.

## Quick Start: Python-Enabled Machines (Non-Nix)

**Requirements:**

- Python 3.13+
- `uv` package manager (`pip install uv` or see https://docs.astral.sh/uv/)
- An Amplify AI token and Vanderbilt email address

**Setup:**

```bash
# Clone the repository
git clone https://github.com/loeeeee/amplify-ai.git
cd amplify-ai

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Create a .env file
cat > .env <<EOF
AMPLIFY_AI_TOKEN=amp-v1-...
AMPLIFY_AI_EMAIL=you@vanderbilt.edu
EOF
```

**Start the server:**

```bash
amplify server
```

The server binds to `0.0.0.0:8080` by default. See the Nix section above for
port and debug options; they work identically.

## Additional Features

### Dashboard

A token usage dashboard is available at:

- `GET /` — plain HTML page showing aggregate usage from `logs/token_stats.csv`
  (or `AMPLIFY_STATS_CSV`). Displays:
  - Totals (requests, prompt tokens, completion tokens, total tokens, errors)
    for the last 24 hours, last 7 days, and lifetime
  - Usage by model for each period
  - Average requests per second and tokens per second over the last 60 seconds
  - The 100 most recent requests, with timestamps shown in the browser's local
    timezone
  - Auto-refreshes every 5 seconds

- `GET /usage` — JSON summary of the same stats for a configurable UTC lookback
  window. Query parameter `seconds` (integer, default `300`, minimum `1`,
  maximum 90 days in seconds) controls the window. Response fields include
  `window_seconds`, `generated_at_utc`, `cutoff_utc`, token totals,
  `total_requests`, `error_count`, `requests_per_second`, `tokens_per_second`,
  HTTP status bucket counts (`http_2xx`, `http_3xx`, `http_4xx`, `http_5xx`,
  `http_other`), and `by_model`. Suitable for use with monitoring tools such as
  Gatus.

The CSV path can be overridden with the `AMPLIFY_STATS_CSV` environment
variable.

### Files

- `GET /v1/files` — list uploaded files
- `POST /v1/files` — upload a file (uses Amplify pre-signed URL and S3 PUT)
- `GET /v1/files/{file_id}` — retrieve a file record
- `DELETE /v1/files/{file_id}` — delete a file
- `GET /v1/files/{file_id}/content` — download file content (Code Interpreter
  files only)

### Assistants

- `GET /v1/assistants` — list assistants
- `POST /v1/assistants` — create an assistant
- `GET /v1/assistants/{assistant_id}` — retrieve an assistant
- `POST /v1/assistants/{assistant_id}` — modify an assistant
- `DELETE /v1/assistants/{assistant_id}` — delete an assistant

### Threads

- `DELETE /v1/threads/{thread_id}` — delete a thread
- All other thread, message, run, and run-step endpoints return `501 Not
  Implemented`

### Vector Stores

- `POST /v1/vector_stores` — create a virtual store (backed by Amplify tags)
- `GET /v1/vector_stores/{id}` — retrieve a vector store
- `DELETE /v1/vector_stores/{id}` — delete a vector store (removes backing tag
  only)
- `GET /v1/vector_stores/{id}/files` — list files in a store
- `POST /v1/vector_stores/{id}/files` — add a file to a store
- All other vector store batch endpoints return `501 Not Implemented`

### Unsupported Endpoints

The following OpenAI-style features are not implemented and return `501 Not
Implemented`:

- Embeddings
- Audio
- Images
- Fine-tuning
- Moderations
- Batch APIs
- Most thread, run, and run-step primitives

### Token Usage Statistics

Every HTTP request is recorded to a CSV file.

| Location | Path |
|---|---|
| Local dev | `logs/token_stats.csv` (relative to working directory) |
| NixOS systemd | `/var/lib/amplify-ai/logs/token_stats.csv` |

CSV columns: `timestamp, ip_address, method, path, status_code, prompt_tokens,
completion_tokens, total_tokens, error, model`

- `timestamp` — ISO 8601 UTC
- `ip_address` — client IP (`X-Forwarded-For` header or direct connection IP)
- `prompt_tokens` / `completion_tokens` — estimated (4 characters per token);
  non-zero only for `POST /v1/chat/completions`
- `total_tokens` — `prompt_tokens + completion_tokens`
- `error` — empty on success; HTTP status or exception message on failure
- `model` — `model` field from the JSON request body when present; empty for
  GET, multipart, or bodies without `model`

## Development Guide

### Versioning and releases

This project follows semantic versioning.

- Current version: `1.0.0`
- Release tags are formatted as `v<version>` (example: `v1.0.0`)

### Running Tests

**Unit tests (no Amplify token required):**

```bash
uv run pytest tests/unit -v
```

Unit tests cover chat endpoint functionality, tool call parsing (canonical,
legacy, JSON, and XML formats), agent response validation, mixed content
handling, parameter compatibility, streaming protocol, response invariants,
upstream error handling, token counting, file operations, assistant management,
thread operations, vector store operations, and dashboard statistics.

**Mocked integration tests (no Amplify token required):**

```bash
uv run pytest tests/integration/mocked -v
```

Exercises the full FastAPI stack with a mocked Amplify upstream, including
Cline, Kilo, and OpenClaw usage patterns.

**Live integration tests (real Amplify API, token required):**

```bash
AMPLIFY_AI_TOKEN="..." uv run pytest tests/integration/live/test_live_endpoints.py -v -s
```

If a `.env` file is present, `python-dotenv` loads `AMPLIFY_AI_TOKEN`
automatically.

### Load Testing

The `tests/load/load_test.py` script fires concurrent requests at a running
server instance to measure throughput and latency. It targets `POST
/v1/chat/completions`.

The server must be running before executing the load test.

```bash
AMPLIFY_AI_TOKEN="..." python tests/load/load_test.py \
    --url http://localhost:8080 \
    --concurrency 10 \
    --total 50 \
    --model gpt-4o
```

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8080` | Base URL of the running server |
| `--token` | env `AMPLIFY_AI_TOKEN` | Bearer token (falls back to env var) |
| `--concurrency` | `10` | Number of parallel workers |
| `--total` | `50` | Total number of requests to send |
| `--model` | `gpt-4o` | Model identifier to use |
| `--prompt` | short single-word reply prompt | Prompt text sent in each request |
| `--timeout` | `60.0` | Per-request timeout in seconds |

Run logs are written to `logs/load_test.log`.

### NixOS Installation

The `nix/` directory provides a NixOS module that runs the server as a
persistent systemd service.

**Secrets setup** (must not live in the Nix store):

```bash
sudo install -m 400 -o root -g root /dev/null /run/secrets/amplify-ai.env
sudo tee /run/secrets/amplify-ai.env <<EOF
AMPLIFY_AI_TOKEN=amp-v1-...
AMPLIFY_AI_EMAIL=you@vanderbilt.edu
EOF
```

Consider `agenix` or `sops-nix` for declarative secret management.

**Example `configuration.nix`:**

```nix
{ config, pkgs, ... }:

let
  amplifyAiSrc = builtins.fetchTarball {
    url    = "https://github.com/loeeeee/amplify-ai/archive/<commit-sha>.tar.gz";
    sha256 = "sha256:0000000000000000000000000000000000000000000000000000";
  };
in {
  imports = [ "${amplifyAiSrc}/nix/module.nix" ];

  services.amplify-ai = {
    enable          = true;
    environmentFile = /run/secrets/amplify-ai.env;

    # Optional overrides (defaults shown):
    # host         = "127.0.0.1";
    # port         = 8080;
    # dataDir      = "/var/lib/amplify-ai";
    # openFirewall = false;
    # debug        = false;
  };
}
```

Fetch the correct `sha256` for a given commit:

```bash
nix-prefetch-url --unpack \
  https://github.com/loeeeee/amplify-ai/archive/<commit-sha>.tar.gz
```

Apply and verify:

```bash
sudo nixos-rebuild switch
systemctl status amplify-ai
curl http://localhost:8080/v1/models
```

**Service details:**

| Property | Value |
|---|---|
| Systemd unit | `amplify-ai.service` |
| Default bind | `127.0.0.1:8080` (configurable via `services.amplify-ai.host` / `port`) |
| Debug logging | `services.amplify-ai.debug` (default `false`; when `true`, sets `AMPLIFY_DEBUG=1`) |
| Log / state dir | `/var/lib/amplify-ai/` (default `dataDir`) |
| User | Ephemeral (`DynamicUser = true`) |
| Restart policy | `on-failure`, 5 s back-off |

# Get API Tokens

- Login to [Vanderbilt.AI](https://www.vanderbilt.ai)
- Open hamburger manual by clicking profile picture at top right corner
- Go to settings
- Create an account at Account tab, the strings can be random
- Go to API Access tab
- Create an API key

Note: No worries, you won't be billed
