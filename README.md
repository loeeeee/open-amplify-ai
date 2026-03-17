# Amplify AI Compatibility Layer

![Build](https://github.com/loeeeee/amplify-ai/actions/workflows/build.yml/badge.svg)

An OpenAI-compatible HTTP layer in front of the Vanderbilt Amplify AI API, designed primarily for internal developers running local AI tools (cline, openclaw, kilo, etc.) and NixOS deployments.

External users are welcome, but this README assumes familiarity with NixOS and Amplify AI. For a concise, probed API reference, see `docs/amplify_api_probed.md`.

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
  - `docs-vibe/17_amplify_api_report.md` — full diagnostic report
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

`timestamp, ip_address, method, path, status_code, prompt_tokens, completion_tokens, total_tokens, error`

- `timestamp` — ISO 8601 UTC
- `ip_address` — client IP (`X-Forwarded-For` header, or direct connection IP)
- `prompt_tokens` / `completion_tokens` — estimated (4 characters per token); non-zero only for `POST /v1/chat/completions`
- `total_tokens` — `prompt_tokens + completion_tokens`
- `error` — empty on success; HTTP status or exception message on failure


