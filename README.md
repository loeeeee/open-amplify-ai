# Amplify AI Compatibility Layer

![Build](https://github.com/loeeeee/amplify-ai/actions/workflows/build.yml/badge.svg)

This project provides an OpenAI-compatible wrapper for the Amplify AI API used at 
Vanderbilt University.

## OpenAI Compatible Server
We have added an OpenAI-compatible API layer that translates requests to the Amplify AI format.

### Running the server
You can start the server locally by running:
```bash
amplify server
```
The server will bind to `http://0.0.0.0:8080` by default.

### Endpoints Supported

#### Models
* `GET /v1/models` — list all available models
* `GET /v1/models/{model}` — retrieve a model by ID
* `DELETE /v1/models/{model}` — returns 405 (Amplify does not support model deletion)

#### Chat Completions
* `POST /v1/chat/completions` — supports streaming (SSE) and non-streaming responses

#### Files
* `GET /v1/files` — list all uploaded files
* `POST /v1/files` — upload a file (two-step: Amplify pre-signed URL + S3 PUT)
* `GET /v1/files/{file_id}` — retrieve a file record
* `DELETE /v1/files/{file_id}` — delete a file
* `GET /v1/files/{file_id}/content` — download file (Code Interpreter files only)

#### Assistants
* `GET /v1/assistants` — list all assistants
* `POST /v1/assistants` — create an assistant
* `GET /v1/assistants/{assistant_id}` — retrieve an assistant
* `POST /v1/assistants/{assistant_id}` — modify an assistant
* `DELETE /v1/assistants/{assistant_id}` — delete an assistant

#### Threads
* `DELETE /v1/threads/{thread_id}` — delete a thread
* All other thread, message, run, and run step endpoints return `501 Not Implemented`

#### Vector Stores
* `POST /v1/vector_stores` — create a virtual store (backed by Amplify tags)
* `GET /v1/vector_stores/{id}` — retrieve a vector store
* `DELETE /v1/vector_stores/{id}` — delete a vector store (removes backing tag only)
* `GET /v1/vector_stores/{id}/files` — list files in a store
* `POST /v1/vector_stores/{id}/files` — add a file to a store
* All other vector store batch endpoints return `501 Not Implemented`

#### Unsupported (501)
Embeddings, Audio, Images, Fine-tuning, Moderations, Batch, and most thread/run primitives
all return `501 Not Implemented` with a clear message.

## DevelopmentEnvironment Setup

The project is built on Python 3.13 and uses `uv` for dependency and package management.
On NixOS, the environment is orchestrated using `shell.nix`.

To develop or run the application:
1. Enter the Nix shell:
    ```bash
    nix-shell
    ```
2. The shell will automatically create a `.venv` (if it doesn't exist) and activate it.
3. Manage dependencies with `uv` (e.g., `uv add <package>`).
4. Run scripts with standard Python natively since the `.venv` is loaded.

> **Note:** Accessing the Amplify AI API requires an API token. Set the following in your `.env` file:
> - `AMPLIFY_AI_TOKEN` — your API key (e.g., `amp-v1-...`)
> - `AMPLIFY_AI_EMAIL` — your Vanderbilt email address (used to construct request payloads)

### Running Tests
To run the mock-based unit tests for the server:
```bash
uv run pytest src/open_amplify_ai/test_server.py
```

### API Prober
To probe all documented endpoints (including conflict variants), run:
```bash
amplify probe
```
The script reads `AMPLIFY_AI_TOKEN` and `AMPLIFY_AI_EMAIL` from `.env`, probes all endpoints 
(including conflict variants for `/chat`, `/files/upload`, and `/files/tags/list`), and generates:
- `docs-vibe/17_amplify_api_report.md` — full diagnostic report
- `docs/amplify_api_probed.md` — concise verified API reference

Email addresses are redacted in all generated reports.

## NixOS Installation

The `nix/` directory provides a NixOS module to run the server as a persistent systemd service.
No local checkout is needed — the module is fetched directly from GitHub.

### Secrets Setup

Secrets are never stored in the Nix store (world-readable). Create a secrets file on the
target machine before running `nixos-rebuild`:

```bash
sudo install -m 400 -o root -g root /dev/null /run/secrets/amplify-ai.env
sudo tee /run/secrets/amplify-ai.env <<EOF
AMPLIFY_AI_TOKEN=amp-v1-...
AMPLIFY_AI_EMAIL=you@vanderbilt.edu
EOF
```

Consider using [agenix](https://github.com/ryantm/agenix) or
[sops-nix](https://github.com/Mic92/sops-nix) to manage this file declaratively.

### configuration.nix

Use `builtins.fetchTarball` to pull the module directly from GitHub:

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

    # Optional overrides (shown with defaults):
    # host        = "127.0.0.1";  # use "0.0.0.0" to expose on all interfaces
    # port        = 8080;
    # openFirewall = false;       # set true to open the TCP port in the firewall
  };
}
```

**Get the correct `sha256`** for a given commit:

```bash
nix-prefetch-url --unpack \
  https://github.com/loeeeee/amplify-ai/archive/<commit-sha>.tar.gz
```

Or to quickly try the latest `main` branch before pinning (not reproducible):

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

| Property | Value |
|---|---|
| Systemd unit | `amplify-ai.service` |
| Default bind | `127.0.0.1:8080` |
| Log / state dir | `/var/lib/amplify-ai/` |
| User | ephemeral (`DynamicUser = true`) |
| Restart policy | `on-failure`, 5 s back-off |

