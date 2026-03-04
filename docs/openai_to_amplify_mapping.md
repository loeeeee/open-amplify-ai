# OpenAI API to Amplify AI Endpoint Mapping

This document maps every standard OpenAI REST API endpoint to the corresponding
Amplify AI endpoint, based on the verified probe results in `amplify_api_probed.md`
and the reference in `amplify_api_reference.md`.

Amplify AI base URL: `https://prod-api.vanderbilt.ai`
OpenAI base URL:     `https://api.openai.com/v1`

## Legend

| Symbol | Meaning |
|--------|---------|
| DIRECT | Amplify has an endpoint that functionally covers this OpenAI endpoint. Adapter code needed to translate request/response shapes. |
| PARTIAL | Amplify has an endpoint with overlapping function, but one or more fields, features, or semantics are missing. |
| WORKAROUND | No single endpoint matches. A multi-step chain of Amplify endpoints can approximate the OpenAI endpoint. |
| NONE | No current Amplify endpoint covers this. Must be implemented locally or proxied to a different provider. |

---

## 1. Models

### `GET /v1/models`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `GET /available_models` |
| Probe status | 200 OK (working) |

**Adapter notes:**

Amplify returns:
```json
{ "success": true, "data": { "models": [ { "id": "...", "name": "...", ... } ] } }
```

OpenAI expects:
```json
{ "object": "list", "data": [ { "id": "...", "object": "model", "created": 0, "owned_by": "..." } ] }
```

The server must iterate `data.models`, project each record to the OpenAI `Model` object shape, and wrap in `{ "object": "list", "data": [...] }`.
Additional Amplify-specific fields (`inputContextWindow`, `outputTokenLimit`, `inputTokenCost`, etc.) can be passed through in a non-standard extension object or discarded.
This mapping is already implemented in `server.py`.

---

### `GET /v1/models/{model}`

| Field | Value |
|-------|-------|
| Coverage | PARTIAL |
| Amplify endpoint | `GET /available_models` (full list, filter client-side) |
| Probe status | 200 OK (working, but full-list only) |

**Adapter notes:**

Amplify has no per-model lookup endpoint. The adapter must:
1. Call `GET /available_models`.
2. Filter the `data.models` array to find the entry whose `id` matches `{model}`.
3. Return a single `Model` object or `404` if not found.

Latency is slightly higher than a direct lookup but functionally equivalent.

---

### `DELETE /v1/models/{model}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify does not support deleting models; models are centrally managed by the platform.
Return `405 Method Not Allowed` or a static error JSON to callers expecting this endpoint.

---

## 2. Chat Completions

### `POST /v1/chat/completions`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `POST /chat` |
| Probe status | 200 OK (working) |

**Adapter notes:**

OpenAI request shape:
```json
{
  "model": "gpt-4o",
  "messages": [{ "role": "user", "content": "Hello" }],
  "temperature": 0.7,
  "max_tokens": 4000,
  "stream": false
}
```

Amplify request shape:
```json
{
  "data": {
    "temperature": 0.7,
    "max_tokens": 4000,
    "messages": [{ "role": "user", "content": "Hello" }],
    "options": { "model": { "id": "gpt-4o" } }
  }
}
```

Key differences:
- `model` is a top-level string in OpenAI but nested as `options.model.id` in Amplify.
- The Amplify response wraps content in `data.data` (string), while OpenAI uses `choices[0].message.content`.
- Amplify may stream via Server-Sent Events (SSE). When `stream: true` is requested by the caller, the adapter must forward SSE chunks from Amplify and re-emit them in OpenAI SSE format (`data: {"choices": [{"delta": {"content": "..."}}]}`).
- Amplify supports `dataSources` and `options.assistantId`. These have no OpenAI equivalent; they can be injected from server configuration or passed via non-standard request fields.

This mapping is partially implemented in `server.py`. Streaming support is marked `TODO`.

---

## 3. Files

### `GET /v1/files`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `POST /files/query` |
| Probe status | 200 OK (working) |

**Adapter notes:**

OpenAI returns a list of file metadata objects. Amplify's `/files/query` returns a paginated list with similar metadata. The adapter must:
1. Issue `POST /files/query` with an appropriate page size.
2. Iterate pages using the `pageKey` cursor.
3. Map each Amplify item to an OpenAI `File` object:
   - `id` <- `item.id`
   - `object` = `"file"`
   - `bytes` <- not directly available; set to `0` or estimate from `totalTokens`
   - `created_at` <- parse ISO `item.createdAt` to Unix timestamp
   - `filename` <- `item.name`
   - `purpose` <- not available; default to `"assistants"` or allow query parameter override

---

### `POST /v1/files`

| Field | Value |
|-------|-------|
| Coverage | PARTIAL |
| Amplify endpoint | `POST /files/upload` (two-step: get pre-signed URL, then PUT to S3) |
| Probe status | 200 OK (working) |

**Adapter notes:**

OpenAI accepts `multipart/form-data` with the file binary and a `purpose` field, returning a `File` object immediately.

Amplify uses a two-step upload:
1. `POST /files/upload` with file metadata returns `{ uploadUrl, key, ... }`.
2. Client must `PUT` the file binary to `uploadUrl` (a pre-signed S3 URL).

The adapter must:
1. Receive the OpenAI multipart request.
2. Call `POST /files/upload` with the filename, MIME type, `knowledgeBase: "default"`, and the optional `actions` array.
3. `PUT` the binary body to the returned `uploadUrl`.
4. Return a synthetic OpenAI `File` response using the returned `key` as the file ID.

The total operation takes two HTTP round trips instead of one.

---

### `DELETE /v1/files/{file_id}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify has no documented file deletion endpoint. The adapter should return `501 Not Implemented`.
As a partial workaround, files can be disassociated from assistants via `POST /assistant/create` (update),
but the file record in the knowledge base cannot be fully deleted through the API.

---

### `GET /v1/files/{file_id}`

| Field | Value |
|-------|-------|
| Coverage | PARTIAL |
| Amplify endpoint | `POST /files/query` (full list query, filter by `id`) |
| Probe status | 200 OK (working via query) |

**Adapter notes:**

Amplify has no single-file lookup endpoint. To simulate `GET /v1/files/{file_id}`:
1. Call `POST /files/query` with a narrow filter or fetch multiple pages.
2. Find the item whose `id` matches `{file_id}`.
3. Map to OpenAI `File` shape (same projection as for `GET /v1/files`).

This is O(n) in the number of files. For large file sets, this is inefficient. Cache the file list or note the limitation.

---

### `GET /v1/files/{file_id}/content`

| Field | Value |
|-------|-------|
| Coverage | PARTIAL |
| Amplify endpoint | `POST /assistant/files/download/codeinterpreter` (code interpreter files only) |
| Probe status | 200 OK |

**Adapter notes:**

The Amplify equivalent only covers files associated with code interpreter assistants.
For general knowledge-base files, there is no direct download endpoint. The S3 content URLs embedded in the upload response are time-limited pre-signed URLs, not re-fetchable via the Amplify API.

Workaround for code interpreter files:
1. Call `POST /assistant/files/download/codeinterpreter` with `{ "data": { "key": file_id } }`.
2. Follow the returned `downloadUrl` to retrieve the binary.

For general files: return `501 Not Implemented`.

---

## 4. Assistants

### `POST /v1/assistants`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `POST /assistant/create` |
| Probe status | 200 OK (working) |

**Adapter notes:**

OpenAI request body: `{ "model", "name", "description", "instructions", "tools", "file_ids" }`

Amplify request body (`data` wrapped):
```json
{
  "name": "...",
  "description": "...",
  "instructions": "...",
  "tags": [],
  "dataSources": [],
  "tools": []
}
```

Mapping:
- `model` has no direct Amplify equivalent at assistant creation. The model is specified per chat call. Ignore or store as a tag.
- `file_ids` -> translate to full `dataSources` objects retrieved from `POST /files/query`.
- `tools` -> Amplify does not support function calling tools natively. `code_interpreter` maps to `/assistant/create/codeinterpreter`. Other tool types have no equivalent.

The response returns `assistantId` (`astp/...`) and `id` (`ast/...`). The `astp/` ID is the shareable pointer; `ast/` is the internal record. Use `astp/` as the OpenAI `assistant_id`.

---

### `GET /v1/assistants/{assistant_id}`

| Field | Value |
|-------|-------|
| Coverage | PARTIAL |
| Amplify endpoint | `GET /assistant/list` (full list, filter client-side) |
| Probe status | 200 OK (working) |

**Adapter notes:**

Amplify has no single-assistant GET endpoint. The adapter must:
1. Call `GET /assistant/list`.
2. Find the entry matching `{assistant_id}` (`assistantId` field).
3. Return the projected record.

The Amplify `GET /assistant/list` response includes `instructions`, `disclaimerHash`, `coreHash`, `dataSources`, and `user` fields but does not return `description`, `file_ids`, or `tools` in OpenAI format. Map best-effort.

---

### `POST /v1/assistants/{assistant_id}`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `POST /assistant/create` (upsert when `assistantId` is provided) |
| Probe status | 200 OK (working) |

**Adapter notes:**

Pass `assistantId` in the request body to trigger an update. The Amplify API treats the presence of an existing `assistantId` in the body as an update command. The same field mapping rules as `POST /v1/assistants` apply.

---

### `DELETE /v1/assistants/{assistant_id}`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `POST /assistant/delete` |
| Probe status | 200 OK |

**Adapter notes:**

OpenAI uses `DELETE /v1/assistants/{assistant_id}`. Amplify uses `POST /assistant/delete` with `{ "data": { "assistantId": "..." } }`.

The adapter must bridge the HTTP method mismatch. Extract `assistant_id` from the path and post it to Amplify. The returned `success` flag must be translated to an OpenAI `DeletedObject` response:
```json
{ "id": "...", "object": "assistant.deleted", "deleted": true }
```

---

### `GET /v1/assistants`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `GET /assistant/list` |
| Probe status | 200 OK (working) |

**Adapter notes:**

Map the Amplify list to OpenAI's paginated `ListAssistantsResponse`. Amplify does not provide cursor-based pagination for the assistant list, so `has_more` is always `false` and `first_id`/`last_id` are derived from the array boundaries.

---

## 5. Threads

### `POST /v1/threads`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `POST /assistant/chat/codeinterpreter` (first message creates thread) |
| Probe status | 400 (failed with empty body; requires valid assistantId) |

**Adapter notes:**

Amplify does not have a standalone "create thread" endpoint. A thread is implicitly created when the first message is sent to a code interpreter assistant. The returned `threadId` (`email/thr/...`) serves as the thread identifier.

Workaround:
1. On `POST /v1/threads`, store the `messages` payload locally.
2. On the first `POST /v1/threads/{thread_id}/runs`, call `POST /assistant/chat/codeinterpreter` with the buffered messages plus the new run messages. Extract the returned `threadId` and substitute it for the local placeholder.

For non-code-interpreter assistants (standard chat), threads have no equivalent -- each `POST /chat` call is stateless. To emulate threads, the adapter must keep conversation history in local state and re-submit the full `messages` array on each `/chat` call.

---

### `GET /v1/threads/{thread_id}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify has no thread retrieval endpoint. The adapter must maintain thread state locally (in-memory or database). Return a synthetic thread object from local storage.

---

### `POST /v1/threads/{thread_id}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Thread metadata updates have no equivalent in Amplify. Adapter stores metadata locally.

---

### `DELETE /v1/threads/{thread_id}`

| Field | Value |
|-------|-------|
| Coverage | DIRECT |
| Amplify endpoint | `DELETE /assistant/openai/thread/delete?threadId=...` |
| Probe status | 200 OK (working, returns success=false for non-existent IDs) |

**Adapter notes:**

Map `thread_id` from the path to the `threadId` query parameter. The Amplify endpoint responds with `success: true/false`; translate to OpenAI `DeletedObject`:
```json
{ "id": "...", "object": "thread.deleted", "deleted": true }
```

---

## 6. Messages

### `POST /v1/threads/{thread_id}/messages`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `POST /assistant/chat/codeinterpreter` (combined message + run) |

**Adapter notes:**

Amplify does not separate "add message" from "run". The adapter must buffer messages locally and submit them combined with the next run request, or submit immediately to Amplify and aggregate results.

For standard (non-code-interpreter) assistants, buffer the message in local thread state and include it in the next `POST /chat` call.

---

### `GET /v1/threads/{thread_id}/messages`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

No Amplify endpoint returns the message history of a thread. Message history must be maintained entirely in local adapter state.

---

### `GET /v1/threads/{thread_id}/messages/{message_id}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Same limitation as `GET /v1/threads/{thread_id}/messages`. Serve from local state.

---

## 7. Runs

### `POST /v1/threads/{thread_id}/runs`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `POST /assistant/chat/codeinterpreter` or `POST /chat` |
| Probe status | 200 OK (code interpreter); 200 OK (standard chat) |

**Adapter notes:**

Chain:
1. Look up local thread state to retrieve all buffered messages plus the new `assistant_id`.
2. If assistant is code interpreter type: POST to `POST /assistant/chat/codeinterpreter`.
3. If assistant is standard type: POST to `POST /chat` with the full message history.
4. Return a synthetic `Run` object with `status: "completed"` (Amplify calls are synchronous, unlike OpenAI's async run model).

OpenAI runs return immediately with `status: "queued"` and require polling. Since Amplify is synchronous, the adapter can simulate a completed run in the same response.

---

### `GET /v1/threads/{thread_id}/runs/{run_id}`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | None (runs are synchronous) |

**Adapter notes:**

Amplify does not have async run execution. The adapter must store the run result after the synchronous call and return it when polled. Since the response is already available, always return `status: "completed"`.

---

### `POST /v1/threads/{thread_id}/runs/{run_id}/cancel`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify calls are synchronous. By the time a cancel request arrives, the Amplify call is already complete. Return a synthetic response with `status: "cancelled"` or `status: "completed"`.

---

### `GET /v1/threads/{thread_id}/runs`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Serve run history from local adapter state.

---

### `POST /v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify does not support function/tool calling with a pause-and-resume run model. Return `501 Not Implemented`.

---

### `POST /v1/threads/runs`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `POST /chat` or `POST /assistant/chat/codeinterpreter` |

**Adapter notes:**

This is the "create thread and run in one call" endpoint. Adapter must:
1. Create a local thread from the supplied `messages`.
2. Immediately submit to the appropriate Amplify chat endpoint.
3. Return a synthetic `Run` with `status: "completed"` and embed the assistant message in the thread state.

---

## 8. Run Steps

### `GET /v1/threads/{thread_id}/runs/{run_id}/steps`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify does not expose individual reasoning steps within a run. Return a synthetic list with a single `tool_calls` step type describing the aggregated output, or return an empty list.

---

### `GET /v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Same as above. Serve from local state or return a synthetic step.

---

## 9. Embeddings

### `POST /v1/embeddings`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `POST /embedding-dual-retrieval` |
| Probe status | 200 OK (working, but returns retrieval results not raw vectors) |

**Adapter notes:**

The Amplify `/embedding-dual-retrieval` endpoint performs a semantic search rather than returning raw embedding vectors. It does not correspond directly to OpenAI's embedding generation endpoint.

Differences:
- OpenAI `POST /v1/embeddings` accepts raw text and returns float vectors.
- Amplify `/embedding-dual-retrieval` accepts a query + data source IDs and returns ranked document snippets.

Workaround: Use the Amplify `/chat` endpoint with a prompt asking the model to embed or compare text, and approximate with cosine similarity from the retrieved chunks. This is a semantic similarity approximation, not a true embedding response.

For applications needing exact embedding vectors, there is no Amplify equivalent -- the request must be forwarded to a real OpenAI or compatible embedding provider.

---

## 10. Audio

### `POST /v1/audio/speech`
### `POST /v1/audio/transcriptions`
### `POST /v1/audio/translations`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify has no audio processing capabilities in the documented API. These endpoints cannot be implemented using any Amplify backend. Either forward to OpenAI directly or return `501 Not Implemented`.

---

## 11. Images

### `POST /v1/images/generations`
### `POST /v1/images/edits`
### `POST /v1/images/variations`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify does not expose an image generation or editing API. These endpoints have no equivalent. Forward to OpenAI or another provider, or return `501 Not Implemented`.

---

## 12. Fine-tuning

### `POST /v1/fine_tuning/jobs`
### `GET /v1/fine_tuning/jobs`
### `GET /v1/fine_tuning/jobs/{fine_tuning_job_id}`
### `POST /v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel`
### `GET /v1/fine_tuning/jobs/{fine_tuning_job_id}/events`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify does not support model fine-tuning. Return `501 Not Implemented` for all fine-tuning endpoints.

---

## 13. Moderations

### `POST /v1/moderations`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify has no content moderation endpoint. The adapter must return a safe-default response (all categories `false`, all scores `0.0`) or forward to OpenAI's moderation endpoint.

---

## 14. Batch

### `POST /v1/batches`
### `GET /v1/batches/{batch_id}`
### `POST /v1/batches/{batch_id}/cancel`
### `GET /v1/batches`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify does not support batch request processing. Return `501 Not Implemented`. For bulk workloads, the adapter could fan out requests concurrently to `POST /chat`, but this does not match OpenAI's async batch semantics.

---

## 15. Vector Stores

### `POST /v1/vector_stores`

| Field | Value |
|-------|-------|
| Coverage | PARTIAL |
| Amplify endpoint | `POST /files/tags/create`, `POST /files/upload` (build a tagged group) |
| Probe status | 200 OK |

**Adapter notes:**

Amplify's knowledge base + tagging system approximates vector stores. A "vector store" can be represented as a tag:
1. Create a unique tag via `POST /files/tags/create` to represent the store.
2. Upload files via `POST /files/upload` and attach the tag via `POST /files/set_tags`.
3. Return a synthetic `VectorStore` object with the tag name as the ID.

---

### `GET /v1/vector_stores/{vector_store_id}`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `GET /files/tags/list` + `POST /files/query` (filter by tag) |
| Probe status | 200 OK |

**Adapter notes:**

1. Call `GET /files/tags/list` and confirm the tag exists.
2. Call `POST /files/query` with `{ "tags": [vector_store_id] }` to count files.
3. Return a synthetic `VectorStore` response with `file_counts` from the query result.

---

### `POST /v1/vector_stores/{vector_store_id}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify has no vector store rename or update endpoint. Tags cannot be renamed via the documented API; the adapter must create the new tag and re-tag all files.

---

### `DELETE /v1/vector_stores/{vector_store_id}`

| Field | Value |
|-------|-------|
| Coverage | PARTIAL |
| Amplify endpoint | `POST /files/tags/delete` |
| Probe status | 200 OK (working) |

**Adapter notes:**

Deleting the tag via `POST /files/tags/delete` removes the virtual store identifier but does not delete the underlying files. The adapter should warn callers or accept a `delete_files=true` query param to also trigger whatever deletion mechanism is available.

---

### `GET /v1/vector_stores/{vector_store_id}/files`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `POST /files/query` with tag filter |
| Probe status | 200 OK (working) |

**Adapter notes:**

Call `POST /files/query` with `{ "data": { "tags": [vector_store_id], "pageSize": 100 } }` and paginate via `pageKey`. Project each result to the OpenAI `VectorStoreFile` shape.

---

### `POST /v1/vector_stores/{vector_store_id}/files`

| Field | Value |
|-------|-------|
| Coverage | WORKAROUND |
| Amplify endpoint | `POST /files/set_tags` |
| Probe status | 200 OK (working) |

**Adapter notes:**

Call `POST /files/set_tags` with `{ "data": { "id": file_id, "tags": [vector_store_id] } }` to associate the file with the virtual store.

---

### `DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify has no endpoint to remove a specific file from a knowledge base or dissociate a tag from a single file without overwriting all tags. `POST /files/set_tags` replaces the full tag list, so the adapter can GET the current tags, remove the target tag, and PUT the remaining tags back.

---

### `POST /v1/vector_stores/{vector_store_id}/file_batches`
### `GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}`
### `POST /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel`
### `GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files`

| Field | Value |
|-------|-------|
| Coverage | NONE |
| Amplify endpoint | None |

**Notes:**

Amplify has no batch file ingestion endpoint. For file batch upload, the adapter can iterate files sequentially calling `POST /files/upload` for each, but there is no async batch status to poll. Return synthetic `completed` batch status immediately.

---

## Summary Table

| OpenAI Endpoint | Method | Coverage | Amplify Endpoint(s) |
|-----------------|--------|----------|---------------------|
| `/v1/models` | GET | DIRECT | `GET /available_models` |
| `/v1/models/{model}` | GET | PARTIAL | `GET /available_models` (filter) |
| `/v1/models/{model}` | DELETE | NONE | — |
| `/v1/chat/completions` | POST | DIRECT | `POST /chat` |
| `/v1/files` | GET | DIRECT | `POST /files/query` |
| `/v1/files` | POST | PARTIAL | `POST /files/upload` (2-step) |
| `/v1/files/{file_id}` | DELETE | NONE | — |
| `/v1/files/{file_id}` | GET | PARTIAL | `POST /files/query` (filter) |
| `/v1/files/{file_id}/content` | GET | PARTIAL | `POST /assistant/files/download/codeinterpreter` |
| `/v1/assistants` | POST | DIRECT | `POST /assistant/create` |
| `/v1/assistants/{id}` | GET | PARTIAL | `GET /assistant/list` (filter) |
| `/v1/assistants/{id}` | POST | DIRECT | `POST /assistant/create` (upsert) |
| `/v1/assistants/{id}` | DELETE | DIRECT | `POST /assistant/delete` |
| `/v1/assistants` | GET | DIRECT | `GET /assistant/list` |
| `/v1/threads` | POST | WORKAROUND | local state + deferred `POST /chat` |
| `/v1/threads/{id}` | GET | NONE | local state only |
| `/v1/threads/{id}` | POST | NONE | local state only |
| `/v1/threads/{id}` | DELETE | DIRECT | `DELETE /assistant/openai/thread/delete` |
| `/v1/threads/{id}/messages` | POST | WORKAROUND | buffer; submit on run |
| `/v1/threads/{id}/messages` | GET | NONE | local state only |
| `/v1/threads/{id}/messages/{msg_id}` | GET | NONE | local state only |
| `/v1/threads/{id}/runs` | POST | WORKAROUND | `POST /chat` or `POST /assistant/chat/codeinterpreter` |
| `/v1/threads/{id}/runs/{run_id}` | GET | WORKAROUND | local state (sync result) |
| `/v1/threads/{id}/runs/{run_id}/cancel` | POST | NONE | — |
| `/v1/threads/{id}/runs` | GET | NONE | local state only |
| `/v1/threads/{id}/runs/{run_id}/submit_tool_outputs` | POST | NONE | — |
| `/v1/threads/runs` | POST | WORKAROUND | `POST /chat` or `POST /assistant/chat/codeinterpreter` |
| `/v1/threads/{id}/runs/{run_id}/steps` | GET | NONE | local state only |
| `/v1/threads/{id}/runs/{run_id}/steps/{step_id}` | GET | NONE | local state only |
| `/v1/embeddings` | POST | WORKAROUND | `POST /embedding-dual-retrieval` (semantic, not vector) |
| `/v1/audio/speech` | POST | NONE | — |
| `/v1/audio/transcriptions` | POST | NONE | — |
| `/v1/audio/translations` | POST | NONE | — |
| `/v1/images/generations` | POST | NONE | — |
| `/v1/images/edits` | POST | NONE | — |
| `/v1/images/variations` | POST | NONE | — |
| `/v1/fine_tuning/jobs` | POST | NONE | — |
| `/v1/fine_tuning/jobs` | GET | NONE | — |
| `/v1/fine_tuning/jobs/{id}` | GET | NONE | — |
| `/v1/fine_tuning/jobs/{id}/cancel` | POST | NONE | — |
| `/v1/fine_tuning/jobs/{id}/events` | GET | NONE | — |
| `/v1/moderations` | POST | NONE | — |
| `/v1/batches` | POST | NONE | — |
| `/v1/batches/{id}` | GET | NONE | — |
| `/v1/batches/{id}/cancel` | POST | NONE | — |
| `/v1/batches` | GET | NONE | — |
| `/v1/vector_stores` | POST | PARTIAL | tags as virtual stores |
| `/v1/vector_stores/{id}` | GET | WORKAROUND | `GET /files/tags/list` + `POST /files/query` |
| `/v1/vector_stores/{id}` | POST | NONE | — |
| `/v1/vector_stores/{id}` | DELETE | PARTIAL | `POST /files/tags/delete` |
| `/v1/vector_stores/{id}/files` | GET | WORKAROUND | `POST /files/query` (tag filter) |
| `/v1/vector_stores/{id}/files` | POST | WORKAROUND | `POST /files/set_tags` |
| `/v1/vector_stores/{id}/files/{file_id}` | DELETE | NONE | — |
| `/v1/vector_stores/{id}/file_batches` | POST | NONE | — |
| `/v1/vector_stores/{id}/file_batches/{id}` | GET | NONE | — |
| `/v1/vector_stores/{id}/file_batches/{id}/cancel` | POST | NONE | — |
| `/v1/vector_stores/{id}/file_batches/{id}/files` | GET | NONE | — |

---

## Amplify-specific Endpoints with No OpenAI Equivalent

These Amplify endpoints expose functionality not present in the OpenAI API. They can be
exposed as extension endpoints or used internally by the adapter.

| Amplify Endpoint | Method | Purpose |
|------------------|--------|---------|
| `/files/tags/list` | GET | List tags (virtual stores / labels) |
| `/files/tags/create` | POST | Create new tags |
| `/files/tags/delete` | POST | Delete a tag |
| `/files/set_tags` | POST | Associate tags with a file |
| `/embedding-dual-retrieval` | POST | Semantic search over knowledge base |
| `/state/share` | GET | List items shared with the user |
| `/state/share/load` | POST | Load a specific shared item by key |
| `/assistant/share` | POST | Share an assistant with other users |
| `/assistant/create/codeinterpreter` | POST | Create code interpreter assistant |
| `/assistant/chat/codeinterpreter` | POST | Chat with code interpreter assistant |
| `/assistant/files/download/codeinterpreter` | POST | Download code interpreter output files |
| `/assistant/openai/delete` | DELETE | Delete code interpreter assistant (OpenAI object) |

---

*Sources: `docs/amplify_api_reference.md`, `docs/amplify_api_probed.md`*
*Generated: 2026-03-03*
