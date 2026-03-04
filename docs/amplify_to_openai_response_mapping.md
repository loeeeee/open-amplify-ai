# Amplify AI to OpenAI Response Mapping

This document describes how the responses from the in-house **Amplify AI API** are parsed and formatted into **OpenAI API compatible responses** by the `server.py` application.

## 1. Chat Completions

### Non-Streaming (`POST /v1/chat/completions`)

The endpoint forwards requests to Amplify's `POST /chat`.

- **Success Responses**: The application expects the actual text response to either be within the JSON path `data.data` or just the raw HTTP text (`response.text`).
- **Tool Calling Context**:
  - The application attempts to parse the content as a tool call. If the content string (after stripping whitespaces) begins with `{"command"`, it parses the string.
  - The Amplify `"command"` field maps to OpenAI's tool call `function.name`.
  - The Amplify `"parameters"` field maps to OpenAI's tool call `function.arguments` as a JSON-encoded string.
  - A random `call_UUID` is generated for the tool call `id`.
- **Finish Reasons**:
  - If a valid tool call is found, `"finish_reason"` is set to `"tool_calls"`.
  - Otherwise, `"finish_reason"` defaults to `"stop"`.

### Streaming (`POST /v1/chat/completions` with `stream: true`)

The endpoint maintains a persistent HTTP connection to Amplify's streaming `/chat` endpoint.

- **Data Parsing**: Streams are handled line-by-line. If a line begins with `data: `, the payload is extracted. It ignores empty lines or lines without the prefix unless the non-prefixed text itself contains text content.
- **Content Fields**: If the `.data` payload is JSON, the application checks for `"data"`, `"content"`, or `"message"` fields. If JSON decoding fails, the raw delta string is yielded.
- **Tool Calling in Streams**:
  - The stream also checks if a generated delta block resembles a json-encoded tool call (`{"command"`).
  - If it matches, the payload is bundled into the `"tool_calls"` schema block of the chunk.
- **Completion Check**:
  - The stream explicitly yields `[DONE]` whenever Amplify outputs `data: [DONE]`.
  - The final chunk explicitly asserts `"finish_reason": "stop"`.
- **Usage Context**: If `stream_options: {"include_usage": true}` is present in the initial client request, a final `"usage"` chunk is appended with all zero counts.

## 2. Models

### Listing Models (`GET /v1/models`)

The application queries Amplify's `GET /available_models` and maps its values into the `Model` object shape.

- `id`: Passed directly.
- `object`: Set universally to `"model"`.
- `created`: Populated using the current integer UNIX timestamp.
- `owned_by`: Hardcoded to `"amplify-ai"`.

## 3. Files

### Listing/Retrieving Files (`GET /v1/files`)

The `server.py` queries Amplify's GraphQL-styled `POST /files/query` using pagination logic and lists available files as an OpenAI File schema array.

- **`bytes`**: Amplify does not natively supply bytes count. The proxy retrieves the `totalTokens` field and estimates by multiplying by 4.
- **`created_at`**: It parses Amplify's `createdAt` ISO string and generates an integer UNIX timestamp.
- **`purpose`**: Always defaulted strictly to `"assistants"`.
- **`id` and `filename`**: Directly linked to `id` and `name`.

### Uploading Files (`POST /v1/files`)

Mapped to a two-step process in Amplify (creating pre-signed URL + PUT S3 upload).
- The returned file inherits the server's time for `created_at`.
- Evaluates the incoming file's bytes length to return `bytes`.
- The AWS return `key` is used as the file's final `id`.
