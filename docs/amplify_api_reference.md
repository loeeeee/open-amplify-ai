# Amplify AI API Reference

This document is synthesized from three source files: `amplify-api-docs.csv`, `amplify-api-docs.pdf`,
and `amplify-api-docs.json` (Postman collection). Where the sources conflict, each conflict is
explicitly described in the endpoint's **Conflicts** section. The most conservative or complete
interpretation is used for the canonical definition.

## Base URL

```
https://prod-api.vanderbilt.ai
```

## Authentication

All endpoints require a Bearer token in the `Authorization` header.

```
Authorization: Bearer <amp-token>
```

---

## Conflict Summary

| # | Endpoint | Conflict |
|---|----------|----------|
| 1 | `/files/tags/list` | HTTP method: `POST` (CSV) vs `GET` (PDF, JSON) |
| 2 | `/files/upload` | Request body: CSV requires `actions` array; PDF and JSON examples omit it |
| 3 | `/chat` | `model` parameter location: CSV shows it as top-level string AND in `options`; PDF/JSON show it only inside `options` as an object |
| 4 | `/embedding-dual-retrieval` | JSON Postman entry is named `embedding-dual-retrieval` (missing leading `/`) |
| 5 | `/assistant/files/download/codeinterpreter` | JSON Postman entry is named `assistant/files/download/codeinterpreter` (missing leading `/`); PDF lists `downloadUrl` type as `boolean` instead of `string` |
| 6 | `/assistant/create/codeinterpreter` | PDF response schema has `assitantId` (typo) and `message (boolean)` (wrong type); CSV/JSON correctly use `assistantId` and `message (string)` |

---

## Endpoints

### 1. `GET /available_models`

**Summary**: View a list of available models, including model ID, name, description, context window
size, output token limit, provider, image support, system prompt support, and any additional system
prompts Amplify appends.

**Request**: No request body required.

**Response `200`**:
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": "gpt-4o-mini",
        "name": "GPT-4o-mini",
        "description": "...",
        "inputContextWindow": 128000,
        "outputTokenLimit": 16384,
        "supportsImages": true,
        "provider": "Azure",
        "supportsSystemPrompts": true,
        "systemPrompt": "Additional prompt"
      }
    ],
    "default": "<model object>",
    "advanced": "<model object>",
    "cheapest": "<model object>"
  }
}
```

**Conflicts**: None.

---

### 2. `POST /chat`

**Summary**: Real-time chat via streaming. Supports GPT, Claude, Mistral, and other advanced models.
Responses are delivered as a stream.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | No (see conflicts) | Model ID. Get IDs from `/available_models`. |
| `temperature` | number | No | Sampling temperature. |
| `max_tokens` | integer | No | Maximum tokens in response. |
| `dataSources` | array of strings | No | Data source IDs. Obtain from `/files/query` or `/files/upload`. |
| `messages` | array of objects | Yes | Chat messages. Each has `role` (system/assistant/user) and `content` (string). |
| `type` | string | No | Options include `prompt`. |
| `options` | object | No | See sub-fields below. |
| `options.dataSourceOptions` | object | No | Options for data source behavior. |
| `options.ragOnly` | boolean | No | Use only RAG results. |
| `options.skipRag` | boolean | No | Skip RAG entirely. |
| `options.assistantId` | string | No | Associate a specific assistant. |
| `options.model` | object | No | Model selection: `{ "id": "<model_id>" }`. |
| `options.prompt` | string | No | Override prompt text. |

**Example Request Body**:
```json
{
  "data": {
    "temperature": 0.7,
    "max_tokens": 4000,
    "dataSources": ["user@vanderbilt.edu/2014-qwertyuio"],
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "options": {
      "ragOnly": false,
      "skipRag": true,
      "model": { "id": "gpt-4o" },
      "assistantId": "astp/abcdefghi",
      "prompt": "What is the capital of France?"
    }
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Chat completed successfully",
  "data": "The capital of France is Paris."
}
```

**Other responses**: `400` (invalid/missing fields), `401` (unauthorized), `403` (forbidden), `404` (not found).

**Conflicts**:
- **`model` parameter location**: CSV shows `model` as a required top-level string in the request body AND as an object inside `options`. The PDF and JSON (Postman) consistently show `model` only inside `options` as an object `{ "id": "<model_id>" }`. The PDF/JSON representation is preferred.

---

### 3. `GET /state/share`

**Summary**: View a list of shared resources (assistants, conversations, organizational folders)
distributed by other Amplify platform users.

**Request**: No request body required.

**Response `200`**:
```json
[
  {
    "note": "testing share with a doc",
    "sharedAt": 1720714099836,
    "key": "yourEmail@vanderbilt.edu/sharedByEmail@vanderbilt.edu/932804035837948202934805-24382.json",
    "sharedBy": "sharedByEmail@vanderbilt.edu"
  }
]
```

**Conflicts**: None.

---

### 4. `POST /state/share/load`

**Summary**: Retrieve and examine an individual shared element using the unique identifier key from
`/state/share`.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | string | Yes | The unique key from the shared resource listing. |

**Example Request Body**:
```json
{
  "data": {
    "key": "yourEmail@vanderbilt.edu/sharedByEmail@vanderbilt.edu/932804035837948202934805-24382.json"
  }
}
```

**Response `200`** - Returns the export schema, for example:
```json
{
  "version": 1,
  "prompts": [
    {
      "id": "ast/7b32fc3f-fe93-4026-b358-0e286e4a6013",
      "name": "api share test",
      "content": "<content>",
      "data": {
        "assistant": {
          "definition": {
            "instructions": "<instructions>",
            "user": "sampleUser@vanderbilt.edu",
            "dataSources": [{ "name": "car_sales.csv" }],
            "name": "api share test"
          }
        },
        "provider": "amplify",
        "noCopy": true,
        "noEdit": true,
        "noDelete": true,
        "noShare": true
      }
    }
  ]
}
```

**Other responses**: `401` (no access to share functionality), `404` (data not found).

**Conflicts**: None.

---

### 5. `POST /files/upload`

**Summary**: Initiate a file upload. Returns a time-limited pre-authenticated URL for secure file
transmission.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `actions` | array of objects | Yes (CSV) / Omitted (PDF, JSON) | List of processing actions. See conflict note. Each object has `name` (string) and `params` (object). |
| `type` | string | Yes | MIME type of the file (e.g., `application/pdf`). |
| `name` | string | Yes | File name. |
| `knowledgeBase` | string | Yes | Knowledge base to store the file in (e.g., `"default"`). |
| `tags` | array of strings | Yes | Tags to associate with the file. |
| `data` | object | Yes | Additional metadata. |

**Example Request Body (CSV version, full)**:
```json
{
  "data": {
    "actions": [
      { "name": "saveAsData", "params": {} },
      { "name": "createChunks", "params": {} },
      { "name": "ingestRag", "params": {} },
      { "name": "makeDownloadable", "params": {} },
      { "name": "extractText", "params": {} }
    ],
    "type": "application/pdf",
    "name": "fileName.pdf",
    "knowledgeBase": "default",
    "tags": [],
    "data": {}
  }
}
```

**Example Request Body (PDF/JSON version)**:
```json
{
  "data": {
    "type": "application/fileExtension",
    "name": "fileName.pdf",
    "knowledgeBase": "default",
    "tags": [],
    "data": {}
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "uploadUrl": "<uploadUrl>",
  "statusUrl": "<statusUrl>",
  "contentUrl": "<contentUrl>",
  "metadataUrl": "<metadataUrl>",
  "key": "yourEmail@vanderbilt.edu/date/2930497329-490823.json"
}
```

**Other responses**: `400` (bad request), `401` (unauthorized).

**Conflicts**:
- **`actions` field**: CSV marks `actions` (array of objects) as **Required** in the request body and lists valid action names (`saveAsData`, `createChunks`, `ingestRag`, `makeDownloadable`, `extractText`). The PDF and JSON (Postman) example bodies both omit the `actions` array entirely. It is unclear whether `actions` is truly required or optional. Treat as optional but recommended.

---

### 6. `POST /files/query`

**Summary**: List uploaded Amplify data sources with filtering and pagination.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `startDate` | string (date-time) | No | `2021-01-01T00:00:00Z` | Filter by creation date. |
| `pageSize` | integer | No | `10` | Number of results per page. |
| `pageKey` | object | No | - | Pagination key: `{ id, createdAt, type }`. |
| `namePrefix` | string or null | No | - | Filter by name prefix. |
| `createdAtPrefix` | string or null | No | - | Filter by creation date prefix. |
| `typePrefix` | string or null | No | - | Filter by type prefix. |
| `types` | array of strings | No | `[]` | Filter by file types. |
| `tags` | array of strings | No | `[]` | Filter by tags. |
| `pageIndex` | integer | No | `0` | Page index for pagination. |
| `forwardScan` | boolean | No | `false` | Scan direction. |
| `sortIndex` | string | No | `createdAt` | Field to sort by. |

**Example Request Body**:
```json
{
  "data": {
    "pageSize": 10,
    "sortIndex": "",
    "forwardScan": false
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "createdAt": "2024-07-15T17:12:45.046682",
        "updatedBy": "yourEmail@vanderbilt.edu",
        "createdBy": "yourEmail@vanderbilt.edu",
        "name": "fileName.doc",
        "knowledgeBase": "default",
        "data": {},
        "updatedAt": "2024-07-15T17:12:45.046700",
        "totalTokens": 12644,
        "dochash": "25ef6a3e472d8d90a3784d1df9abe0ae390cf4da2c2f9a4f82d91cbe501915c1",
        "id": "yourEmail@vanderbilt.edu/date/238904934298030943.json",
        "tags": [],
        "totalItems": 1025,
        "type": "application/fileExtension"
      }
    ],
    "pageKey": "<next page key object>"
  }
}
```

**Conflicts**: None.

---

### 6a. `POST /files` (action dispatch: `op=/delete`)

**Summary**: Delete a file from the Amplify knowledge base. Discovered from web UI network traffic.
This endpoint uses a generic action dispatch pattern rather than a dedicated REST verb.

**Request Body** (top-level JSON object, not `data`-wrapped in the usual sense):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | string | Yes | Base64-encoded JSON `{"key": file_key}` where `file_key` is the Amplify file key. |
| `method` | string | Yes | Always `"POST"`. |
| `op` | string | Yes | Always `"/delete"`. |
| `path` | string | Yes | Always `"/files"`. |
| `service` | string | Yes | Always `"file"`. |

**Example Request Body**:
```json
{
  "data": "eyJrZXkiOiJ5b3VyRW1haWxAdmFuZGVyYmlsdC5lZHUvMjAyNi0wMS0wMS91dWlkLmpzb24ifQ==",
  "method": "POST",
  "op": "/delete",
  "path": "/files",
  "service": "file"
}
```

The base64 string decodes to:
```json
{"key": "yourEmail@vanderbilt.edu/2026-01-01/uuid.json"}
```

**Response `200`** (expected):
```json
{
  "success": true
}
```

**Source**: Observed in web UI network traffic on 2026-03-03. Not in official documentation.

**Conflicts**: A plain-JSON `data` object variant (without base64 encoding) may also work. Probe
both forms to determine the canonical format.

---

### 7. `GET /files/tags/list`

**Summary**: View a list of your Amplify tags that can be tied to data sources, conversations, and
assistants.

**Request**: No request body required.

**Response `200`**:
```json
{
  "success": true,
  "data": {
    "tags": ["NewTag"]
  }
}
```

**Conflicts**:
- **HTTP Method**: CSV lists this endpoint as `POST`. PDF and JSON (Postman) both list it as `GET`. The `GET` method is more semantically correct for a list operation and is supported by two sources. **Use `GET`.**

---

### 8. `POST /files/tags/create`

**Summary**: Create new Amplify tags that can be tied to data sources, conversations, and assistants.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tags` | array of strings | No (default `[]`) | Tags to create. |

**Example Request Body**:
```json
{
  "data": {
    "tags": ["NewTag"]
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Tags added successfully"
}
```

**Conflicts**: None.

---

### 9. `POST /files/tags/delete`

**Summary**: Delete an Amplify tag.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tag` | string | Yes | The tag to delete. |

**Example Request Body**:
```json
{
  "data": {
    "tag": "NewTag"
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Tag deleted successfully"
}
```

**Conflicts**: None.

---

### 10. `POST /files/set_tags`

**Summary**: Associate an Amplify tag with a specific data source, conversation, or assistant.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | The ID of the resource to tag. |
| `tags` | array of strings | No (default `[]`) | Tags to associate. |

**Example Request Body**:
```json
{
  "data": {
    "id": "yourEmail@vanderbilt.edu/date/23094023573924890-208.json",
    "tags": ["NewTag"]
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Tags updated and added to user"
}
```

**Conflicts**: None.

---

### 11. `POST /embedding-dual-retrieval`

**Summary**: Retrieve Amplify data source embeddings based on user input using dual retrieval.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `userInput` | string | Yes | The query input from the user. |
| `dataSources` | array of strings | Yes | Data source IDs to search (use `global/` prefix IDs). |
| `limit` | integer | No | Maximum number of results to return. |

**Example Request Body**:
```json
{
  "data": {
    "userInput": "Can you describe the policies outlined in the document?",
    "dataSources": ["global/09342587234089234890.content.json"],
    "limit": 10
  }
}
```

**Response `200`**:
```json
{
  "result": [
    {
      "content": "xmlns:w=3D'urn:schemas-microsoft-com:office:word' ...",
      "file": "global/2405939845893094580341.content.json",
      "line_numbers": [15, 30],
      "score": 0.7489801645278931
    }
  ]
}
```

**Conflicts**:
- **Postman entry name**: The JSON (Postman) collection names this item `embedding-dual-retrieval` (missing the leading `/`). The actual URL in the Postman entry correctly uses `/embedding-dual-retrieval`. This is a display-name inconsistency only, not a functional difference.
- **Response key typo**: The CSV example response uses `"rH12esult"` instead of `"result"`. This appears to be a typo in the CSV. PDF and JSON examples use `"result"` as the key. **Use `result`.**

---

### 12. `POST /assistant/create`

**Summary**: Create or update a customizable Amplify assistant. If `assistantId` is provided and
corresponds to an existing assistant, it will be updated.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Assistant name. |
| `description` | string | Yes | Assistant description. |
| `assistantId` | string | No | Required only when updating an existing assistant. |
| `tags` | array of strings | Yes | Tags for the assistant. |
| `instructions` | string | Yes | System instructions for the assistant. |
| `disclaimer` | string | No | Disclaimer text. |
| `uri` | string or null | No | Optional URI. |
| `dataSources` | array of objects | Yes | Full data source objects to attach. |
| `dataSourceOptions` | array of objects | No | Options controlling data source usage in prompts/RAG. |

**`dataSourceOptions` sub-fields**:

| Field | Type | Description |
|-------|------|-------------|
| `insertAttachedDocumentsMetadata` | boolean | Include attached data source metadata in prompt. |
| `insertAttachedDocuments` | boolean | Include attached documents in prompt. |
| `insertConversationDocuments` | boolean | Include conversation documents in prompt. |
| `disableDataSources` | boolean | Disable data source insertion. |
| `insertConversationDocumentsMetadata` | boolean | Include conversation data source metadata in prompt. |
| `ragConversationDocuments` | boolean | Include conversation documents in RAG. |
| `ragAttachedDocuments` | boolean | Include attached documents in RAG. |

**Example Request Body**:
```json
{
  "data": {
    "name": "Sample Assistant 3",
    "description": "This is a sample assistant for demonstration purposes",
    "assistantId": "",
    "tags": ["test"],
    "instructions": "Respond to user queries about general knowledge topics",
    "disclaimer": "This assistant's responses are for informational purposes only",
    "dataSources": [
      {
        "id": "e48759073324384kjsf",
        "name": "api_paths_summary.csv",
        "type": "text/csv",
        "raw": "",
        "data": "",
        "key": "yourEmail@vanderbilt.edu/date/w3ou009we3.json",
        "metadata": {
          "name": "api_paths_summary.csv",
          "totalItems": 20,
          "locationProperties": ["row_number"],
          "contentKey": "yourEmail@vanderbilt.edu/date/w3ou009we3.json.content.json",
          "createdAt": "2024-07-15T18:58:24.912235",
          "totalTokens": 3750,
          "tags": [],
          "props": {}
        }
      }
    ],
    "tools": []
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Assistant created successfully",
  "data": {
    "assistantId": "astp/3io4u5ipy34jkelkdfweiorwur",
    "id": "ast/03uio3904583049859482",
    "version": 1
  }
}
```

**Conflicts**: None.

---

### 13. `GET /assistant/list`

**Summary**: Retrieve a list of all Amplify assistants owned by the authenticated user.

**Request**: No request body required.

**Response `200`**:
```json
{
  "success": true,
  "message": "Assistants retrieved successfully",
  "data": [
    {
      "assistantId": "astp/498370528-38594",
      "version": 3,
      "instructions": "<instructions>",
      "disclaimerHash": "348529340098580234959824580-pueiorupo4",
      "coreHash": "eiouqent84832n8989pdeer",
      "user": "yourEmail@vanderbilt.edu",
      "uri": null,
      "createdAt": "2024-07-15T19:07:57",
      "dataSources": [
        {
          "metadata": "<metadata>",
          "data": "",
          "name": "api_documentation.yml",
          "raw": "",
          "id": "global/7834905723785897982345088927.content.json",
          "type": "application/x-yaml"
        }
      ]
    }
  ]
}
```

**Conflicts**: None.

---

### 14. `POST /assistant/share`

**Summary**: Share an Amplify assistant with other Amplify platform users.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `assistantId` | string | Yes | The ID of the assistant to share. |
| `recipientUsers` | array of strings | Yes | Email addresses of recipient users. |
| `note` | string | No | Optional message to recipients. |

**Example Request Body**:
```json
{
  "data": {
    "assistantId": "ast/8934572093982034020-9",
    "recipientUsers": ["yourEmail@vanderbilt.edu"],
    "note": "check this out!"
  }
}
```

**Response `200`** (failure example - sharing failed for some users):
```json
{
  "success": false,
  "message": "Unable to share with some users",
  "failedShares": ["yourEmail@vanderbilt.edu"]
}
```

**Conflicts**: None.

---

### 15. `POST /assistant/delete`

**Summary**: Delete an Amplify assistant.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `assistantId` | string | Yes | The ID of the assistant to delete. |

**Example Request Body**:
```json
{
  "data": {
    "assistantId": "astp/3209457834985793094"
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Assistant deleted successfully."
}
```

**Conflicts**: None.

---

### 16. `POST /assistant/create/codeinterpreter`

**Summary**: Create a new Code Interpreter Assistant backed by OpenAI's code interpreter tool.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Assistant name. |
| `description` | string | Yes | Assistant description. |
| `tags` | array of strings | Yes | Tags for the assistant. |
| `instructions` | string | Yes | System instructions. |
| `dataSources` | array of strings | Yes | File IDs starting with your email (from `/files/query`). |

**Example Request Body**:
```json
{
  "data": {
    "name": "Data Analysis Assistant",
    "description": "An AI assistant specialized in data analysis and visualization",
    "tags": ["data analysis"],
    "instructions": "Analyze data files, perform statistical operations, and create visualizations as requested by the user",
    "dataSources": ["yourEmail@vanderbilt.edu/2024-05-08/0f20f0447b.json"],
    "fileKeys": [],
    "tools": [{ "type": "code_interpreter" }]
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Assistant created successfully",
  "data": {
    "assistantId": "yourEmail@vanderbilt.edu/ast/39408573849029843",
    "provider": "<provider>"
  }
}
```

**Conflicts**:
- **`assistantId` typo in PDF**: The PDF response schema spells the returned field as `assitantId` (missing an `s`). CSV and JSON (Postman) correctly spell it `assistantId`. **Use `assistantId`.**
- **`message` type in PDF**: The PDF response schema shows `message (boolean)`, but the field is clearly a status string (e.g., `"Assistant created successfully"`). CSV and JSON correctly show `message (string)`. **Use `string`.**

---

### 17. `POST /assistant/files/download/codeinterpreter`

**Summary**: Download Code Interpreter generated files using a pre-authenticated URL.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | string | Yes | The file key (starts with your email or message ID). |
| `fileName` | string | No | If provided, will trigger automatic download of the specific file. |

**Example Request Body**:
```json
{
  "data": {
    "key": "yourEmail@vanderbilt.edu/ast/3498523804729"
  }
}
```

**Response `200`**:
```json
{
  "success": true,
  "downloadUrl": "<Download URL>"
}
```

**Conflicts**:
- **Postman entry name**: The JSON (Postman) collection names this item `assistant/files/download/codeinterpreter` (missing the leading `/`). The URL in the entry correctly uses `/assistant/files/download/codeinterpreter`. This is a display-name inconsistency only.
- **`downloadUrl` type in PDF**: The PDF response schema lists `downloadUrl (boolean)`. This is clearly a documentation error; it should be a string (URL). CSV and the example responses in all sources show it returning a URL string. **Use `string`.**

---

### 18. `DELETE /assistant/openai/thread/delete`

**Summary**: Delete an OpenAI thread, removing the existing conversation history with a Code
Interpreter assistant.

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `threadId` | string | Yes | The thread ID to delete (e.g., `yourEmail@vanderbilt.edu/thr/<id>`). |

**Example URL**:
```
DELETE /assistant/openai/thread/delete?threadId=yourEmail@vanderbilt.edu/thr/8923047385920349782093
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Thread deleted successfully"
}
```

**Conflicts**: None.

---

### 19. `DELETE /assistant/openai/delete`

**Summary**: Delete a Code Interpreter assistant instance (the OpenAI assistant object, not the
Amplify assistant record).

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `assistantId` | string | Yes | The assistant ID to delete (e.g., `yourEmail@vanderbilt.edu/ast/<id>`). |

**Example URL**:
```
DELETE /assistant/openai/delete?assistantId=yourEmail@vanderbilt.edu/ast/38940562397049823
```

**Response `200`**:
```json
{
  "success": true,
  "message": "Assistant deleted successfully"
}
```

**Conflicts**: None.

---

### 20. `POST /assistant/chat/codeinterpreter`

**Summary**: Establishes or continues a conversation with a Code Interpreter assistant. Returns a
unique thread identifier. Use the returned `threadId` in subsequent calls to append new messages
without re-sending the full conversation history.

**Request Body** (wrapped in `data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `assistantId` | string | Yes | The Code Interpreter assistant ID. |
| `threadId` | string | No | Existing thread ID to continue a conversation. |
| `messages` | array of objects | Yes | New messages to add. Each has `role`, `content`, and `dataSourceIds`. |
| `messages[].role` | string | Yes | `user` or `assistant`. |
| `messages[].content` | string | Yes | Message text. |
| `messages[].dataSourceIds` | array of strings | Yes | File IDs to attach (starts with your email). |

**Example Request Body**:
```json
{
  "data": {
    "assistantId": "yourEmail@vanderbilt.edu/ast/43985037429849290398",
    "messages": [
      {
        "role": "user",
        "content": "Can you tell me something about the data analytics and what you are able to do?",
        "dataSourceIds": ["yourEmail@vanderbilt.edu/2024-05-08/0f20f0447b.json"]
      }
    ]
  }
}
```

**Response `200`** (text-only):
```json
{
  "success": true,
  "message": "Chat completed successfully",
  "data": {
    "data": {
      "threadId": "yourEmail@vanderbilt.edu/thr/892345790239402934234",
      "role": "assistant",
      "textContent": "<response text>",
      "content": []
    }
  }
}
```

**Response `200`** (with generated file):
```json
{
  "success": true,
  "message": "Chat completed successfully",
  "data": {
    "threadId": "yourEmail@vanderbilt.edu/thr/442309eb-0772-42d0-b6ef-34e20ee2355e",
    "role": "assistant",
    "textContent": "I've saved the generated pie chart as a PNG file...",
    "content": [
      {
        "type": "image/png",
        "values": {
          "file_key": "yourEmail@vanderbilt.edu/msg_..._pie_chart.png",
          "presigned_url": "https://vu-amplify-assistants-dev-code-interpreter-files.s3.amazonaws.com/...",
          "file_key_low_res": "<low_res_key>",
          "presigned_url_low_res": "<low_res_url>",
          "file_size": 149878
        }
      }
    ]
  }
}
```

**Conflicts**: None.

---

*Generated from: `amplify-api-docs.csv`, `amplify-api-docs.pdf`, `amplify-api-docs.json`*
*Last updated: 2026-03-03*
