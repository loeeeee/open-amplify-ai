# Amplify AI API Probed Reference

Verified API surface based on live probes. Conflict variants are shown with notes.

## GET /available_models
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "name": "Claude 4 Sonnet",
        "description": "\"Claude is a family of state-of-the-art large language models developed by Anthropic. Claude Sonnet 4 is our high-performance model with exceptional reasoning and efficiency\"",
        "inputContextWindow": 200000,
        "outputTokenLimit": 64000,
        "supportsImages": true,
        "supportsReasoning": true,
        "provider": "Bedrock",
        "supportsSystemPrompts": true,
        "systemPrompt": "",
       
... [truncated]
```

## POST /chat (Canonical (PDF/JSON): model as object inside options only)
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "message": "Chat endpoint response retrieved",
  "data": "The capital of France is Paris."
}
```

## POST /chat (Conflict (CSV): model also as top-level string field)
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "message": "Chat endpoint response retrieved",
  "data": "The capital of France is Paris."
}
```

## GET /state/share
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "items": [
    {
      "note": "test",
      "sharedAt": 1759269979023,
      "key": "<user-email>/<user-email>/2025-09-30/e80e1164-5ae6-4d7c-b917-97e812aa13bf.json",
      "sharedBy": "<user-email>"
    }
  ]
}
```

## POST /state/share/load
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": false,
  "message": "Data not found"
}
```

## POST /files/upload (Canonical (PDF/JSON): no actions field)
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "uploadUrl": "REDACTED_PRESIGNED_S3_URL
... [truncated]
```

## POST /files/upload (Conflict (CSV): includes actions array in body)
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "uploadUrl": "REDACTED_PRESIGNED_S3_URL
... [truncated]
```

## POST /files/query
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "knowledgeBase": "default",
        "data": {},
        "updatedAt": "2026-03-03T20:24:06.197209",
        "updatedBy": "<user-email>",
        "id": "<user-email>/2026-03-03/feb009ea-0602-4d3c-b139-2cfab2c55a59.json",
        "createdBy": "<user-email>",
        "name": "probe-test-file.pdf",
        "tags": [],
        "type": "application/pdf",
        "createdAt": "2026-03-03T20:24:06.197197"
      },
      {
        "knowledgeBase": "default",
        "data": {},
        "updatedAt": "2026-03-03T20:24:05.557125",
        "upd
... [truncated]
```

## POST /files (File delete dispatch (UI-observed): base64-encoded {key} in data field)
- **Status**: 403
- **Analysis**: FORBIDDEN.

```json
{
  "message": "Invalid key=value pair (missing equal-sign) in Authorization header (hashed with SHA-256 and encoded with Base64): 'c4c5o9/mNVBmKK65loEj+T+z+Apc7uaFjxfgzRycghs='."
}
```

## POST /files (File delete dispatch (alternative): plain JSON {key} in data field)
- **Status**: 403
- **Analysis**: FORBIDDEN.

```json
{
  "message": "Invalid key=value pair (missing equal-sign) in Authorization header (hashed with SHA-256 and encoded with Base64): 'c4c5o9/mNVBmKK65loEj+T+z+Apc7uaFjxfgzRycghs='."
}
```

## GET /files/tags/list (Canonical (PDF/JSON): GET method)
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "data": {
    "tags": [
      "ci-test-1759260560",
      "to-review",
      "ci-test-1759208241",
      "ci-test-1759210550",
      "ci-test-1759209622",
      "ci-test-1759264496",
      "api-test",
      "ci-test-1759257231",
      "ci-test-1759265823",
      "ci-test-1759207450",
      "ci-test-1759260495",
      "ci-test-1759254244",
      "ci-test-1759266367",
      "ci-test-1759253415",
      "ci-test-1759209218",
      "ci-test-1759207567",
      "ci-test-1759254093",
      "ci-test-1759265793",
      "ci-test-1759260417",
      "ci-test-1759207772",
      "ci-te
... [truncated]
```

## POST /files/tags/list (Conflict (CSV): POST method)
- **Status**: 403
- **Analysis**: FORBIDDEN.

```json
{
  "message": "Invalid key=value pair (missing equal-sign) in Authorization header (hashed with SHA-256 and encoded with Base64): 'c4c5o9/mNVBmKK65loEj+T+z+Apc7uaFjxfgzRycghs='."
}
```

## POST /files/tags/create
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "message": "Tags added successfully"
}
```

## POST /files/tags/delete
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "message": "Tag deleted successfully"
}
```

## POST /files/set_tags
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": false,
  "message": "File not found or not authorized to update tags"
}
```

## POST /embedding-dual-retrieval
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "error": "No accessible data sources. User may not have permissions to the provided sources.",
  "details": {
    "total_sources_requested": 1,
    "accessible_sources": 0
  }
}
```

## POST /assistant/create
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "message": "Assistant created successfully",
  "data": {
    "assistantId": "astp/0c274834-a9fa-4519-b197-ba67399f28e1",
    "id": "ast/aac3cac5-7932-481c-b595-18df6dddbb8d",
    "version": 1,
    "data_sources": [],
    "ast_data": {
      "websiteUrls": [],
      "integrationDriveData": {}
    }
  }
}
```

## GET /assistant/list
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "message": "Assistants retrieved successfully",
  "data": [
    {
      "assistantId": "astp/4d99da5c-1e88-42ad-b170-053b9be81e3d",
      "instructions": "Respond to user queries about general knowledge topics. Be helpful and concise.",
      "disclaimerHash": "cdaeda8acecdb2d259ae6a3ef1013e217ffec018bd129539e6ebfd727c4c1425",
      "coreHash": "0471bea0194beaac602a737d47e2380b78a1b72a9a0270896c36a1cfa9000c51",
      "user": "<user-email>",
      "uri": null,
      "createdAt": "2025-09-30T22:33:11",
      "dataSources": [],
      "name": "General Assistant (API Test)",

... [truncated]
```

## POST /assistant/share
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": false,
  "message": "Assistant not found"
}
```

## POST /assistant/delete
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": false,
  "message": "You are not authorized to delete this assistant."
}
```

## POST /assistant/create/codeinterpreter
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": true,
  "message": "Assistant created successfully",
  "data": {
    "assistantId": "<user-email>/ast/a7b9d81a-6871-417e-a03b-e5d89c196462"
  }
}
```

## POST /assistant/files/download/codeinterpreter
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": false,
  "message": "File not found"
}
```

## DELETE /assistant/openai/thread/delete
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": false,
  "message": "Invalid or missing thread id parameter"
}
```

## DELETE /assistant/openai/delete
- **Status**: 200
- **Analysis**: WORKING.

```json
{
  "success": false,
  "message": "Invalid or missing assistant id parameter"
}
```

## POST /assistant/chat/codeinterpreter
- **Status**: 400
- **Analysis**: FAILED (400).

```json
{
  "error": "Error: 400 - Invalid data or path"
}
```

