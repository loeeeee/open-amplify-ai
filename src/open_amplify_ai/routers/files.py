"""Files endpoints mapped to the Amplify API."""
import base64
import json
import logging
import time
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.types import AmplifyFileUploadRequest, AmplifyKeyRequest
from open_amplify_ai.utils import amplify_item_to_openai_file, query_amplify_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/files", tags=["Files"])


@router.get("")
async def list_files(headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    List all files via Amplify POST /files/query.

    Maps Amplify file records to the OpenAI File object shape.
    """
    logger.info("Listing files")
    try:
        items = query_amplify_files(headers)
        return {
            "object": "list",
            "data": [amplify_item_to_openai_file(item) for item in items],
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error listing files: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.post("")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
    headers: dict = Depends(get_amplify_headers),
) -> Dict[str, Any]:
    """
    Upload a file using Amplify's two-step process.

    Step 1: POST /files/upload to get a pre-signed S3 URL.
    Step 2: PUT the file binary to that URL.
    Returns an OpenAI File object using the Amplify key as the ID.
    """
    logger.info("Uploading file: %s (purpose=%s)", file.filename, purpose)
    content_type = file.content_type or "application/octet-stream"
    file_bytes = await file.read()

    upload_payload: AmplifyFileUploadRequest = {
        "data": {
            "type": content_type,
            "name": file.filename or "upload",
            "knowledgeBase": "default",
            "tags": [],
            "data": {},
        }
    }

    try:
        init_resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/upload",
            headers=headers,
            json=upload_payload,
            timeout=30,
        )
        init_resp.raise_for_status()
        init_data = init_resp.json()

        if not init_data.get("success"):
            raise HTTPException(status_code=500, detail="Amplify file upload init failed")

        upload_url: str = init_data["uploadUrl"]
        file_key: str = init_data["key"]

        # PUT the binary directly to S3 (no auth header, pre-signed URL)
        s3_resp = requests.put(
            upload_url,
            data=file_bytes,
            headers={"Content-Type": content_type},
            timeout=60,
        )
        s3_resp.raise_for_status()

        return {
            "id": file_key,
            "object": "file",
            "bytes": len(file_bytes),
            "created_at": int(time.time()),
            "filename": file.filename or "upload",
            "purpose": purpose,
        }
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error uploading file: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.get("/{file_id:path}/content")
async def retrieve_file_content(
    file_id: str, headers: dict = Depends(get_amplify_headers)
) -> Any:
    """
    Download file content via Amplify POST /assistant/files/download/codeinterpreter.

    Only supported for Code Interpreter files. Returns a redirect or raw bytes.
    This route MUST be registered before the bare GET /v1/files/{file_id:path} route
    to prevent the greedy :path wildcard from swallowing the /content suffix.
    """
    logger.info("Retrieving file content: %s", file_id)
    payload: AmplifyKeyRequest = {"data": {"key": file_id}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/files/download/codeinterpreter",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        download_url = data.get("downloadUrl", "")
        if not download_url:
            raise HTTPException(status_code=404, detail="File content not available or file not found")
        # Proxy the actual binary content
        content_resp = requests.get(download_url, timeout=60)
        content_resp.raise_for_status()
        return StreamingResponse(
            iter([content_resp.content]),
            media_type=content_resp.headers.get("Content-Type", "application/octet-stream"),
        )
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error retrieving file content %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.get("/{file_id:path}")
async def retrieve_file(file_id: str, headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    Retrieve a single file record by ID.

    Amplify has no per-file endpoint. Fetches the full list and filters by id.
    """
    logger.info("Retrieving file: %s", file_id)
    try:
        items = query_amplify_files(headers)
        match = next((item for item in items if item.get("id") == file_id), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
        return amplify_item_to_openai_file(match)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error retrieving file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")



@router.delete("/{file_id:path}")
async def delete_file(file_id: str, headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    Translate OpenAI DELETE /v1/files/{file_id} to Amplify POST /files op=/delete.

    The Amplify file delete uses a generic action dispatch pattern observed in the web UI:
      POST /files
      body: {
        "data": base64({"key": file_id}),
        "method": "POST",
        "op": "/delete",
        "path": "/files",
        "service": "file"
      }

    The file_id must be the Amplify key returned by POST /files/upload,
    e.g. "email@vanderbilt.edu/2024-07-15/uuid.json".
    """
    logger.info("Deleting file: %s", file_id)
    encoded_data = base64.b64encode(json.dumps({"key": file_id}).encode()).decode()
    amplify_body = {
        "data": encoded_data,
        "method": "POST",
        "op": "/delete",
        "path": "/files",
        "service": "file",
    }
    try:
        response = requests.post(
            f"{AMPLIFY_BASE_URL}/files",
            headers=headers,
            json=amplify_body,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        deleted = bool(data.get("success", False))
        return {
            "id": file_id,
            "object": "file",
            "deleted": deleted,
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error deleting file %s: %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")
