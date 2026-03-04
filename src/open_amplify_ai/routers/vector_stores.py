"""Vector stores endpoints mapped to the Amplify API."""
import logging
import time
import uuid
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.types import AmplifyTagsRequest
from open_amplify_ai.utils import not_implemented, query_amplify_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/vector_stores", tags=["Vector Stores"])


@router.post("")
async def create_vector_store(
    request: Request, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Create a virtual vector store backed by an Amplify tag.

    Amplify knowledge base + tagging approximates OpenAI's vector stores.
    A unique tag is created to represent the store.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    name = body.get("name", f"vs-{uuid.uuid4().hex[:8]}")
    tag_id = name
    logger.info("Creating vector store (tag): %s", tag_id)

    payload: AmplifyTagsRequest = {"data": {"tags": [tag_id]}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/tags/create",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return {
            "id": tag_id,
            "object": "vector_store",
            "created_at": int(time.time()),
            "name": name,
            "usage_bytes": 0,
            "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "status": "completed",
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error creating vector store %s: %s", tag_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.get("/{vector_store_id}")
async def retrieve_vector_store(
    vector_store_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Retrieve a vector store by ID.

    Confirms the backing tag exists and counts the associated files.
    """
    logger.info("Retrieving vector store: %s", vector_store_id)
    try:
        tags_resp = requests.get(
            f"{AMPLIFY_BASE_URL}/files/tags/list",
            headers=headers,
            timeout=30,
        )
        tags_resp.raise_for_status()
        tags = tags_resp.json().get("data", {}).get("tags", [])
        if vector_store_id not in tags:
            raise HTTPException(status_code=404, detail=f"Vector store '{vector_store_id}' not found")

        items = query_amplify_files(headers, tags=[vector_store_id])
        file_count = len(items)
        return {
            "id": vector_store_id,
            "object": "vector_store",
            "created_at": int(time.time()),
            "name": vector_store_id,
            "usage_bytes": sum(item.get("totalTokens", 0) * 4 for item in items),
            "file_counts": {
                "in_progress": 0,
                "completed": file_count,
                "failed": 0,
                "cancelled": 0,
                "total": file_count,
            },
            "status": "completed",
        }
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error retrieving vector store %s: %s", vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.post("/{vector_store_id}")
async def modify_vector_store(vector_store_id: str, request: Request) -> None:
    """Amplify has no tag rename endpoint; vector store modification is not supported."""
    raise not_implemented("Vector store modification")


@router.delete("/{vector_store_id}")
async def delete_vector_store(
    vector_store_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Delete a vector store by removing its backing tag via Amplify POST /files/tags/delete.

    Note: this removes the tag only; underlying files remain in the knowledge base.
    """
    logger.info("Deleting vector store (tag): %s", vector_store_id)
    payload: AmplifyTagsRequest = {"data": {"tag": vector_store_id}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/tags/delete",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": vector_store_id,
            "object": "vector_store.deleted",
            "deleted": bool(data.get("success", False)),
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error deleting vector store %s: %s", vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.get("/{vector_store_id}/files")
async def list_vector_store_files(
    vector_store_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """List all files in a vector store by querying Amplify files filtered by tag."""
    logger.info("Listing files in vector store: %s", vector_store_id)
    try:
        items = query_amplify_files(headers, tags=[vector_store_id])
        data = [
            {
                "id": item.get("id", ""),
                "object": "vector_store.file",
                "usage_bytes": item.get("totalTokens", 0) * 4,
                "created_at": 0,
                "vector_store_id": vector_store_id,
                "status": "completed",
            }
            for item in items
        ]
        return {"object": "list", "data": data}
    except requests.exceptions.RequestException as e:
        logger.error("Error listing vector store files for %s: %s", vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.post("/{vector_store_id}/files")
async def create_vector_store_file(
    vector_store_id: str,
    request: Request,
    headers: dict = Depends(get_amplify_headers),
) -> Dict[str, Any]:
    """
    Associate a file with a vector store by adding the store tag to the file.

    Uses Amplify POST /files/set_tags with the vector_store_id as a tag.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    file_id = body.get("file_id", "")
    logger.info("Adding file %s to vector store %s", file_id, vector_store_id)

    payload: AmplifyTagsRequest = {"data": {"id": file_id, "tags": [vector_store_id]}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/files/set_tags",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return {
            "id": file_id,
            "object": "vector_store.file",
            "usage_bytes": 0,
            "created_at": int(time.time()),
            "vector_store_id": vector_store_id,
            "status": "completed",
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error adding file %s to vector store %s: %s", file_id, vector_store_id, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.delete("/{vector_store_id}/files/{file_id:path}")
async def delete_vector_store_file(vector_store_id: str, file_id: str) -> None:
    """
    Amplify has no endpoint to remove a single tag from a single file.

    POST /files/set_tags replaces the entire tag list; partial removal via that
    endpoint requires first fetching the current tags, which is not exposed.
    """
    raise not_implemented("Vector store file removal")


# Vector store file batch endpoints (501 stubs) --------------------------------


@router.post("/{vector_store_id}/file_batches")
async def create_vector_store_file_batch(vector_store_id: str, request: Request) -> None:
    """Amplify has no async batch file ingestion endpoint."""
    raise not_implemented("Vector store file batch creation")


@router.get("/{vector_store_id}/file_batches/{batch_id}")
async def retrieve_vector_store_file_batch(vector_store_id: str, batch_id: str) -> None:
    """Amplify has no async batch file status endpoint."""
    raise not_implemented("Vector store file batch retrieval")


@router.post("/{vector_store_id}/file_batches/{batch_id}/cancel")
async def cancel_vector_store_file_batch(vector_store_id: str, batch_id: str) -> None:
    """Amplify has no async batch file status endpoint."""
    raise not_implemented("Vector store file batch cancellation")


@router.get("/{vector_store_id}/file_batches/{batch_id}/files")
async def list_vector_store_file_batch_files(vector_store_id: str, batch_id: str) -> None:
    """Amplify has no async batch file status endpoint."""
    raise not_implemented("Vector store file batch files listing")
