"""Assistants endpoints mapped to the Amplify API."""
import logging
import time
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.types import AmplifyAssistantCreateRequest
from open_amplify_ai.utils import amplify_assistant_to_openai, handle_upstream_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/assistants", tags=["Assistants"])


@router.get("")
async def list_assistants(headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """List all assistants via Amplify GET /assistant/list."""
    logger.info("Listing assistants")
    try:
        resp = requests.get(f"{AMPLIFY_BASE_URL}/assistant/list", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        assistants = data.get("data", [])
        openai_assistants = [amplify_assistant_to_openai(a) for a in assistants]
        return {
            "object": "list",
            "data": openai_assistants,
            "first_id": openai_assistants[0]["id"] if openai_assistants else None,
            "last_id": openai_assistants[-1]["id"] if openai_assistants else None,
            "has_more": False,
        }
    except requests.exceptions.RequestException as e:
        raise handle_upstream_error(logger, e, "listing")


@router.post("")
async def create_assistant(
    request: Request, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Create a new assistant via Amplify POST /assistant/create.

    Maps OpenAI assistant fields to Amplify assistant fields.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    logger.info("Creating assistant: %s", body.get("name", "<unnamed>"))

    amplify_payload: AmplifyAssistantCreateRequest = {
        "data": {
            "name": body.get("name", ""),
            "description": body.get("description", ""),
            "tags": body.get("metadata", {}).get("tags", []) if body.get("metadata") else [],
            "instructions": body.get("instructions", ""),
            "dataSources": [],
            "tools": body.get("tools", []),
        }
    }

    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/create",
            headers=headers,
            json=amplify_payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("data", {})
        return {
            "id": result.get("assistantId", result.get("id", "")),
            "object": "assistant",
            "created_at": int(time.time()),
            "name": body.get("name", ""),
            "description": body.get("description", None),
            "model": body.get("model", "amplify"),
            "instructions": body.get("instructions", None),
            "tools": body.get("tools", []),
            "file_ids": [],
            "metadata": body.get("metadata", {}),
        }
    except requests.exceptions.RequestException as e:
        raise handle_upstream_error(logger, e, "creating")


@router.get("/{assistant_id:path}")
async def retrieve_assistant(
    assistant_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Retrieve a single assistant by ID.

    Amplify has no per-assistant endpoint; fetches the full list and filters.
    """
    logger.info("Retrieving assistant: %s", assistant_id)
    try:
        resp = requests.get(f"{AMPLIFY_BASE_URL}/assistant/list", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        assistants = data.get("data", [])
        match = next(
            (a for a in assistants if a.get("assistantId") == assistant_id or a.get("id") == assistant_id),
            None,
        )
        if not match:
            raise HTTPException(status_code=404, detail=f"Assistant '{assistant_id}' not found")
        return amplify_assistant_to_openai(match)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise handle_upstream_error(logger, e, "retrieving")


@router.post("/{assistant_id:path}")
async def modify_assistant(
    assistant_id: str,
    request: Request,
    headers: dict = Depends(get_amplify_headers),
) -> Dict[str, Any]:
    """
    Modify an existing assistant via Amplify POST /assistant/create (upsert).

    Passing assistantId in the body triggers an update.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    logger.info("Modifying assistant: %s", assistant_id)

    amplify_payload: AmplifyAssistantCreateRequest = {
        "data": {
            "name": body.get("name", ""),
            "description": body.get("description", ""),
            "assistantId": assistant_id,
            "tags": body.get("metadata", {}).get("tags", []) if body.get("metadata") else [],
            "instructions": body.get("instructions", ""),
            "dataSources": [],
            "tools": body.get("tools", []),
        }
    }

    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/create",
            headers=headers,
            json=amplify_payload,
            timeout=30,
        )
        resp.raise_for_status()
        return {
            "id": assistant_id,
            "object": "assistant",
            "created_at": int(time.time()),
            "name": body.get("name", ""),
            "description": body.get("description", None),
            "model": body.get("model", "amplify"),
            "instructions": body.get("instructions", None),
            "tools": body.get("tools", []),
            "file_ids": [],
            "metadata": body.get("metadata", {}),
        }
    except requests.exceptions.RequestException as e:
        raise handle_upstream_error(logger, e, "modifying")


@router.delete("/{assistant_id:path}")
async def delete_assistant(
    assistant_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """Delete an assistant via Amplify POST /assistant/delete."""
    logger.info("Deleting assistant: %s", assistant_id)
    payload: AmplifyAssistantCreateRequest = {"data": {"assistantId": assistant_id}}
    try:
        resp = requests.post(
            f"{AMPLIFY_BASE_URL}/assistant/delete",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": assistant_id,
            "object": "assistant.deleted",
            "deleted": bool(data.get("success", False)),
        }
    except requests.exceptions.RequestException as e:
        raise handle_upstream_error(logger, e, "deleting")
