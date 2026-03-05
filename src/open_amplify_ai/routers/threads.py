"""Threads endpoints mapped to the Amplify API."""
import logging
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.utils import not_implemented, handle_upstream_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/threads", tags=["Threads"])


@router.post("")
async def create_thread(request: Request) -> None:
    """Amplify has no standalone thread creation endpoint."""
    raise not_implemented("Thread creation")


@router.get("/{thread_id}")
async def retrieve_thread(thread_id: str) -> None:
    """Amplify has no thread retrieval endpoint."""
    raise not_implemented("Thread retrieval")


@router.post("/{thread_id}")
async def modify_thread(thread_id: str, request: Request) -> None:
    """Amplify has no thread modification endpoint."""
    raise not_implemented("Thread modification")


@router.delete("/{thread_id:path}")
async def delete_thread(
    thread_id: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Delete an Amplify thread via DELETE /assistant/openai/thread/delete.

    The thread_id is passed as a query parameter to the Amplify endpoint.
    """
    logger.info("Deleting thread: %s", thread_id)
    try:
        resp = requests.delete(
            f"{AMPLIFY_BASE_URL}/assistant/openai/thread/delete",
            headers=headers,
            params={"threadId": thread_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": thread_id,
            "object": "thread.deleted",
            "deleted": bool(data.get("success", False)),
        }
    except requests.exceptions.RequestException as e:
        raise handle_upstream_error(logger, e, "deleting")


# ---------------------------------------------------------------------------
# Messages endpoints
# ---------------------------------------------------------------------------


@router.post("/{thread_id}/messages")
async def create_message(thread_id: str, request: Request) -> None:
    """Amplify has no standalone message creation endpoint."""
    raise not_implemented("Message creation")


@router.get("/{thread_id}/messages")
async def list_messages(thread_id: str) -> None:
    """Amplify has no thread message history endpoint."""
    raise not_implemented("Message listing")


@router.get("/{thread_id}/messages/{message_id}")
async def retrieve_message(thread_id: str, message_id: str) -> None:
    """Amplify has no single message retrieval endpoint."""
    raise not_implemented("Message retrieval")


# ---------------------------------------------------------------------------
# Runs endpoints
# ---------------------------------------------------------------------------


@router.post("/{thread_id}/runs")
async def create_run(thread_id: str, request: Request) -> None:
    """Amplify run model is synchronous and does not match the OpenAI async run API."""
    raise not_implemented("Run creation via threads")


@router.get("/{thread_id}/runs/{run_id}")
async def retrieve_run(thread_id: str, run_id: str) -> None:
    """Amplify has no async run status endpoint."""
    raise not_implemented("Run retrieval")


@router.post("/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> None:
    """Amplify is synchronous; run cancellation is not supported."""
    raise not_implemented("Run cancellation")


@router.get("/{thread_id}/runs")
async def list_runs(thread_id: str) -> None:
    """Amplify has no run history endpoint."""
    raise not_implemented("Run listing")


@router.post("/{thread_id}/runs/{run_id}/submit_tool_outputs")
async def submit_tool_outputs(thread_id: str, run_id: str, request: Request) -> None:
    """Amplify does not support tool-calling pause-and-resume run semantics."""
    raise not_implemented("Tool output submission")


@router.post("/runs")
async def create_thread_and_run(request: Request) -> None:
    """Amplify has no combined create-thread-and-run endpoint."""
    raise not_implemented("Create thread and run")


# ---------------------------------------------------------------------------
# Run steps endpoints
# ---------------------------------------------------------------------------


@router.get("/{thread_id}/runs/{run_id}/steps")
async def list_run_steps(thread_id: str, run_id: str) -> None:
    """Amplify does not expose run step details."""
    raise not_implemented("Run steps listing")


@router.get("/{thread_id}/runs/{run_id}/steps/{step_id}")
async def retrieve_run_step(thread_id: str, run_id: str, step_id: str) -> None:
    """Amplify does not expose individual run step details."""
    raise not_implemented("Run step retrieval")
