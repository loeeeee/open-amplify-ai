from open_amplify_ai.utils import handle_upstream_error
"""Models endpoints mapped to the Amplify API."""
import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.types import ModelInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/models", tags=["Models"])


@router.get("")
async def list_models(headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """Convert Amplify GET /available_models to OpenAI GET /v1/models."""
    logger.info("Listing available models")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{AMPLIFY_BASE_URL}/available_models", headers=headers
            )
            response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise HTTPException(
                status_code=500, detail="Failed to fetch models from Amplify AI"
            )

        amplify_models = data.get("data", {}).get("models", [])
        models = [
            ModelInfo(
                id=m.get("id"),
                max_output_tokens=m.get("outputTokenLimit"),
                context_length=m.get("inputContextWindow"),
                max_model_len=(
                    m.get("inputContextWindow", 0) + m.get("outputTokenLimit", 0)
                    if m.get("inputContextWindow") and m.get("outputTokenLimit")
                    else None
                ),
            )
            for m in amplify_models
        ]

        return {
            "object": "list",
            "data": [
                {
                    "id": m.id,
                    "object": m.object,
                    "created": m.created,
                    "owned_by": m.owned_by,
                    "max_output_tokens": m.max_output_tokens,
                    "context_length": m.context_length,
                    "max_model_len": m.max_model_len,
                }
                for m in models
            ],
        }
    except httpx.HTTPError as e:
        raise handle_upstream_error(logger, e, "fetching")


@router.get("/{model}")
async def retrieve_model(
    model: str, headers: dict = Depends(get_amplify_headers)
) -> Dict[str, Any]:
    """
    Retrieve a single model by ID.

    Amplify has no per-model endpoint, so this fetches the full list and filters.
    """
    logger.info("Retrieving model: %s", model)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{AMPLIFY_BASE_URL}/available_models", headers=headers
            )
            response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise HTTPException(
                status_code=500, detail="Failed to fetch models from Amplify AI"
            )

        amplify_models = data.get("data", {}).get("models", [])
        match = next((m for m in amplify_models if m.get("id") == model), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

        info = ModelInfo(
            id=match.get("id"),
            max_output_tokens=match.get("outputTokenLimit"),
            context_length=match.get("inputContextWindow"),
            max_model_len=(
                match.get("inputContextWindow", 0) + match.get("outputTokenLimit", 0)
                if match.get("inputContextWindow") and match.get("outputTokenLimit")
                else None
            ),
        )
        return {
            "id": info.id,
            "object": info.object,
            "created": info.created,
            "owned_by": info.owned_by,
            "max_output_tokens": info.max_output_tokens,
            "context_length": info.context_length,
            "max_model_len": info.max_model_len,
        }
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        raise handle_upstream_error(logger, e, "fetching")


@router.delete("/{model}")
async def delete_model(model: str) -> Dict[str, Any]:
    """
    Amplify does not support model deletion.

    Returns 405 Method Not Allowed per the mapping document.
    """
    logger.info("Attempted deletion of model %s (not supported)", model)
    raise HTTPException(
        status_code=405, detail="Model deletion is not supported by Amplify AI."
    )
