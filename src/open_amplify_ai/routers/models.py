"""Models endpoints mapped to the Amplify API."""
import logging
from typing import Any, Dict

import requests
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
        response = requests.get(f"{AMPLIFY_BASE_URL}/available_models", headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise HTTPException(status_code=500, detail="Failed to fetch models from Amplify AI")

        amplify_models = data.get("data", {}).get("models", [])
        models = [ModelInfo(id=m.get("id")) for m in amplify_models]

        return {
            "object": "list",
            "data": [
                {
                    "id": m.id,
                    "object": m.object,
                    "created": m.created,
                    "owned_by": m.owned_by,
                }
                for m in models
            ],
        }
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching models: %s", e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.get("/{model}")
async def retrieve_model(model: str, headers: dict = Depends(get_amplify_headers)) -> Dict[str, Any]:
    """
    Retrieve a single model by ID.

    Amplify has no per-model endpoint, so this fetches the full list and filters.
    """
    logger.info("Retrieving model: %s", model)
    try:
        response = requests.get(f"{AMPLIFY_BASE_URL}/available_models", headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise HTTPException(status_code=500, detail="Failed to fetch models from Amplify AI")

        amplify_models = data.get("data", {}).get("models", [])
        match = next((m for m in amplify_models if m.get("id") == model), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

        info = ModelInfo(id=match.get("id"))
        return {
            "id": info.id,
            "object": info.object,
            "created": info.created,
            "owned_by": info.owned_by,
        }
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching model %s: %s", model, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with Amplify AI: {e}")


@router.delete("/{model}")
async def delete_model(model: str) -> Dict[str, Any]:
    """
    Amplify does not support model deletion.

    Returns 405 Method Not Allowed per the mapping document.
    """
    logger.info("Attempted deletion of model %s (not supported)", model)
    raise HTTPException(status_code=405, detail="Model deletion is not supported by Amplify AI.")
