"""Models endpoints mapped to the Amplify API."""
import logging
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, Depends, HTTPException

from open_amplify_ai.config import AMPLIFY_BASE_URL
from open_amplify_ai.auth import get_amplify_headers
from open_amplify_ai.types import (
    ModelCapabilities,
    ModelCost,
    ModelInfo,
    ModelLimit,
)
from open_amplify_ai.utils import handle_upstream_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/models", tags=["Models"])


def _build_model_info(m: Dict[str, Any]) -> ModelInfo:
    """Map a single Amplify model dict to a ModelInfo with Kilo-consumable fields.

    Pricing values from Amplify are passed through as-is. The convention used
    throughout this translator is dollars per million tokens, applied here in
    exactly one place.
    """
    input_ctx = m.get("inputContextWindow")
    output_limit = m.get("outputTokenLimit")

    cost = ModelCost(
        input=m.get("inputTokenCost"),
        output=m.get("outputTokenCost"),
        cache_read=m.get("cachedInputTokenCost"),
        cache_write=m.get("cachedOutputTokenCost"),
    )

    limit = ModelLimit(
        context=input_ctx,
        output=output_limit,
    )

    capabilities = ModelCapabilities(
        images=m.get("supportsImages"),
        system_prompt=m.get("supportsSystemPrompts"),
        description=m.get("description"),
    )

    max_model_len = (
        input_ctx + output_limit
        if input_ctx is not None and output_limit is not None
        else None
    )

    return ModelInfo(
        id=m.get("id", ""),
        cost=cost,
        limit=limit,
        capabilities=capabilities,
        display_name=m.get("name"),
        max_output_tokens=output_limit,
        context_length=input_ctx,
        max_model_len=max_model_len,
    )


def _model_to_dict(info: ModelInfo) -> Dict[str, Any]:
    """Serialize a ModelInfo to a response dict with Kilo-compatible fields."""
    result: Dict[str, Any] = {
        "id": info.id,
        "object": info.object,
        "created": info.created,
        "owned_by": info.owned_by,
    }
    if info.cost is not None:
        cost_dict = info.cost.to_dict()
        if cost_dict:
            result["cost"] = cost_dict
    if info.limit is not None:
        limit_dict = info.limit.to_dict()
        if limit_dict:
            result["limit"] = limit_dict
    if info.capabilities is not None:
        cap_dict = info.capabilities.to_dict()
        if cap_dict:
            result["capabilities"] = cap_dict
    if info.display_name is not None:
        result["display_name"] = info.display_name
    # Legacy flat fields for backward compatibility
    result["max_output_tokens"] = info.max_output_tokens
    result["context_length"] = info.context_length
    result["max_model_len"] = info.max_model_len
    return result


def _filter_alias_models(amplify_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove alias entries that are not real callable models.

    Amplify returns entries like 'default', 'advanced', 'cheapest', and
    'documentCaching' as metadata or recommendations. These should not
    appear as separate selectable model IDs.
    """
    alias_keys = {"default", "advanced", "cheapest", "documentCaching"}
    return [
        m for m in amplify_models
        if m.get("id") not in alias_keys
    ]


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
        amplify_models = _filter_alias_models(amplify_models)
        models = [_build_model_info(m) for m in amplify_models]

        return {
            "object": "list",
            "data": [_model_to_dict(m) for m in models],
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

        info = _build_model_info(match)
        return _model_to_dict(info)
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
