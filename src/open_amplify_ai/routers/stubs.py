"""Stubs endpoints for unimplemented functionality."""
from fastapi import APIRouter, Request

from open_amplify_ai.utils import not_implemented

router = APIRouter(prefix="/v1", tags=["Stubs"])

# ---------------------------------------------------------------------------
# Embeddings (501 stub)
# ---------------------------------------------------------------------------

@router.post("/embeddings")
async def create_embedding(request: Request) -> None:
    """
    Amplify has no raw embedding vector endpoint.

    POST /embedding-dual-retrieval returns ranked document snippets, not float vectors.
    """
    raise not_implemented("Embedding vector generation")

# ---------------------------------------------------------------------------
# Audio endpoints (501 stubs)
# ---------------------------------------------------------------------------

@router.post("/audio/speech")
async def create_speech(request: Request) -> None:
    """Amplify has no text-to-speech capability."""
    raise not_implemented("Audio speech synthesis")

@router.post("/audio/transcriptions")
async def create_transcription(request: Request) -> None:
    """Amplify has no audio transcription capability."""
    raise not_implemented("Audio transcription")

@router.post("/audio/translations")
async def create_translation(request: Request) -> None:
    """Amplify has no audio translation capability."""
    raise not_implemented("Audio translation")

# ---------------------------------------------------------------------------
# Images endpoints (501 stubs)
# ---------------------------------------------------------------------------

@router.post("/images/generations")
async def create_image(request: Request) -> None:
    """Amplify has no image generation capability."""
    raise not_implemented("Image generation")


@router.post("/images/edits")
async def create_image_edit(request: Request) -> None:
    """Amplify has no image editing capability."""
    raise not_implemented("Image editing")


@router.post("/images/variations")
async def create_image_variation(request: Request) -> None:
    """Amplify has no image variation capability."""
    raise not_implemented("Image variation")

# ---------------------------------------------------------------------------
# Fine-tuning endpoints (501 stubs)
# ---------------------------------------------------------------------------

@router.post("/fine_tuning/jobs")
async def create_fine_tuning_job(request: Request) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job creation")


@router.get("/fine_tuning/jobs")
async def list_fine_tuning_jobs() -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job listing")


@router.get("/fine_tuning/jobs/{fine_tuning_job_id}")
async def retrieve_fine_tuning_job(fine_tuning_job_id: str) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job retrieval")


@router.post("/fine_tuning/jobs/{fine_tuning_job_id}/cancel")
async def cancel_fine_tuning_job(fine_tuning_job_id: str) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning job cancellation")


@router.get("/fine_tuning/jobs/{fine_tuning_job_id}/events")
async def list_fine_tuning_events(fine_tuning_job_id: str) -> None:
    """Amplify does not support fine-tuning."""
    raise not_implemented("Fine-tuning event listing")

# ---------------------------------------------------------------------------
# Moderations (501 stub)
# ---------------------------------------------------------------------------

@router.post("/moderations")
async def create_moderation(request: Request) -> None:
    """Amplify has no content moderation endpoint."""
    raise not_implemented("Content moderation")


# ---------------------------------------------------------------------------
# Batch endpoints (501 stubs)
# ---------------------------------------------------------------------------


@router.post("/batches")
async def create_batch(request: Request) -> None:
    """Amplify has no batch processing endpoint."""
    raise not_implemented("Batch creation")


@router.get("/batches")
async def list_batches() -> None:
    """Amplify has no batch listing endpoint."""
    raise not_implemented("Batch listing")


@router.get("/batches/{batch_id}")
async def retrieve_batch(batch_id: str) -> None:
    """Amplify has no batch status endpoint."""
    raise not_implemented("Batch retrieval")


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str) -> None:
    """Amplify has no batch cancellation endpoint."""
    raise not_implemented("Batch cancellation")
