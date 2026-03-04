"""Authentication dependencies for the API."""
import logging
import os
from typing import Dict

from fastapi import Depends, HTTPException

logger = logging.getLogger(__name__)

def get_amplify_token() -> str:
    """Retrieve the API token from the environment."""
    token = os.getenv("AMPLIFY_AI_TOKEN")
    if not token:
        logger.error("AMPLIFY_AI_TOKEN not found in environment.")
        raise HTTPException(status_code=401, detail="Amplify AI token not configured")
    return token

def get_amplify_headers(token: str = Depends(get_amplify_token)) -> Dict[str, str]:
    """Generate headers with the authorization token required by Amplify API."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
