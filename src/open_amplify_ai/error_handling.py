"""Precise error normalization for upstream errors."""
import logging
from typing import Optional

import httpx
from fastapi import HTTPException

from open_amplify_ai.types import ErrorType

logger = logging.getLogger(__name__)


def normalize_upstream_error(
    error: Exception,
    context: str,
    request_id: Optional[str] = None,
) -> HTTPException:
    """
    Normalize upstream errors to OpenAI-compatible error responses.
    
    Maps HTTP status codes appropriately:
    - 400 -> 400 (invalid_request_error)
    - 401 -> 401 (authentication_error)
    - 403 -> 403 (permission_error)
    - 404 -> 404 (not_found_error)
    - 408 -> 408 (timeout_error)
    - 422 -> 400 (invalid_request_error)
    - 429 -> 429 (rate_limit_error)
    - 500 -> 500 (api_error)
    - 502 -> 502 (service_unavailable_error)
    - 503 -> 503 (service_unavailable_error)
    - 504 -> 504 (timeout_error)
    - Others -> 500 (api_error)
    """
    # Extract error details
    status_code = 500
    error_type = ErrorType.API_ERROR
    error_message = str(error)
    error_detail = ""
    
    # Handle httpx errors
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        
        # Try to extract response body
        try:
            error_detail = error.response.text
        except Exception:
            pass
        
        # Map status codes to error types
        if status_code == 400 or status_code == 422:
            error_type = ErrorType.INVALID_REQUEST_ERROR
            status_code = 400
        elif status_code == 401:
            error_type = ErrorType.AUTHENTICATION_ERROR
        elif status_code == 403:
            error_type = ErrorType.PERMISSION_ERROR
        elif status_code == 404:
            error_type = ErrorType.NOT_FOUND_ERROR
        elif status_code == 408:
            error_type = ErrorType.TIMEOUT_ERROR
        elif status_code == 429:
            error_type = ErrorType.RATE_LIMIT_ERROR
        elif status_code == 502 or status_code == 503:
            error_type = ErrorType.SERVICE_UNAVAILABLE_ERROR
        elif status_code == 504:
            error_type = ErrorType.TIMEOUT_ERROR
        elif 500 <= status_code < 600:
            error_type = ErrorType.API_ERROR
            status_code = 500
        
        error_message = f"Amplify API error during {context}: HTTP {error.response.status_code}"
    
    elif isinstance(error, httpx.TimeoutException):
        status_code = 504
        error_type = ErrorType.TIMEOUT_ERROR
        error_message = f"Request timeout during {context}"
    
    elif isinstance(error, httpx.ConnectError):
        status_code = 502
        error_type = ErrorType.SERVICE_UNAVAILABLE_ERROR
        error_message = f"Could not connect to Amplify API during {context}"
    
    elif isinstance(error, httpx.NetworkError):
        status_code = 502
        error_type = ErrorType.SERVICE_UNAVAILABLE_ERROR
        error_message = f"Network error during {context}"
    
    else:
        # Unknown error
        error_type = ErrorType.API_ERROR
        error_message = f"Error during {context}: {error}"
    
    # Log the error with full details
    logger.error(
        "Upstream error during %s: %s (status=%d, type=%s)",
        context,
        error_message,
        status_code,
        error_type.value,
    )
    
    if error_detail:
        logger.error("Response body: %s", error_detail[:1000])  # Limit log size
    
    # Log request details if available
    if hasattr(error, "request"):
        try:
            req = error.request
            logger.error(
                "Request: %s %s",
                req.method if hasattr(req, "method") else "?",
                req.url if hasattr(req, "url") else "?",
            )
        except Exception:
            pass
    
    # Build OpenAI-compatible error response
    error_response = {
        "error": {
            "message": error_message,
            "type": error_type.value,
            "param": None,
            "code": None,
        }
    }
    
    # Add request ID if available
    if request_id:
        error_response["error"]["code"] = request_id
    
    return HTTPException(status_code=status_code, detail=error_response)


def create_validation_error(
    message: str,
    param: Optional[str] = None,
) -> HTTPException:
    """Create a 400 validation error in OpenAI format."""
    error_response = {
        "error": {
            "message": message,
            "type": ErrorType.INVALID_REQUEST_ERROR.value,
            "param": param,
            "code": None,
        }
    }
    return HTTPException(status_code=400, detail=error_response)
