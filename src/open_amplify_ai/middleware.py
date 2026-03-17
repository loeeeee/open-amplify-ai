"""Middleware definitions."""
import json
import logging
import os
from fastapi import Request

from open_amplify_ai.stats import (
    build_record,
    extract_completion_tokens,
    extract_prompt_tokens,
    write_token_stats,
)

logger = logging.getLogger(__name__)

_TOKEN_STATS_CSV = os.path.join("logs", "token_stats.csv")

class ErrorLoggingMiddleware:
    """
    Pure ASGI middleware that logs the request body when the endpoint returns an error
    (status code >= 400) or raises an unhandled exception.
    Skips multipart form data to avoid memory issues.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
            
        request = Request(scope, receive)
        content_type = request.headers.get("content-type", "")
        
        if "multipart/form-data" in content_type:
            try:
                await self.app(scope, receive, send)
            except Exception as e:
                logger.error(f"Unhandled exception: {e}\\n{request.method} {request.url}\\nBody omitted (multipart)")
                raise
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        body_sent = False
        async def new_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        status_code = None
        response_body = b""
        async def custom_send(message):
            nonlocal status_code, response_body
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)
            
        try:
            await self.app(scope, new_receive, custom_send)
            if status_code and status_code >= 400:
                logger.error(f"Request error (status {status_code}): {request.method} {request.url}\\nRequest Body: {body.decode(errors='replace')}\\nResponse Body: {response_body.decode(errors='replace')}")
        except Exception as e:
            logger.error(f"Unhandled exception: {e}\n{request.method} {request.url}\nRequest Body: {body.decode(errors='replace')}\nResponse Body: {response_body.decode(errors='replace')}")
            raise


class DebugLoggingMiddleware:
    """
    Pure ASGI middleware that logs every request and response body for debugging purposes.
    Skips multipart form data to avoid memory issues.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
            
        if os.getenv("AMPLIFY_DEBUG", "0").lower() not in ("1", "true", "yes"):
            return await self.app(scope, receive, send)
            
        request = Request(scope, receive)
        content_type = request.headers.get("content-type", "")
        
        if "multipart/form-data" in content_type:
            logger.debug(f"DEBUG LOG: {request.method} {request.url} (multipart body omitted)")
            try:
                await self.app(scope, receive, send)
            except Exception as e:
                logger.debug(f"DEBUG LOG (Unhandled exception: {e}): {request.method} {request.url} (multipart body omitted)")
                raise
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        body_sent = False
        async def new_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        status_code = None
        response_body = b""
        async def custom_send(message):
            nonlocal status_code, response_body
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)
            
        try:
            await self.app(scope, new_receive, custom_send)
            logger.debug(f"DEBUG LOG (status {status_code}): {request.method} {request.url}\nRequest Body: {body.decode(errors='replace')}\nResponse Body: {response_body.decode(errors='replace')}")
        except Exception as e:
            logger.debug(f"DEBUG LOG (Unhandled exception: {e}): {request.method} {request.url}\nRequest Body: {body.decode(errors='replace')}\nResponse Body: {response_body.decode(errors='replace')}")
            raise


class TokenCounterMiddleware:
    """
    Pure ASGI middleware that records per-request token usage statistics to a CSV file.

    For /v1/chat/completions requests, prompt and completion token counts are
    estimated from the request and response bodies (4 characters per token).
    All other endpoints are recorded with zero token counts but with full IP
    address, status code, and error information.

    The CSV file is written to logs/token_stats.csv relative to the server's
    working directory (resolves to /var/lib/amplify-ai/logs/token_stats.csv
    under the NixOS systemd service).
    """

    def __init__(self, app):
        """Initialise with the next ASGI app in the stack."""
        self.app = app

    async def __call__(self, scope, receive, send):
        """Intercept each HTTP request and append a stats row to the CSV."""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope, receive)
        content_type = request.headers.get("content-type", "")
        path = request.url.path
        method = request.method

        ip_address = request.headers.get("x-forwarded-for", "")
        if not ip_address and request.client:
            ip_address = request.client.host or ""

        is_chat = path == "/v1/chat/completions"
        is_multipart = "multipart/form-data" in content_type

        if is_multipart:
            error_desc = ""
            status_code = 0
            async def _track_status(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)
            try:
                await self.app(scope, receive, _track_status)
            except Exception as exc:
                error_desc = str(exc)
                raise
            finally:
                record = build_record(
                    ip_address=ip_address,
                    method=method,
                    path=path,
                    status_code=status_code,
                    prompt_tokens=0,
                    completion_tokens=0,
                    error=error_desc,
                )
                write_token_stats(record, _TOKEN_STATS_CSV)
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        body_sent = False
        async def new_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        status_code = 0
        response_body = b""
        async def custom_send(message):
            nonlocal status_code, response_body
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)

        is_streaming = False
        if is_chat:
            try:
                req_data = json.loads(body.decode("utf-8", errors="replace"))
                is_streaming = bool(req_data.get("stream", False))
            except Exception:
                pass

        error_desc = ""
        try:
            await self.app(scope, new_receive, custom_send)
            if status_code >= 400:
                error_desc = f"HTTP {status_code}"
        except Exception as exc:
            error_desc = str(exc)
            raise
        finally:
            prompt_tokens = extract_prompt_tokens(body) if is_chat else 0
            completion_tokens = (
                extract_completion_tokens(response_body, is_streaming) if is_chat else 0
            )
            record = build_record(
                ip_address=ip_address,
                method=method,
                path=path,
                status_code=status_code,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=error_desc,
            )
            write_token_stats(record, _TOKEN_STATS_CSV)
