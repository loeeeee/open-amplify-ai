"""Middleware definitions."""
import logging
from fastapi import Request

logger = logging.getLogger(__name__)

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
        async def custom_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
            
        try:
            await self.app(scope, new_receive, custom_send)
            if status_code and status_code >= 400:
                logger.error(f"Request error (status {status_code}): {request.method} {request.url}\\nBody: {body.decode(errors='replace')}")
        except Exception as e:
            logger.error(f"Unhandled exception: {e}\\n{request.method} {request.url}\\nBody: {body.decode(errors='replace')}")
            raise
