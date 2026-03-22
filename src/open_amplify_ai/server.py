"""ASGI application entry point for the Amplify AI OpenAI-compatible server."""
import logging
import os
import sys
from typing import Optional

import dotenv
import uvicorn
from fastapi import FastAPI

from open_amplify_ai.middleware import (
    DebugLoggingMiddleware,
    ErrorLoggingMiddleware,
    TokenCounterMiddleware,
)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

app = FastAPI(title="Amplify AI OpenAI Compatible API")

app.add_middleware(DebugLoggingMiddleware)
app.add_middleware(ErrorLoggingMiddleware)
app.add_middleware(TokenCounterMiddleware)

if os.getenv("AMPLIFY_DEBUG", "0").lower() in ("1", "true", "yes"):
    logger.setLevel(logging.DEBUG)
    logging.getLogger("open_amplify_ai.middleware").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from open_amplify_ai.routers import (  # noqa: E402
    assistants,
    chat,
    dashboard,
    files,
    models,
    stubs,
    threads,
    vector_stores,
)

app.include_router(dashboard.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(assistants.router)
app.include_router(threads.router)
app.include_router(vector_stores.router)
app.include_router(stubs.router)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(port: Optional[int] = None, debug: bool = False) -> None:
    """Start the Uvicorn server.

    Bind address and port are read from environment variables so that the
    NixOS systemd unit can configure them without requiring code changes:
      AMPLIFY_SERVER_HOST  - defaults to 0.0.0.0
      AMPLIFY_SERVER_PORT  - defaults to 8080

    CLI argument for port overrides the environment variable.
    """
    if debug:
        os.environ["AMPLIFY_DEBUG"] = "1"
        logger.setLevel(logging.DEBUG)
        logging.getLogger("open_amplify_ai.middleware").setLevel(logging.DEBUG)

    host = os.getenv("AMPLIFY_SERVER_HOST", "0.0.0.0")
    port = port or int(os.getenv("AMPLIFY_SERVER_PORT", "8080"))
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run("open_amplify_ai.server:app", host=host, port=port, reload=False)
