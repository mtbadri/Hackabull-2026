"""
services/brain/main.py — FastAPI application entry point for the AI Brain service.

Wires together the lifespan startup/shutdown sequence, global exception handler,
and route mounts. All downstream service initialisation happens in the lifespan
so that app.state is populated before any request is handled.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 8.5
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from services.brain.config import get_settings
from services.brain.services.audio import init_pygame
from services.brain.services.mongodb import init_motor, verify_mongodb

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.

    Startup sequence (Requirements 9.1, 9.2, 9.3):
    1. Load validated settings — exits with code 1 if any env var is missing.
    2. Initialise Motor MongoDB client.
    3. Verify MongoDB connectivity — log WARNING on failure, do NOT exit.
    4. Initialise Pygame mixer — log WARNING on failure, do NOT exit.
    5. Construct ElevenLabs client.
    6. Store all clients and settings on app.state.

    Shutdown (Requirement 9.4):
    - Close the Motor MongoDB client cleanly.
    """
    # 1. Load settings
    settings = get_settings()

    # 2. Initialise Motor MongoDB client
    motor_client = init_motor(settings.MONGODB_URI)

    # 3. Verify MongoDB connectivity (degraded start allowed)
    try:
        connected = await verify_mongodb(motor_client)
        if not connected:
            logger.warning(
                "MongoDB connectivity check failed at startup — "
                "continuing in degraded state."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MongoDB connectivity check raised an exception at startup: %s — "
            "continuing in degraded state.",
            exc,
        )

    # 4. Initialise Pygame mixer (degraded start allowed)
    try:
        init_pygame(settings.GLASSES_AUDIO_DEVICE)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Pygame initialisation failed at startup: %s — "
            "continuing in degraded state.",
            exc,
        )

    # 5. Construct ElevenLabs client
    from elevenlabs import ElevenLabs  # type: ignore

    elevenlabs_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    # 6. Store on app.state
    app.state.motor_client = motor_client
    app.state.elevenlabs_client = elevenlabs_client
    app.state.settings = settings

    logger.info("AI Brain startup complete.")

    yield  # application runs

    # Shutdown
    motor_client.close()
    logger.info("AI Brain shutdown complete.")


# FastAPI application

app = FastAPI(
    title="AuraGuard AI Brain",
    description="Central coordinator for the AuraGuard AI assistive platform.",
    version="1.0.0",
    lifespan=lifespan,
)


# Global exception handler (Requirement 8.5)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions, log the full stack trace, and return HTTP 500.

    The response body always includes the event_id from request.state when
    available, falling back to "unknown" for requests that never set it.

    Requirement 8.5: IF an unhandled exception occurs, THE Brain SHALL catch it,
    log the full stack trace, and return HTTP 500 with a structured JSON body.
    """
    event_id = getattr(request.state, "event_id", "unknown")
    logger.exception(
        "Unhandled exception during request processing (event_id=%s)", event_id
    )
    return JSONResponse(
        status_code=500,
        content={
            "event_id": event_id,
            "status": "error",
            "message": "Internal server error.",
        },
    )


# Route mounts

from services.brain.routes.event import router as event_router  # noqa: E402
from services.brain.routes.health import router as health_router  # noqa: E402

app.include_router(event_router)
app.include_router(health_router)
