# AuraGuard AI — Coding Standards

## Python Version

Python 3.10+ required. Use `match` statements, `X | Y` union types, and `from __future__ import annotations` where needed.

## General Rules

- Never hardcode secrets, API keys, or configuration values — always read from environment variables via `Settings`
- All external API calls must have explicit timeouts via `asyncio.wait_for`
- Every downstream failure (Gemini, ElevenLabs, Pygame, MongoDB) must be caught, logged, and absorbed — never propagate to the caller
- The Brain must always return HTTP 200 to the Vision Engine for integration failures; HTTP 5xx is reserved for unhandled exceptions only
- `image_b64` must never be stored in MongoDB — enforced at the `EventRecord` model level, not by runtime deletion

## Imports

- Brain service modules import models from `services.brain.models`, not directly from `shared.contract`
- `shared/contract.py` is the only place Pydantic models are defined — never duplicate them
- Use absolute imports throughout

## Pydantic Models

- All models inherit from `pydantic.BaseModel`
- Settings use `pydantic_settings.BaseSettings` with `SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")`
- Use `Literal` types for constrained string fields (e.g. `type`, `source`, `processing_status`)
- Validate at the boundary — let FastAPI/Pydantic reject bad input with HTTP 422 automatically

## Async

- All FastAPI route handlers and service functions that do I/O must be `async def`
- Use `asyncio.wait_for` for all external calls with these timeouts:
  - Gemini: 10 seconds
  - ElevenLabs: 15 seconds
  - MongoDB write: 5 seconds
  - Health check ping: 3 seconds
- Never use `time.sleep` in async code — use `await asyncio.sleep`

## Error Handling

```python
# Pattern for all downstream service calls
try:
    result = await asyncio.wait_for(some_call(), timeout=10.0)
except asyncio.TimeoutError:
    logger.error("Call timed out after 10s")
    return fallback_value
except Exception as e:
    logger.error(f"Call failed: {e}")
    return fallback_value
```

## Logging

- Use the standard `logging` module — `logger = logging.getLogger(__name__)` in every module
- Log at `WARNING` for degraded-but-recoverable states (MongoDB unreachable at startup, Pygame device fallback)
- Log at `ERROR` for failures during request processing (Gemini failure, ElevenLabs failure, MongoDB write failure)
- Log at `EXCEPTION` (includes stack trace) only in the global exception handler
- Never log the value of `image_b64` or any API key

## FastAPI Conventions

- Routers live in `services/brain/routes/` — one file per route group
- Mount routers in `main.py` via `app.include_router(router)`
- Use `request.state.event_id` to pass the event ID to the global exception handler
- The global exception handler must use `JSONResponse` to ensure `Content-Type: application/json`
- Use `@asynccontextmanager` lifespan (not deprecated `on_event` hooks)

## Testing

- All tests live under `tests/` mirroring the service structure
- Property-based tests use `hypothesis` with `@settings(max_examples=100)`
- Unit tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`)
- Mock all downstream services in `tests/brain/conftest.py` — tests must not make real API calls
- Tag each property test with a comment referencing its property number from the design doc
- Use `monkeypatch` for environment variable manipulation in config tests

## File Naming

- Snake_case for all Python files and directories
- Test files mirror source structure: `services/brain/services/gemini.py` → `tests/brain/test_gemini.py`
