# Implementation Plan: AuraGuard AI — Brain Service

## Overview

Implement the `services/brain/` FastAPI service end-to-end: configuration, Pydantic models, Gemini verification, ElevenLabs streaming TTS, Pygame audio playback, Motor MongoDB logging, and the shared contract module. Each task builds incrementally toward a fully wired service that always returns HTTP 200 to the Vision Engine regardless of downstream failures.

All property-based tests use `hypothesis` with `@settings(max_examples=100)`.

---

## Tasks

- [x] 1. Scaffold project structure and shared contract
  - Create `shared/contract.py` with Pydantic models: `PersonProfile`, `IdentityMetadata`, `HealthMetadata`, `Event`, `EventRecord`, `EventResponse`, `HealthResponse`
  - `Event` must include all JSON Contract fields: `event_id`, `timestamp`, `patient_id`, `type` (`Literal["health","identity"]`), `subtype`, `confidence` (`float`), `image_b64`, `metadata`, `source` (`Literal["vision_engine_v1"]`)
  - `EventRecord` must include all Event fields except `image_b64`, plus `verified`, `voice_script`, `processing_status` (`Literal["success","partial_failure"]`), `processed_at`
  - Create `services/brain/` directory with empty `__init__.py` files for `routes/` and `services/` sub-packages
  - Create `tests/brain/` directory with `conftest.py` (FastAPI `TestClient` fixture with all downstream services mocked)
  - _Requirements: 1.3, 6.2, 16.2, 16.3, 16.5_

- [x] 2. Implement configuration layer
  - [x] 2.1 Create `services/brain/config.py` with `Settings(BaseSettings)` model
    - Declare all 9 required fields: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`, `PATIENT_NAME`, `PATIENT_ID`, `GLASSES_AUDIO_DEVICE`
    - Set `model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")`
    - Wrap `Settings()` construction in a `get_settings()` cached function; catch `ValidationError`, log missing fields, and `sys.exit(1)`
    - _Requirements: 7.1, 7.2, 7.3, 19.4_

  - [x] 2.2 Write unit tests for config validation
    - Test that missing any single required env var causes `sys.exit(1)` (use `monkeypatch.delenv`)
    - Test that all 9 vars present returns a valid `Settings` instance
    - _Requirements: 7.2, 7.3_

- [x] 3. Implement FastAPI app entry point and lifespan
  - [x] 3.1 Create `services/brain/main.py` with FastAPI app and `@asynccontextmanager` lifespan
    - Lifespan startup: call `get_settings()`, `init_motor()`, `await verify_mongodb()` (log warning on failure, do not exit), `init_pygame()` (log warning on failure), construct `ElevenLabs()` client
    - Store `motor_client`, `elevenlabs_client`, `settings` on `app.state`
    - Lifespan shutdown: call `motor_client.close()`
    - Register global exception handler: catch `Exception`, log full stack trace, return HTTP 500 `{"event_id": getattr(request.state, "event_id", "unknown"), "status": "error", "message": "Internal server error."}`
    - Mount `routes/event.router` and `routes/health.router`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 8.5_

  - [x] 3.2 Write unit tests for lifespan behavior
    - Test that MongoDB init failure logs a warning but does not raise
    - Test that Pygame init failure logs a warning but does not raise
    - _Requirements: 9.3_

- [x] 4. Implement Gemini verification service
  - [x] 4.1 Create `services/brain/services/gemini.py`
    - Implement `build_verification_prompt(subtype: str) -> str` that returns a subtype-specific YES/NO question
    - Implement `parse_gemini_verified(response_text: str) -> bool`: return `True` if `response_text.strip().upper()` starts with `"YES"` or contains `"YES,"` / `"YES."`, else `False`
    - Implement `async verify_health_event(image_b64: str, subtype: str, api_key: str) -> bool` using `asyncio.wait_for(..., timeout=10.0)`; on `TimeoutError` or any exception log and return `False`
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6_

  - [x] 4.2 Write property test for Gemini response parsing (Property 5)
    - **Property 5: Gemini Response Parsing Is Deterministic**
    - Generate strings starting with `"YES"` (any case) → assert `parse_gemini_verified` returns `True`
    - Generate strings not starting with `"YES"` → assert returns `False`
    - **Validates: Requirements 2.3**

  - [x] 4.3 Write property test for subtype-specific Gemini prompts (Property 4)
    - **Property 4: Health Event Gemini Prompts Are Subtype-Specific**
    - For each subtype in `["eating", "drinking", "medicine_taken"]`, assert the prompt from `build_verification_prompt` contains a keyword unique to that subtype and does not contain keywords from the other subtypes
    - **Validates: Requirements 2.2**

- [x] 5. Implement voice script generation
  - [x] 5.1 Create voice script logic in `services/brain/services/gemini.py` (or a dedicated `voice.py`)
    - Define `IDENTITY_TEMPLATE` and `HEALTH_TEMPLATES` dict exactly as specified
    - Implement `generate_identity_script(person_profile: PersonProfile, patient_name: str) -> str`
    - Implement `generate_health_script(subtype: str, patient_name: str) -> str`; return `""` for unknown subtypes
    - Implement `generate_voice_script(event: Event, verified: bool, patient_name: str) -> str`:
      - `type="identity"` → call `generate_identity_script`
      - `type="health"` and `verified=True` → call `generate_health_script`
      - `type="health"` and `verified=False` → return `""`
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 5.2 Write property test for identity voice scripts (Property 7)
    - **Property 7: Identity Voice Scripts Contain All Person Profile Fields**
    - Generate arbitrary `PersonProfile` instances; assert `voice_script` contains `name`, `relationship`, `background`, and `last_conversation` as substrings
    - **Validates: Requirements 3.1**

  - [x] 5.3 Write property test for health voice scripts (Property 8)
    - **Property 8: Health Voice Scripts Contain Patient Name and Are Subtype-Appropriate**
    - For each verified health subtype, generate arbitrary `PATIENT_NAME` strings; assert `voice_script` contains `patient_name` and a subtype-relevant activity keyword
    - **Validates: Requirements 3.2, 3.4**

  - [x] 5.4 Write property test for unverified health events (Property 6)
    - **Property 6: Unverified Health Events Produce Empty Voice Scripts and No Audio**
    - Generate health Events with `verified=False`; assert `generate_voice_script` returns `""` and that the ElevenLabs client is never called when `voice_script` is empty
    - **Validates: Requirements 3.3, 4.1**

- [x] 6. Checkpoint — core logic verified
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement ElevenLabs streaming TTS service
  - [x] 7.1 Create `services/brain/services/elevenlabs.py`
    - Implement `synthesize_audio(voice_script: str, voice_id: str, client: ElevenLabs) -> io.BytesIO`
    - Call `client.text_to_speech.convert(text=voice_script, voice_id=voice_id, model_id="eleven_flash_v2_5")`
    - Iterate chunks into `io.BytesIO` buffer; seek to 0 before returning
    - Wrap entire call in `asyncio.wait_for(..., timeout=15.0)`; on `TimeoutError` or any exception log and return `None`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 7.2 Write property test for no temp files (Property 9)
    - **Property 9: ElevenLabs Streaming Produces Audio Without Temp Files**
    - Mock the ElevenLabs client to yield arbitrary byte chunks; call `synthesize_audio`; assert no file matching `audio/*.mp3` or `audio/*.wav` exists on disk before, during, or after the call
    - **Validates: Requirements 4.1, 5.1**

- [x] 8. Implement Pygame audio playback service
  - [x] 8.1 Create `services/brain/services/audio.py`
    - Implement `init_pygame(device: str) -> None`: call `pygame.mixer.pre_init()` targeting `device`; call `pygame.mixer.init()`; on any exception log warning and attempt `pygame.mixer.init()` with default device; log warning on fallback
    - Implement `play_audio(buffer: io.BytesIO) -> None`: call `pygame.mixer.music.load(buffer)`, `pygame.mixer.music.play()`, then `while pygame.mixer.music.get_busy(): pygame.time.wait(50)`; on any exception log error and return without raising
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 8.2 Write unit tests for Pygame fallback logic
    - Test that when the target device raises on init, `init_pygame` falls back to default and logs a warning
    - Test that `play_audio` catches exceptions and does not re-raise
    - _Requirements: 5.4, 5.6_

- [x] 9. Implement Motor MongoDB service
  - [x] 9.1 Create `services/brain/services/mongodb.py`
    - Implement `init_motor(uri: str) -> AsyncIOMotorClient`
    - Implement `async verify_mongodb(client: AsyncIOMotorClient) -> bool`: ping `admin` db; return `True` on success, `False` on exception (log warning)
    - Implement `async write_event_record(record: EventRecord, client: AsyncIOMotorClient, db: str, collection: str) -> bool`
    - Use `asyncio.wait_for(..., timeout=5.0)` around the insert; on `TimeoutError` or any exception log error and return `False`
    - Serialize `record` with `record.model_dump()` — `image_b64` must not be present (enforced by `EventRecord` model)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 9.2 Write property test for EventRecord transformation (Property 10)
    - **Property 10: EventRecord Excludes image_b64 and Includes All Enrichment Fields**
    - Generate arbitrary `Event` instances containing `image_b64`; construct `EventRecord` from them; assert `"image_b64"` is absent from `record.model_dump()` and that `verified`, `voice_script`, `processing_status`, `processed_at` are all present
    - **Validates: Requirements 6.2, 6.6, 16.5**

- [x] 10. Implement POST /event route and full processing pipeline
  - [x] 10.1 Create `services/brain/routes/event.py`
    - Define `router = APIRouter()`; implement `POST /event` handler accepting `event: Event`
    - Store `event.event_id` on `request.state.event_id` for the global exception handler
    - Orchestrate the pipeline in order:
      1. Gemini verification (health only, `verified=True` for identity)
      2. Voice script generation
      3. ElevenLabs synthesis (skip if `voice_script == ""`)
      4. Pygame playback (skip if synthesis returned `None`)
      5. MongoDB write; set `processing_status="partial_failure"` if write returns `False`
    - Absorb all downstream exceptions internally; never propagate to FastAPI error handler
    - Return `EventResponse(event_id=event.event_id, status="processed", message=...)` with HTTP 200
    - `message` = `"Event processed successfully."` when no failures; `"Event processed with partial failures."` when any step failed
    - _Requirements: 1.1, 1.5, 1.6, 8.2, 8.3, 8.4_

  - [x] 10.2 Write property test for valid/invalid event acceptance (Property 1)
    - **Property 1: Valid Events Are Accepted, Invalid Events Are Rejected**
    - Generate valid `Event` instances → assert HTTP 200
    - Generate payloads with missing required fields or wrong types → assert HTTP 422
    - **Validates: Requirements 1.2, 1.4, 16.2**

  - [x] 10.3 Write property test for event_id echo (Property 2)
    - **Property 2: HTTP 200 Response Always Contains Echoed event_id**
    - Generate valid `Event` instances with mocked downstream; assert `response.json()["event_id"] == event.event_id`
    - **Validates: Requirements 1.5, 8.2**

  - [x] 10.4 Write property test for identity events skipping Gemini (Property 3)
    - **Property 3: Identity Events Always Set verified=True Without Calling Gemini**
    - Generate Events with `type="identity"`; assert Gemini client is never called and the resulting `EventRecord.verified` is `True`
    - **Validates: Requirements 2.4**

  - [x] 10.5 Write property test for downstream failures never producing HTTP 5xx (Property 11)
    - **Property 11: Downstream Failures Never Produce HTTP 5xx**
    - Inject failures (raise `Exception`) into Gemini, ElevenLabs, Pygame, and MongoDB mocks in all combinations; assert every response has status code 200
    - **Validates: Requirements 8.3, 8.4, 20.2, 20.3, 20.4**

- [x] 11. Implement GET /health route
  - [x] 11.1 Create `services/brain/routes/health.py`
    - Define `router = APIRouter()`; implement `GET /health` handler
    - Attempt `await motor_client.admin.command("ping")` with a 3-second timeout
    - Return `HealthResponse(status="ok")` with HTTP 200 on success
    - Return `JSONResponse(status_code=503, content={"status": "degraded", "reason": "mongodb_unreachable"})` on failure
    - _Requirements: 7.4, 7.5_

  - [x] 11.2 Write unit tests for health endpoint
    - Test HTTP 200 `{"status": "ok"}` when Motor ping succeeds
    - Test HTTP 503 `{"status": "degraded", "reason": "mongodb_unreachable"}` when Motor ping raises
    - _Requirements: 7.4, 7.5_

- [x] 12. Checkpoint — full pipeline wired
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Enforce Content-Type and API contract compliance
  - [x] 13.1 Verify all responses from `/event` and `/health` carry `Content-Type: application/json`
    - FastAPI returns this by default for `JSONResponse` and Pydantic response models; confirm the global exception handler also uses `JSONResponse`
    - Add `response_class=JSONResponse` to any route that might return a plain `dict`
    - _Requirements: 8.1_

  - [x] 13.2 Write property test for Content-Type header (Property 12)
    - **Property 12: All Brain Responses Have Content-Type application/json**
    - Send valid events, invalid events, and requests that trigger the global exception handler; assert every response has `Content-Type: application/json`
    - **Validates: Requirements 8.1**

- [x] 14. Wire `services/brain/models.py` and finalize imports
  - Create `services/brain/models.py` re-exporting `Event`, `EventRecord`, `EventResponse`, `HealthResponse` from `shared/contract.py` for clean intra-service imports
  - Confirm all route and service modules import from `services/brain/models` (not directly from `shared/`)
  - Run `python -m pytest tests/brain/ -x` and fix any import or type errors
  - _Requirements: 16.2, 16.3_

- [x] 15. Final checkpoint — all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All property tests use `hypothesis` with `@settings(max_examples=100)` and are tagged with their property number from the design document
- Downstream failures (Gemini, ElevenLabs, Pygame, MongoDB) are always absorbed — the Brain never returns HTTP 5xx for integration failures
- `image_b64` exclusion from `EventRecord` is enforced at the model level, not by runtime deletion
- The ElevenLabs streaming implementation uses an in-memory `io.BytesIO` buffer — no temp files are written at any point
- Pygame device selection happens at `init_pygame()` time (lifespan startup), not per-request
