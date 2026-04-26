# AuraGuard AI — Graceful Degradation Rules

This is a life-critical system. Every downstream failure must be absorbed. The pipeline must never crash due to an external service being unavailable.

## The Golden Rule

The Brain always returns HTTP 200 to the Vision Engine for integration failures. HTTP 5xx is only for unhandled exceptions caught by the global exception handler.

## Failure Handling by Service

### Gemini (Brain)
- Timeout: 10 seconds via `asyncio.wait_for`
- On timeout or any exception: log `ERROR`, set `verified = False`, continue pipeline
- Never retry — move on immediately
- Health events with `verified = False` produce empty `voice_script` and skip ElevenLabs + Pygame

### ElevenLabs (Brain)
- Timeout: 15 seconds via `asyncio.wait_for`
- On timeout or any exception: log `ERROR`, skip audio playback, continue to MongoDB write
- Only called when `voice_script` is non-empty

### Pygame (Brain)
- Device selection at startup (`init_pygame`), not per-request
- If target device (`GLASSES_AUDIO_DEVICE`) fails: fall back to default system audio, log `WARNING`
- If fallback also fails: log `ERROR`, skip playback, continue
- `play_audio` must never raise — catch all exceptions internally

### MongoDB (Brain)
- Timeout: 5 seconds via `asyncio.wait_for`
- On timeout or any exception: log `ERROR`, set `processing_status = "partial_failure"`, return HTTP 200
- Startup connectivity check: log `WARNING` on failure, do not exit

### Brain (Vision Engine)
- On non-200 response or network error: log error with `event_id`, continue capture loop
- POST timeout: 30 seconds
- Never retry failed POSTs

### MongoDB (Caregiver Portal)
- On connection failure during refresh: display last successfully fetched data, show warning banner
- Do not crash or stop the refresh loop

### Snowflake (Caregiver Portal)
- On connection failure: display placeholder chart with warning message
- Continue rendering the live event feed uninterrupted

## processing_status Values

| Value | Meaning |
|-------|---------|
| `"success"` | All pipeline steps completed without error |
| `"partial_failure"` | One or more steps failed (Gemini, ElevenLabs, Pygame, or MongoDB) |

Set `processing_status = "partial_failure"` if any step fails. The response message changes accordingly:
- All success: `"Event processed successfully."`
- Any failure: `"Event processed with partial failures."`

## Startup Degradation

The Brain starts in degraded state rather than refusing to start if:
- MongoDB is unreachable at startup (log `WARNING`, continue)
- Pygame cannot initialize (log `WARNING`, continue)

The only startup failure that causes `sys.exit(1)` is missing required environment variables.
