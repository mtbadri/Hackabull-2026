# AuraGuard AI — Testing Strategy

## Test Runner

```bash
# Run all Brain tests
python -m pytest tests/brain/ -v

# Run with coverage
python -m pytest tests/brain/ --cov=services/brain --cov=shared -v

# Run a single test file
python -m pytest tests/brain/test_gemini.py -v

# Stop on first failure
python -m pytest tests/brain/ -x
```

## Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.hypothesis]
max_examples = 100
```

## Test Structure

```
tests/
  brain/
    conftest.py               # TestClient fixture, all downstream mocks
    test_config.py            # Task 2.2 — config validation unit tests
    test_lifespan.py          # Task 3.2 — startup/shutdown behavior
    test_gemini.py            # Tasks 4.2, 4.3 — Properties 4 & 5
    test_voice_scripts.py     # Tasks 5.2, 5.3, 5.4 — Properties 6, 7 & 8
    test_audio.py             # Tasks 7.2, 8.2 — Property 9 + Pygame unit tests
    test_event_record.py      # Task 9.2 — Property 10
    test_event_route.py       # Tasks 10.2–10.5 — Properties 1, 2, 3 & 11
    test_health_route.py      # Task 11.2 — health endpoint unit tests
    test_api_contract.py      # Task 13.2 — Property 12
  vision/
    test_event_builder.py     # Property 13
  integration/
    test_contract.py          # Property 14 — round-trip compatibility
```

## conftest.py Pattern

The `tests/brain/conftest.py` must mock all downstream services so no test makes real API calls:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from services.brain.main import app

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice-id")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGODB_DB", "test_db")
    monkeypatch.setenv("MONGODB_COLLECTION", "test_events")
    monkeypatch.setenv("PATIENT_NAME", "TestPatient")
    monkeypatch.setenv("PATIENT_ID", "test_patient_001")
    monkeypatch.setenv("GLASSES_AUDIO_DEVICE", "Test Device")

@pytest.fixture
def client(mock_settings):
    with TestClient(app) as c:
        yield c
```

## Property-Based Test Pattern

Every property test must:
1. Reference its property number in a comment
2. Use `@settings(max_examples=100)`
3. Use `hypothesis` strategies to generate inputs
4. Assert the universal property holds for all generated inputs

```python
from hypothesis import given, settings
from hypothesis import strategies as st

# Property 5: Gemini Response Parsing Is Deterministic
# Validates: Requirements 2.3
@given(st.text(min_size=1).filter(lambda s: s.strip().upper().startswith("YES")))
@settings(max_examples=100)
def test_parse_gemini_verified_yes(response_text):
    assert parse_gemini_verified(response_text) is True

@given(st.text().filter(lambda s: not s.strip().upper().startswith("YES")))
@settings(max_examples=100)
def test_parse_gemini_verified_no(response_text):
    assert parse_gemini_verified(response_text) is False
```

## Hypothesis Strategies for Domain Types

```python
from hypothesis import strategies as st
from shared.contract import Event, PersonProfile

# Generate valid PersonProfile instances
person_profile_strategy = st.builds(
    PersonProfile,
    name=st.text(min_size=1, max_size=50),
    relationship=st.text(min_size=1, max_size=30),
    background=st.text(min_size=1, max_size=200),
    last_conversation=st.text(min_size=1, max_size=200),
)

# Generate valid identity Events
identity_event_strategy = st.builds(
    Event,
    event_id=st.uuids().map(str),
    timestamp=st.just("2025-01-01T00:00:00Z"),
    patient_id=st.text(min_size=1, max_size=50),
    type=st.just("identity"),
    subtype=st.just("face_recognized"),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_b64=st.text(min_size=1),
    metadata=st.fixed_dictionaries({"person_profile": person_profile_strategy}),
    source=st.just("vision_engine_v1"),
)

# Generate valid health Events
health_event_strategy = st.builds(
    Event,
    event_id=st.uuids().map(str),
    timestamp=st.just("2025-01-01T00:00:00Z"),
    patient_id=st.text(min_size=1, max_size=50),
    type=st.just("health"),
    subtype=st.sampled_from(["eating", "drinking", "medicine_taken"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_b64=st.text(min_size=1),
    metadata=st.fixed_dictionaries({"detected_item": st.sampled_from(["food", "water", "medicine"])}),
    source=st.just("vision_engine_v1"),
)
```

## What Tests Must NOT Do

- Make real HTTP calls to Gemini, ElevenLabs, or MongoDB
- Write files to disk (assert this explicitly in Property 9)
- Use `time.sleep` — use `asyncio.sleep` or mock timers
- Share mutable state between test cases
- Depend on `.env` values — always use `monkeypatch` to set env vars
