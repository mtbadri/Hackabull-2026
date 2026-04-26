# AuraGuard AI 🧠👓
### *Life-Critical Assistive Platform for Alzheimer's Patients*

> A wearable AI co-pilot that watches, understands, and speaks — so the patient is never truly alone.

**Hackabull VII — Tech For Good | Health Care & Wellness**

---

## The Problem

Over 6 million Americans live with Alzheimer's disease, and 16+ million family members provide unpaid care. Patients face daily life-threatening risks — leaving the stove on, not recognizing family members, forgetting to hydrate or take medication. Caregivers cannot be present 24/7.

## Our Solution

AuraGuard AI transforms Meta Smart Glasses into a life-critical safety system. The glasses stream a first-person POV to a laptop running three coordinated services: a **Vision Engine** that detects hazards and familiar faces, an **AI Brain** that reasons about what it sees and speaks empathetically to the patient, and a **Caregiver Portal** that gives families and clinicians a live and longitudinal view of their loved one's day.

---

## Branch Guide — Who Owns What

This project is split across feature branches. Each branch owns one service. They all merge into `dev` for integration.

| Branch | Owner | Service | Port | Status |
|--------|-------|---------|------|--------|
| `Ismail` | Ismail | **AI Brain** (`services/brain/`) | 8000 | ✅ Complete |
| `mohammed` | Mohammed | Vision Engine (`services/vision/`) | 5000 | In progress |
| `taikhoom` | Taikhoom | Caregiver Portal (`services/dashboard/`) | 8501 | In progress |
| `dev` | All | Integration branch | — | Merges all three |
| `main` | All | Production / demo branch | — | Stable |

**You are on the `Ismail` branch.** This branch contains the fully implemented and tested AI Brain service — the central coordinator of the entire pipeline.

---

## What the `Ismail` Branch Contains

The AI Brain is a FastAPI service that sits between the Vision Engine and everything else. It receives structured event payloads, reasons about them, speaks to the patient, and logs everything to MongoDB.

### Files Implemented on This Branch

```
shared/
  contract.py               # Canonical Pydantic models shared by all three services
                            # (Event, EventRecord, EventResponse, HealthResponse,
                            #  PersonProfile, IdentityMetadata, HealthMetadata)

services/brain/
  main.py                   # FastAPI app + lifespan startup/shutdown
  config.py                 # Pydantic-settings: loads and validates all 9 env vars
  models.py                 # Re-exports from shared/contract.py for clean imports

  routes/
    event.py                # POST /event — full processing pipeline
    health.py               # GET /health — MongoDB connectivity check

  services/
    gemini.py               # Gemini verification + voice script generation
    elevenlabs.py           # ElevenLabs streaming TTS → in-memory BytesIO buffer
    audio.py                # Pygame mixer: device selection + audio playback
    mongodb.py              # Motor async client: EventRecord writes

tests/brain/
  conftest.py               # TestClient fixture with all downstream services mocked
  test_config.py            # Config validation (missing env vars → sys.exit(1))
  test_lifespan.py          # Startup degradation (MongoDB/Pygame failure → warning only)
  test_gemini.py            # Properties 4 & 5: prompt specificity + response parsing
  test_voice_scripts.py     # Properties 6, 7 & 8: voice script correctness
  test_audio.py             # Property 9: no temp files + Pygame fallback logic
  test_event_record.py      # Property 10: EventRecord excludes image_b64
  test_event_route.py       # Properties 1, 2, 3 & 11: HTTP contract + pipeline
  test_health_route.py      # Health endpoint unit tests
  test_api_contract.py      # Property 12: Content-Type: application/json on all responses
```

### What Is NOT on This Branch

The Vision Engine (`services/vision/`) and Caregiver Portal (`services/dashboard/`) live on the `mohammed` and `taikhoom` branches respectively. This branch only contains the Brain and the shared contract. When all three branches merge into `dev`, the full system runs together.

---

## How the Brain Fits Into the Full System

```
Meta Smart Glasses (POV stream)
        │
        ▼
┌─────────────────────────────────┐
│  Vision Engine  (mohammed)      │  Flask · :5000
│  - Face recognition             │
│  - Health item detection        │
│    via Gemini                   │
└────────────────┬────────────────┘
                 │  POST /event  (JSON Contract)
                 ▼
┌─────────────────────────────────┐   ◄── THIS BRANCH
│  AI Brain  (Ismail)             │  FastAPI · :8000
│  1. Validate event payload      │
│  2. Gemini secondary verify     │
│  3. Generate voice script       │
│  4. ElevenLabs TTS synthesis    │
│  5. Pygame → glasses speaker    │
│  6. Write EventRecord to Mongo  │
│  → Always returns HTTP 200      │
└────────────────┬────────────────┘
                 │  Motor async write
                 ▼
┌─────────────────────────────────┐
│  MongoDB Atlas                  │  event_records collection
└────────────────┬────────────────┘
                 │  pymongo read (every 5s)
                 ▼
┌─────────────────────────────────┐
│  Caregiver Portal  (taikhoom)   │  Streamlit · :8501
│  - Live event feed              │
│  - Snowflake health trends      │
└─────────────────────────────────┘
```

The Brain is the only service that writes to MongoDB. The Caregiver Portal only reads. The Vision Engine only writes to the Brain. This means **the Brain can be developed and tested in complete isolation** — which is exactly what this branch does.

---

## The JSON Contract

`shared/contract.py` is the single source of truth for all data models. All three services import from it. Never duplicate these definitions.

### Event (Vision Engine → Brain)

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-04-25T14:32:00Z",
  "patient_id": "patient_001",
  "type": "health",
  "subtype": "drinking",
  "confidence": 0.91,
  "image_b64": "<base64-encoded-JPEG-frame>",
  "metadata": {
    "detected_item": "water"
  },
  "source": "vision_engine_v1"
}
```

For identity events, `metadata` contains a `person_profile` instead:

```json
{
  "type": "identity",
  "subtype": "face_recognized",
  "metadata": {
    "person_profile": {
      "name": "Hussain",
      "relationship": "son",
      "background": "Software engineer living in Tampa.",
      "last_conversation": "Told you about his new job at a tech startup."
    }
  }
}
```

### EventRecord (Brain → MongoDB Atlas)

Same as Event but with `image_b64` removed and four enrichment fields added:

```json
{
  "event_id": "...",
  "timestamp": "...",
  "patient_id": "...",
  "type": "health",
  "subtype": "drinking",
  "confidence": 0.91,
  "metadata": { "detected_item": "water" },
  "source": "vision_engine_v1",
  "verified": true,
  "voice_script": "Good job, Ismail. I can see you are drinking water. Stay hydrated.",
  "processing_status": "success",
  "processed_at": "2025-04-25T14:32:01.234Z"
}
```

`image_b64` is intentionally excluded at the model level — it never reaches MongoDB.

---

## Running the Brain in Isolation (This Branch)

You do not need the Vision Engine or Caregiver Portal to run or test the Brain. Everything below works on this branch alone.

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in the Brain's required variables. The minimum set to start the Brain:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for health event verification |
| `ELEVENLABS_API_KEY` | ✅ | ElevenLabs API key for voice synthesis |
| `ELEVENLABS_VOICE_ID` | ✅ | Voice ID to use (e.g. `21m00Tcm4TlvDq8ikWAM` for Rachel) |
| `MONGODB_URI` | ✅ | MongoDB Atlas connection string |
| `MONGODB_DB` | ✅ | Database name (e.g. `auraguard`) |
| `MONGODB_COLLECTION` | ✅ | Collection name (e.g. `events`) |
| `PATIENT_NAME` | ✅ | Patient's first name for voice scripts (e.g. `Ismail`) |
| `PATIENT_ID` | ✅ | Patient identifier (e.g. `patient_001`) |
| `GLASSES_AUDIO_DEVICE` | ✅ | Audio device name for the glasses speaker |

If any of these are missing at startup, the Brain logs the missing variable and exits with code 1.

### 3. Start the Brain

```bash
uvicorn services.brain.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     MongoDB connectivity verified.
INFO:     AI Brain startup complete.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If MongoDB is unreachable, you'll see a `WARNING` instead — the Brain still starts in degraded state and will accept events.

### 4. Verify It's Running

```bash
curl http://localhost:8000/health
```

Expected responses:

```json
{ "status": "ok" }                                          # MongoDB reachable
{ "status": "degraded", "reason": "mongodb_unreachable" }  # MongoDB down
```

### 5. Send a Test Event

You can POST a synthetic event directly to the Brain without the Vision Engine:

```bash
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-04-25T14:32:00Z",
    "patient_id": "patient_001",
    "type": "identity",
    "subtype": "face_recognized",
    "confidence": 0.95,
    "image_b64": "aGVsbG8=",
    "metadata": {
      "person_profile": {
        "name": "Hussain",
        "relationship": "son",
        "background": "Software engineer living in Tampa.",
        "last_conversation": "Told you about his new job at a tech startup."
      }
    },
    "source": "vision_engine_v1"
  }'
```

Expected response:

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processed",
  "message": "Event processed successfully."
}
```

The Brain will:
1. Skip Gemini (identity events are always `verified: true`)
2. Generate the voice script: *"Ismail, Hussain is here. Hussain is your son. Software engineer living in Tampa. Last time you spoke, Told you about his new job at a tech startup."*
3. Call ElevenLabs to synthesize the script
4. Play the audio through the configured audio device
5. Write the EventRecord to MongoDB (without `image_b64`)
6. Return HTTP 200

---

## Running the Tests

All tests are isolated — no real API calls are made. Every downstream service (Gemini, ElevenLabs, MongoDB, Pygame) is mocked.

```bash
# Run all Brain tests
python -m pytest tests/brain/ -v

# Run with coverage report
python -m pytest tests/brain/ --cov=services/brain --cov=shared -v

# Stop on first failure
python -m pytest tests/brain/ -x

# Run a single test file
python -m pytest tests/brain/test_gemini.py -v
```

Expected output:

```
tests/brain/test_api_contract.py::test_valid_event_response_has_json_content_type PASSED
tests/brain/test_api_contract.py::test_invalid_event_422_has_json_content_type PASSED
tests/brain/test_api_contract.py::test_health_ok_has_json_content_type PASSED
tests/brain/test_api_contract.py::test_health_degraded_has_json_content_type PASSED
tests/brain/test_api_contract.py::test_unhandled_exception_500_has_json_content_type PASSED
...
66 passed in 0.36s
```

### What the Tests Cover

The test suite uses both unit tests and property-based tests (`hypothesis`). Each property test is tagged with its property number from the design document.

| Test File | What It Tests |
|-----------|--------------|
| `test_config.py` | Missing any of the 9 env vars causes `sys.exit(1)` |
| `test_lifespan.py` | MongoDB/Pygame failure at startup logs WARNING, does not crash |
| `test_gemini.py` | **Property 4**: prompts are subtype-specific; **Property 5**: YES/NO parsing is deterministic |
| `test_voice_scripts.py` | **Property 6**: unverified health → empty script; **Property 7**: identity scripts contain all profile fields; **Property 8**: health scripts contain patient name + activity keyword |
| `test_audio.py` | **Property 9**: no temp files written; Pygame device fallback logic |
| `test_event_record.py` | **Property 10**: `image_b64` absent from EventRecord; enrichment fields present |
| `test_event_route.py` | **Property 1**: valid events → 200, invalid → 422; **Property 2**: `event_id` echoed; **Property 3**: identity events skip Gemini; **Property 11**: downstream failures never produce 5xx |
| `test_health_route.py` | HTTP 200 when MongoDB reachable; HTTP 503 when unreachable or timeout |
| `test_api_contract.py` | **Property 12**: every response has `Content-Type: application/json` |

---

## The Processing Pipeline (POST /event)

When the Brain receives an event, it runs this pipeline synchronously before returning HTTP 200. Every step is wrapped in error handling — a failure in any step is logged and absorbed, never propagated.

```
POST /event
    │
    ├─ 1. Pydantic validation (automatic)
    │      → HTTP 422 if schema invalid
    │
    ├─ 2. Gemini verification (health events only, 10s timeout)
    │      → verified = True/False
    │      → identity events: always verified = True, Gemini never called
    │      → on failure: verified = False, continue
    │
    ├─ 3. Voice script generation
    │      → identity: "Ismail, Hussain is here. He is your son..."
    │      → health + verified=True: "Good job, Ismail. I can see you are drinking water..."
    │      → health + verified=False: "" (empty — no audio)
    │
    ├─ 4. ElevenLabs TTS synthesis (if voice_script non-empty, 15s timeout)
    │      → streams audio chunks into in-memory BytesIO buffer
    │      → no files written to disk
    │      → on failure: skip playback, continue
    │
    ├─ 5. Pygame audio playback (if synthesis succeeded)
    │      → routes to GLASSES_AUDIO_DEVICE or falls back to default
    │      → blocks until playback complete
    │      → on failure: log error, continue
    │
    ├─ 6. MongoDB write (Motor async, 5s timeout)
    │      → writes EventRecord (without image_b64)
    │      → on failure: processing_status = "partial_failure", continue
    │
    └─ 7. Return HTTP 200
           → "Event processed successfully."     (all steps succeeded)
           → "Event processed with partial failures."  (any step failed)
```

The Brain **never returns HTTP 5xx for integration failures**. The only 5xx is HTTP 500 from the global exception handler for truly unhandled exceptions.

---

## Graceful Degradation

This is a life-critical system. Every downstream failure is absorbed.

| Failure | Brain Behavior |
|---------|---------------|
| Gemini timeout (>10s) | `verified = False`, skip voice for health events, continue |
| Gemini API error | Same as timeout |
| ElevenLabs timeout (>15s) | Skip audio playback, continue to MongoDB write |
| ElevenLabs API error | Same as timeout |
| Glasses speaker not found | Fall back to default system audio, log WARNING |
| Pygame init/play failure | Log ERROR, skip audio, continue to MongoDB write |
| MongoDB write timeout (>5s) | `processing_status = "partial_failure"`, return HTTP 200 |
| MongoDB write error | Same as timeout |
| MongoDB unreachable at startup | Log WARNING, start in degraded state |
| Pygame init failure at startup | Log WARNING, start in degraded state |

The only startup failure that causes `sys.exit(1)` is a missing required environment variable.

---

## Integrating With the Other Branches

### When Mohammed's Vision Engine Is Ready

The Vision Engine POSTs events to `http://localhost:8000/event`. No changes needed on this branch — the Brain is already listening. The Vision Engine just needs to:

1. Set `BRAIN_HOST=localhost` and `BRAIN_PORT=8000` in `.env`
2. Construct events that match the `Event` schema in `shared/contract.py`
3. POST to `/event` with `Content-Type: application/json`

The Brain will validate the payload and return HTTP 200 or 422. The Vision Engine should log the response and continue its capture loop regardless.

### When Taikhoom's Caregiver Portal Is Ready

The Portal reads `EventRecord` documents from MongoDB Atlas. The Brain already writes them. The Portal just needs to:

1. Connect to the same `MONGODB_URI`, `MONGODB_DB`, and `MONGODB_COLLECTION`
2. Query the collection ordered by `processed_at` descending
3. Render the fields defined in `EventRecord` in `shared/contract.py`

No API calls to the Brain are needed from the Portal — it reads directly from MongoDB.

### Merging Into `dev`

```bash
# From each feature branch, merge into dev
git checkout dev
git merge Ismail
git merge mohammed
git merge taikhoom
```

Once all three branches are merged, run the full system with:

```bash
python run_all.py
```

---

## Running the Full System (After Integration)

Once all three branches are merged into `dev`:

### Prerequisites

- Python 3.10+
- Meta Smart Glasses paired and connected via ADB/scrcpy
- All API keys filled in `.env`
- Known faces directory populated (see below)

### Known Faces Directory

The Vision Engine loads face encodings from a local directory at startup. Each person needs two files with matching base names:

```
known_faces/
  hussain.jpg          # Clear reference photo of the person's face
  hussain.json         # Person profile JSON
  dr_ahmed.jpg
  dr_ahmed.json
```

Profile JSON format:

```json
{
  "name": "Hussain",
  "relationship": "son",
  "background": "Software engineer living in Tampa.",
  "last_conversation": "He told you about his new job at a tech startup."
}
```

### Launch All Services

```bash
python run_all.py
```

This starts all three services simultaneously:

| Service | URL | Health Check |
|---------|-----|-------------|
| Vision Engine | `http://localhost:5000` | `GET /health` |
| AI Brain | `http://localhost:8000` | `GET /health` |
| Caregiver Portal | `http://localhost:8501` | Open in browser |

Open `http://localhost:8501` in a browser to see the live event feed.

---

## Demo Walkthrough

### Feature 1 — Face Recognition & Identity Alert

**Setup:** Add a photo and profile JSON for at least one person to `known_faces/`.

**Demo:** Have that person walk into the patient's field of view.

**What happens:**
1. Vision Engine matches the face against stored encodings and POSTs an `identity` event to the Brain
2. Brain skips Gemini (identity events are always verified), generates the voice script, synthesizes speech via ElevenLabs, and plays it through the glasses speaker
3. Patient hears: *"Ismail, Hussain is here. He is your son. Software engineer living in Tampa. Last time you spoke, he told you about his new job."*
4. Caregiver Portal shows a new **green** identity event card within 5 seconds

---

### Feature 2 — Health Item Detection

**Demo:** Have the patient pick up a glass of water, eat food, or hold a pill bottle in view.

**What happens:**
1. Vision Engine sends the frame to Gemini, detects the activity, POSTs a `health` event to the Brain
2. Brain sends the same frame to Gemini for **secondary verification** — a targeted yes/no question filters false positives
3. If verified, Brain generates a positive reinforcement script and plays it through the glasses speaker
4. Patient hears: *"Good job, Ismail. I can see you are drinking water. Stay hydrated."*
5. Caregiver Portal shows a new **yellow** health event card with `verified: true`

---

### Feature 3 — Graceful Degradation Under Failure

**Demo:** Kill the Brain process mid-demo (`Ctrl+C`), then watch the Vision Engine logs.

**What happens:**
- Vision Engine logs the failed POST and continues its capture loop — no crash
- Restart the Brain and events resume immediately
- Same pattern for Gemini or ElevenLabs failures: Brain logs the error, marks `processing_status: partial_failure`, returns HTTP 200

---

### Feature 4 — Caregiver Dashboard

**Demo:** Open `http://localhost:8501` while the system is running.

**What happens:**
- Live event feed polls MongoDB every 5 seconds, newest events first
- Color coding: 🟡 yellow = health event, 🟢 green = identity event
- Each card shows: timestamp, type, subtype, confidence, verified flag, voice script spoken, processing status
- Snowflake health trend charts show event frequency over time
- If MongoDB goes down mid-demo, the portal shows the last fetched data and a warning banner — no crash

---

## Architecture

```
Meta Smart Glasses (POV stream)
        │
        ▼
┌─────────────────────┐
│  Vision Engine      │  Python + OpenCV  :5000
│  - Face Recognition │
│  - Health Detection │
│    (Gemini)         │
└────────┬────────────┘
         │ POST /event  (JSON Contract)
         ▼
┌─────────────────────┐
│  AI Brain (FastAPI) │  Gemini + ElevenLabs  :8000
│  - Multimodal verify│
│  - Voice synthesis  │
│  - Pygame playback  │
└────────┬────────────┘
         │ Motor async write
         ▼
┌─────────────────────┐        ┌──────────────────────┐
│   MongoDB Atlas     │        │   Snowflake DW        │
│   (live events)     │        │   (health trends)     │
└────────┬────────────┘        └──────────┬───────────┘
         └──────────────┬─────────────────┘
                        ▼
              ┌──────────────────┐
              │ Caregiver Portal │  Streamlit  :8501
              │  Live Feed       │
              │  Trend Charts    │
              └──────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Wearable Hardware | Meta Smart Glasses (POV stream via scrcpy/ADB) |
| Vision Engine | Python, OpenCV, `face_recognition`, Flask |
| AI Reasoning | Google Gemini (multimodal) |
| Voice Synthesis | ElevenLabs (`eleven_flash_v2_5` — ~75ms to first audio chunk) |
| Audio Playback | Pygame |
| AI Brain API | FastAPI, Uvicorn, Pydantic v2 |
| Async DB Driver | Motor (async MongoDB driver) |
| Live Database | MongoDB Atlas |
| Data Warehouse | Snowflake |
| Caregiver Dashboard | Streamlit, Plotly, Pandas |
| Testing | pytest, hypothesis (property-based testing) |

---

## Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Run Brain only (this branch)
uvicorn services.brain.main:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
python -m pytest tests/brain/ -v

# Run full system (after integration)
python run_all.py

# Check Brain health
curl http://localhost:8000/health

# Send a test event
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{"event_id":"test-001","timestamp":"2025-04-25T14:00:00Z","patient_id":"patient_001","type":"identity","subtype":"face_recognized","confidence":0.95,"image_b64":"aGVsbG8=","metadata":{"person_profile":{"name":"Hussain","relationship":"son","background":"Software engineer.","last_conversation":"Told you about his new job."}},"source":"vision_engine_v1"}'
```

---

## Team

| Role | Branch | Service | Port |
|------|--------|---------|------|
| AI Architect | `Ismail` | `services/brain/` | 8000 |
| Vision Lead | `mohammed` | `services/vision/` | 5000 |
| Dashboard Lead | `taikhoom` | `services/dashboard/` | 8501 |

---

## Impact

Reduced safety incidents, earlier health intervention, and restored dignity for patients who deserve to live independently for as long as possible.

---

*Last updated: 2025-04-26 — Ismail branch (AI Brain service complete)*
