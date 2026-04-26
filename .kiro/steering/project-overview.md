# AuraGuard AI — Project Overview

AuraGuard AI is a life-critical assistive platform for Alzheimer's patients built for Hackabull VII. Three Python services run on a single laptop and communicate through a shared JSON Contract.

## Services

| Service | Directory | Port | Tech |
|---------|-----------|------|------|
| Vision Engine | `services/vision/` | 5000 | Python, OpenCV, face_recognition, Flask |
| AI Brain | `services/brain/` | 8000 | FastAPI, Uvicorn, Pydantic, Motor |
| Caregiver Portal | `services/dashboard/` | 8501 | Streamlit, Plotly, Pandas |

## Key External APIs

- **Google Gemini** — health item detection (Vision Engine) and secondary verification (Brain)
- **ElevenLabs** — streaming TTS using `eleven_flash_v2_5` model (~75ms to first chunk)
- **MongoDB Atlas** — event storage via Motor async driver
- **Snowflake** — longitudinal health trend data for the dashboard

## Shared Contract

`shared/contract.py` is the single source of truth for all Pydantic models. All three services import from it. Never duplicate model definitions across services.

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in secrets
cp .env.example .env

# Launch all three services
python run_all.py
```

Individual service launch:
```bash
uvicorn services.brain.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

All secrets and config live in a single `.env` at the repo root. See `.env.example` for all required variables. Never hardcode secrets.

## Project Structure

```
shared/
  contract.py           # Canonical Pydantic models (Event, EventRecord, etc.)
services/
  brain/
    main.py             # FastAPI app + lifespan
    config.py           # Pydantic-settings Settings model
    models.py           # Re-exports from shared/contract.py
    routes/
      event.py          # POST /event
      health.py         # GET /health
    services/
      gemini.py         # Gemini verification + voice script generation
      elevenlabs.py     # ElevenLabs streaming TTS
      audio.py          # Pygame playback
      mongodb.py        # Motor async writes
  vision/               # Vision Engine (Flask)
  dashboard/            # Caregiver Portal (Streamlit)
tests/
  brain/
    conftest.py         # TestClient fixture with mocked downstream
  vision/
  integration/
run_all.py              # Launches all three services as subprocesses
```
