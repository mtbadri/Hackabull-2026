---
inclusion: always
---

# AuraGuard AI — Project Steering

## What This Project Is
AuraGuard AI is a life-critical assistive platform for Alzheimer's patients. It uses Meta Smart Glasses to stream a first-person POV to three coordinated Python microservices.

## Services
| Service | Location | Port | Purpose |
|---------|----------|------|---------|
| Vision Engine | `services/vision/` | 5000 | Face recognition + Gemini verification + ElevenLabs voice |
| AI Brain | `services/brain/` | 8000 | FastAPI orchestration (planned) |
| Caregiver Dashboard | `services/dashboard/` | 8501 | Streamlit live feed + health trends |

## Tech Stack
- **AI**: Google Gemini (multimodal), `google-generativeai`
- **Voice**: ElevenLabs + Pygame
- **Vision**: OpenCV, `face_recognition`
- **Backend**: FastAPI + Uvicorn
- **Dashboard**: Streamlit + Plotly + Pandas
- **Databases**: MongoDB Atlas (live events), Snowflake (health trends)
- **Config**: `python-dotenv`, `pydantic-settings`

## Environment Variables (all in `.env`)
- `GEMINI_API_KEY` — Google Gemini
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`
- `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`
- `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_WAREHOUSE`
- `PATIENT_NAME`, `PATIENT_ID`
- `VISION_HOST`, `VISION_PORT`, `BRAIN_HOST`, `BRAIN_PORT`

## Event Schema (MongoDB + inter-service)
```json
{
  "event_id": "uuid-v4",
  "timestamp": "ISO-8601",
  "patient_id": "patient_001",
  "type": "health | identity",
  "subtype": "eating | drinking | medicine_taken | face_recognized",
  "confidence": 0.91,
  "metadata": {},
  "verified": true,
  "voice_script": "...",
  "processing_status": "success",
  "processed_at": "ISO-8601"
}
```

## Known Faces
Stored in `services/vision/known_faces/`. Each person needs:
- `{name}.jpg` — reference photo
- `{name}.json` — profile with `name`, `relationship`, `background`, `last_conversation`

## Key Constants (Vision Engine)
- `GEMINI_PARALLEL_CALLS = 5` — parallel votes per match
- `CHECK_EVERY_N_FRAMES = 10` — frames between Gemini checks
- `COOLDOWN_SECONDS = 60` — min time between alerts for same person
- `CONFIDENCE_THRESHOLD = 0.65` — minimum confidence to accept match
