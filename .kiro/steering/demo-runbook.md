---
inclusion: manual
---

# AuraGuard AI — Demo Runbook (Hackabull VII)

Quick-reference for demo day. Follow this order every time.

## Pre-Demo Checklist (30 minutes before)

- [ ] `.env` filled in with all real API keys (Gemini, ElevenLabs, MongoDB Atlas, Snowflake)
- [ ] At least one face + JSON profile in `known_faces/` (see `known-faces-setup.md`)
- [ ] Meta Smart Glasses charged and paired via ADB (see `hardware-mirror.md`)
- [ ] scrcpy mirror window open and showing the glasses POV
- [ ] `VIDEO_SOURCE` in `.env` set to the correct capture index
- [ ] `GLASSES_AUDIO_DEVICE` in `.env` matches the exact glasses speaker device name
- [ ] MongoDB Atlas cluster is running and accessible
- [ ] Snowflake warehouse is running (resume it if suspended)
- [ ] Run `python -m pytest tests/brain/ -x -q` — all tests must pass

## Launch Sequence

```bash
# 1. Install / verify dependencies
pip install -r requirements.txt

# 2. Start all services
python run_all.py
```

Wait for all three services to print their startup messages:
- Brain: `Uvicorn running on http://0.0.0.0:8000`
- Vision Engine: `Vision Engine started`
- Dashboard: `You can now view your Streamlit app in your browser`

## Verify Everything is Up

```bash
# Brain health check
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Dashboard
open http://localhost:8501
```

## Demo Script (5-minute walkthrough)

### Scene 1 — Face Recognition (2 min)
1. Have a person whose photo is in `known_faces/` walk into the glasses' field of view
2. Point out: Vision Engine detects the face → POSTs identity event to Brain
3. Brain generates personalized voice script → ElevenLabs synthesizes → plays through glasses speaker
4. Show the Caregiver Portal: new **green** identity event card appears within 5 seconds
5. Read the voice script aloud from the portal card

### Scene 2 — Health Detection (2 min)
1. Have the patient pick up a glass of water (or hold a water bottle)
2. Point out: Vision Engine sends frame to Gemini → detects `drinking` → POSTs health event
3. Brain runs secondary Gemini verification → generates positive reinforcement script
4. Audio plays: *"Good job, [Patient Name]. I can see you are drinking water. Stay hydrated."*
5. Show the Caregiver Portal: new **yellow** health event card with `verified: true`

### Scene 3 — Graceful Degradation (1 min)
1. Kill the Brain process (`Ctrl+C` on its terminal)
2. Show the Vision Engine continuing to run without crashing
3. Restart the Brain: `uvicorn services.brain.main:app --host 0.0.0.0 --port 8000`
4. Events resume flowing immediately — no data loss

## If Something Goes Wrong

| Symptom | Fix |
|---------|-----|
| Brain returns `{"status": "degraded"}` | MongoDB Atlas cluster may be paused — resume it in the Atlas console |
| No audio from glasses | Check `GLASSES_AUDIO_DEVICE` in `.env`; run the audio device finder script in `hardware-mirror.md` |
| Vision Engine not detecting faces | Ensure `known_faces/` has at least one `.jpg` + `.json` pair; check lighting |
| ElevenLabs synthesis fails | Verify `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` in `.env`; check ElevenLabs quota |
| Caregiver Portal shows no events | Check MongoDB Atlas network access list — add `0.0.0.0/0` for the demo |
| Snowflake chart is blank | Resume the Snowflake warehouse; check `SNOWFLAKE_*` env vars |
| `run_all.py` exits immediately | Run each service individually to see the specific error |

## Individual Service Restart Commands

```bash
# Brain only
uvicorn services.brain.main:app --host 0.0.0.0 --port 8000

# Vision Engine only
python -m services.vision.main

# Dashboard only
streamlit run services/dashboard/app.py --server.port 8501
```

## Ports Reference

| Service | URL |
|---------|-----|
| AI Brain API | http://localhost:8000 |
| Brain Health Check | http://localhost:8000/health |
| Caregiver Portal | http://localhost:8501 |
| Vision Engine | http://localhost:5000 |

## Key Talking Points for Judges

- **Life-critical context**: Alzheimer's patients face daily safety risks — stove left on, not recognizing family, forgetting medication
- **Two-pass verification**: Vision Engine detects → Brain verifies with Gemini → prevents false alerts before audio fires
- **~75ms to first audio**: `eleven_flash_v2_5` model chosen specifically for low-latency patient alerts
- **Graceful degradation**: Every downstream failure is absorbed; the system never crashes due to an API being unavailable
- **No image storage**: `image_b64` is intentionally excluded from MongoDB at the model level — privacy by design
- **Longitudinal data**: Snowflake enables caregivers to share meaningful health patterns with clinicians
