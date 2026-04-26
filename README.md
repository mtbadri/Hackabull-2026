# AuraGuard AI 🧠👓
### *Life-Critical Assistive Platform for Alzheimer's Patients*

> A wearable AI co-pilot that watches, understands, and speaks — so the patient is never truly alone.

**Hackabull VII — Tech For Good | Health Care & Wellness**

---

## The Problem

Over 6 million Americans live with Alzheimer's disease, and 16+ million family members provide unpaid care. Patients face daily life-threatening risks — leaving the stove on, not recognizing family members, forgetting to hydrate or take medication. Caregivers cannot be present 24/7.

## Our Solution

AuraGuard AI transforms Meta Smart Glasses into a life-critical safety system. The glasses stream a first-person POV to a laptop running three coordinated services: a **Vision Engine** that detects hazards and familiar faces, an **AI Brain** that reasons about what it sees and speaks empathetically to the patient, and a **Caregiver Portal** that gives families and clinicians a live and longitudinal view of their loved one's day.

### Core Features

- **Face Recognition** — identifies familiar people and tells the patient who they are, how they know them, and what they last talked about
- **Health Item Detection** — detects eating, drinking, and medication intake using Google Gemini multimodal AI
- **Empathetic Voice Alerts** — synthesizes personalized, calming speech via ElevenLabs and plays it through the glasses speaker
- **Real-Time Caregiver Dashboard** — live event feed from MongoDB Atlas with color-coded health and identity events
- **Longitudinal Health Trends** — time-series charts from Snowflake so caregivers can share meaningful data with clinicians
- **Graceful Degradation** — each service continues operating independently if another fails

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Wearable Hardware | Meta Smart Glasses (POV stream via scrcpy/ADB) |
| Vision Engine | Python, OpenCV, `face_recognition`, Flask |
| AI Reasoning | Google Gemini (multimodal) |
| Voice Synthesis | ElevenLabs |
| Audio Playback | Pygame |
| AI Brain API | FastAPI, Uvicorn, Pydantic |
| Live Database | MongoDB Atlas + Motor (async driver) |
| Data Warehouse | Snowflake |
| Caregiver Dashboard | Streamlit, Plotly, Pandas |

---

## Demo

> 📹 **[Watch the Demo Video](#)** — walks through face recognition, health detection, voice alerts, and the caregiver dashboard in real time.

*(Link will be updated after submission)*

### How to Run the Demo

#### Prerequisites

- Python 3.10+
- Meta Smart Glasses paired and connected to the laptop via ADB/scrcpy (see `docs/hardware_mirror.md`)
- All API keys filled in (Gemini, ElevenLabs, MongoDB Atlas, Snowflake)
- Known faces directory populated with at least one reference image + profile JSON

#### Step 1 — Environment Setup

```bash
# Clone the repo and install dependencies
pip install -r requirements.txt

# Copy the example env file and fill in your keys
cp .env.example .env
```

Open `.env` and set every value. The minimum required keys for a full demo are:

| Variable | What it does |
|----------|-------------|
| `GEMINI_API_KEY` | Powers health item detection and secondary verification |
| `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID` | Synthesizes the patient's voice alerts |
| `MONGODB_URI` / `MONGODB_DB` / `MONGODB_COLLECTION` | Stores events for the live dashboard |
| `SNOWFLAKE_*` | Feeds the longitudinal health trend charts |
| `PATIENT_NAME` / `PATIENT_ID` | Personalizes every voice alert (e.g. `Ismail`, `patient_001`) |
| `GLASSES_AUDIO_DEVICE` | Routes audio to the glasses speaker |

#### Step 2 — Mirror the Glasses to the Laptop

Follow `docs/hardware_mirror.md` to establish the scrcpy/ADB mirror. The Vision Engine reads from this mirrored video source. Once the mirror window is visible on screen, you're ready.

#### Step 3 — Launch All Services

```bash
python run_all.py
```

This starts all three services simultaneously:

| Service | URL |
|---------|-----|
| Vision Engine | `http://localhost:5000` |
| AI Brain | `http://localhost:8000` |
| Caregiver Portal | `http://localhost:8501` |

Open the Caregiver Portal in your browser at `http://localhost:8501` — you'll see the live event feed and health trend charts load within a few seconds.

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

## Quick Start

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp .env.example .env

# 3. Launch everything
python run_all.py
```

---

## The JSON Contract

Every event flows through this shared schema:

```json
{
  "event_id": "uuid-v4",
  "timestamp": "2025-04-25T14:32:00Z",
  "patient_id": "patient_001",
  "type": "health | identity",
  "subtype": "eating | drinking | medicine_taken | face_recognized",
  "confidence": 0.91,
  "image_b64": "<base64-encoded-frame>",
  "metadata": {
    "person_profile": {
      "name": "Hussain",
      "relationship": "son",
      "background": "Software engineer living in Tampa.",
      "last_conversation": "Told you about his new job."
    }
  },
  "source": "vision_engine_v1"
}
```

---

## Team

| Role | Service | Port |
|------|---------|------|
| Vision Lead | `services/vision/` | 5000 |
| AI Architect | `services/brain/` | 8000 |
| Dashboard Lead | `services/dashboard/` | 8501 |

---

## Hardware Mirror (Meta Smart Glasses → Laptop)

See `docs/hardware_mirror.md` for step-by-step instructions to display the glasses POV live on the laptop for judges.

---

## Impact

Reduced safety incidents, earlier health intervention, and restored dignity for patients who deserve to live independently for as long as possible.

---

*Last updated: 2025-04-25*
