# AuraGuard AI — Caregiver Dashboard Branch

> **Branch focus:** `services/dashboard/` — the Streamlit Caregiver Portal (port 8501)

This document covers how the dashboard works, how to demo it standalone, and how it fits into the broader AuraGuard AI system.

---

## What This Branch Delivers

This branch implements the **Caregiver Portal** — a Streamlit web app that gives family members and clinicians a live, auto-refreshing window into the patient's day. It is one of three coordinated services in AuraGuard AI.

The dashboard does three things:

1. **Live Event Feed** — polls MongoDB Atlas every 5 seconds and displays the 50 most recent events in a color-coded table (yellow = health, green = identity)
2. **Health Trends Chart** — renders a Plotly time-series chart placeholder (health trend data integration is not in this branch)
3. **Family Sync Sidebar** — lets caregivers upload a photo and profile for a new person so the Vision Engine will recognize them on next startup

---

## File Structure

```
services/dashboard/
├── app.py                        # Streamlit entry point — layout, refresh loop, sidebar
├── settings.py                   # Pydantic-settings config — loads all env vars from .env
├── data/
│   ├── __init__.py
│   ├── mongodb_reader.py         # get_mongo_client(), fetch_latest_events()
│   └── snowflake_reader.py       # not implemented on this branch
└── components/
    ├── __init__.py
    ├── event_feed.py             # render_event_feed() — DataFrame + color coding
    └── health_charts.py          # render_health_chart() — Plotly placeholder
```

---

## How the Code Works

### `settings.py` — Configuration

`DashboardSettings` is a `pydantic-settings` `BaseSettings` model. It reads every required variable from `.env` at startup. If any variable is missing, `get_settings()` logs the error and calls `sys.exit(1)` — the app will not start with incomplete config.

```python
class DashboardSettings(BaseSettings):
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_COLLECTION: str
    PATIENT_NAME: str
```

### `data/mongodb_reader.py` — Live Event Data

`fetch_latest_events(n=50)` opens a `pymongo` connection to MongoDB Atlas (TLS-verified via `certifi`), queries the configured collection sorted by `processed_at` descending, and returns `(list[dict], False)` on success.

On any exception — network failure, auth error, timeout — it returns `(_cached_data, True)`. The module-level `_cached_data` list holds the last successful result, so the dashboard never shows a blank table just because MongoDB had a hiccup.

```python
def fetch_latest_events(n: int = 50) -> tuple[list[dict], bool]:
    try:
        ...
        return (docs, False)   # (data, mongo_error=False)
    except Exception:
        return (_cached_data, True)  # (stale data, mongo_error=True)
```

### `components/event_feed.py` — Live Feed Table

`render_event_feed(events, mongo_error)` builds a pandas DataFrame from the raw event dicts, selects the seven display columns, and applies row-level background color via `df.style.apply()`:

| Event type | Row color |
|---|---|
| `health` | Yellow `#fff9c4` |
| `identity` | Green `#c8e6c9` |

If `mongo_error` is `True`, a `st.warning` banner appears above the table. If the events list is empty, an info message is shown instead of an empty table.

### `components/health_charts.py` — Plotly Chart

`render_health_chart(df, error)` renders a `px.line` chart when data is available. On this branch, health trend data is not integrated, so it always shows an empty titled placeholder chart so the tab layout stays intact.

### `app.py` — Main App

The entry point wires everything together. Execution order matters in Streamlit, so the file is structured carefully:

1. `st.set_page_config(...)` — must be the very first Streamlit call
2. `get_settings()` — fails fast if env vars are missing
3. Session state initialization — sets defaults for `events`, `mongo_error`, `health_df`, `last_refresh`
4. Data fetch — calls `fetch_latest_events()` on every rerun, storing results in `st.session_state`
5. Sidebar — Family Sync form (see below)
6. Header — patient name + last refresh timestamp
7. Tabs — `Live Feed` and `Health Trends`
8. `time.sleep(5)` + `st.rerun()` — drives the 5-second auto-refresh loop

#### Family Sync Sidebar

The sidebar lets a caregiver register a new person for the Vision Engine without touching the filesystem manually. On save:

- The uploaded image is written to `services/vision/known_faces/{name_slug}.jpg`
- A JSON profile is written to `services/vision/known_faces/{name_slug}.json`

The Vision Engine loads known faces at startup, so the new person will be recognized the next time the Vision Engine restarts.

#### Auto-Refresh

```python
time.sleep(5)
st.rerun()
```

Streamlit re-executes the entire script on `st.rerun()`. Because data fetches happen at the top of the script (not inside the tabs), every refresh cycle re-queries MongoDB and updates `st.session_state` before rendering. The 5-second sleep keeps the polling rate reasonable without hammering the database.

---

## How to Demo This Branch Standalone

You can run the dashboard without the Vision Engine or AI Brain — you just need a MongoDB Atlas connection with some event documents in it.

### Prerequisites

- Python 3.10+
- A MongoDB Atlas cluster with the `auraguard.events` collection (or whatever names you configure)
- API keys filled in `.env`

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Configure `.env`

```bash
cp .env.example .env
```

Fill in the MongoDB section at minimum. The other variables are required by `DashboardSettings` but can be placeholders on this branch:

```dotenv
# Required — must be real
MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/...
MONGODB_DB=auraguard
MONGODB_COLLECTION=events
PATIENT_NAME=Ismail
PATIENT_ID=patient_001

# Not used by the dashboard — can be placeholders
GEMINI_API_KEY=placeholder
ELEVENLABS_API_KEY=placeholder
ELEVENLABS_VOICE_ID=placeholder
GLASSES_AUDIO_DEVICE=placeholder
```

### Step 3 — Seed MongoDB with test events (optional)

If your collection is empty, run the test script to insert sample events:

```bash
python test_mongo.py
```

This inserts a few sample `health` and `identity` event documents so the live feed has something to display.

### Step 4 — Launch the dashboard

```bash
streamlit run services/dashboard/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

### What you should see

- **Live Feed tab** — a color-coded table of events (yellow rows for health events, green for identity). Refreshes every 5 seconds automatically.
- **Health Trends tab** — an empty Plotly chart placeholder (health trend data integration is not in this branch).
- **Sidebar** — a "Family Sync" form. Upload a photo, fill in the fields, and click Save. Check `services/vision/known_faces/` to confirm the `.jpg` and `.json` were written.
- **MongoDB warning** — disconnect your internet or use a bad URI to see the yellow warning banner appear above the table while stale cached data continues to display.

---

## Graceful Degradation

The dashboard is designed to never crash due to an external service being unavailable:

| Failure | Behavior |
|---|---|
| MongoDB unreachable | Shows last cached events + yellow warning banner |
| MongoDB returns empty | Shows "No events yet." info message |
| Health trends data unavailable | Shows empty placeholder chart |
| Missing env var at startup | Logs error and exits with code 1 before rendering anything |

---

## How It Fits Into the Full System

The dashboard is the **read-only consumer** at the end of the AuraGuard AI pipeline. It never writes to MongoDB (except indirectly via the Family Sync sidebar writing to the local filesystem). Here's where it sits:

```
Meta Smart Glasses
        │  (POV video stream via scrcpy/ADB)
        ▼
Vision Engine  :5000
  - Detects faces using Gemini multimodal voting
  - Detects health items (eating, drinking, medicine)
  - POSTs JSON events to the AI Brain
        │  POST /event
        ▼
AI Brain  :8000
  - Validates events against the JSON Contract
  - Verifies health events with Gemini
  - Generates personalized voice scripts
  - Synthesizes speech via ElevenLabs
  - Plays audio through the glasses speaker
  - Writes enriched Event_Records to MongoDB Atlas
        │  Motor async write
        ▼
MongoDB Atlas  ◄──────────────────────────────────────────┐
        │  pymongo read (every 5 seconds)                  │
        ▼                                                  │
Caregiver Dashboard  :8501  ◄── THIS BRANCH               │
  - Live event feed (color-coded)                          │
  - Health trends chart (placeholder)                      │
  - Family Sync sidebar ─────────────────────────────────►┘
                          writes known_faces/ to disk,
                          Vision Engine picks up on restart
```

### The JSON Contract

Every document the dashboard reads from MongoDB follows this schema (the `image_b64` field is intentionally excluded by the Brain before writing):

```json
{
  "event_id": "uuid-v4",
  "timestamp": "2025-04-25T14:32:00Z",
  "patient_id": "patient_001",
  "type": "health",
  "subtype": "drinking",
  "confidence": 0.91,
  "metadata": { "detected_item": "water" },
  "source": "vision_engine_v1",
  "verified": true,
  "voice_script": "Good job, Ismail. I can see you are drinking water. Stay hydrated.",
  "processing_status": "success",
  "processed_at": "2025-04-25T14:32:01Z"
}
```

The dashboard reads `timestamp`, `type`, `subtype`, `confidence`, `verified`, `voice_script`, and `processing_status` for the live feed table.

### Family Sync — Closing the Loop

The sidebar creates the files that the Vision Engine needs to recognize a new person:

```
services/vision/known_faces/
├── hussain.jpg       ← uploaded via sidebar
├── hussain.json      ← written by sidebar
├── ismail.jpg        ← pre-existing
└── ismail.json       ← pre-existing
```

The Vision Engine loads these at startup. Adding a new person via the dashboard and restarting the Vision Engine is all that's needed to extend the system's recognition capability — no code changes required.

---

## Running the Full System

To run all three services together (requires all API keys and hardware):

```bash
python run_all.py
```

See the main [README.md](README.md) for full setup instructions including hardware mirror setup for the Meta Smart Glasses.

---

## What's Not in This Branch

| Feature | Status |
|---|---|
| Health trend charts | Not integrated — chart tab shows a placeholder |
| AI Brain (`services/brain/`) | Not implemented here — separate branch |
| Vision Engine full pipeline | `face_recognition_engine.py` exists; full Flask app wiring is in the Vision branch |
| `run_all.py` launcher | Not in this branch — lives on main |
