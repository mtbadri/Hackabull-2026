---
inclusion: manual
---

# Dev Workflow

## Running Services (run each in a separate terminal)

```bash
# Vision Engine
python -m services.vision.face_recognition_engine

# AI Brain
uvicorn services.brain.main:app --host 0.0.0.0 --port 8000 --reload

# Caregiver Dashboard
streamlit run services/dashboard/app.py --server.port 8501
```

## Installing Dependencies
```bash
pip install -r requirements.txt
```

## Environment Setup
```bash
cp .env.example .env
# Fill in all values in .env before running any service
```

## Adding a New Known Face
1. Add `{name}.jpg` to `services/vision/known_faces/`
2. Add `{name}.json` with keys: `name`, `relationship`, `background`, `last_conversation`
3. Restart the Vision Engine — faces are loaded at startup

## Testing MongoDB Connection
```bash
python test_mongo.py
```
