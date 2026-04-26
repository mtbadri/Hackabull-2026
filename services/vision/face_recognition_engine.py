"""
Face Recognition Engine (Gemini-powered)
-----------------------------------------
How it works:
1. Loads reference photos + profiles from known_faces/ at startup
2. Captures frames from webcam continuously
3. Every N frames, sends the current frame + all reference photos to Gemini
4. Asks Gemini: "Does this person match anyone in the reference photos?"
5. If matched, logs an identity event to MongoDB with the person's profile
"""

import os
import json
import uuid
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import certifi
import google.generativeai as genai
from pymongo import MongoClient
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import pygame
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Gemini setup ─────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ── MongoDB setup ─────────────────────────────────────────────────────────────

def connect_to_mongo():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "aura_guard")
    collection_name = os.getenv("MONGODB_COLLECTION", "events")
    if not uri:
        raise ValueError("MONGODB_URI is not set in your .env file")
    client = MongoClient(uri, tlsCAFile=certifi.where())
    log.info(f"Connected to MongoDB: {db_name}.{collection_name}")
    return client[db_name][collection_name]

# ── Known faces loader ────────────────────────────────────────────────────────

def load_known_faces(known_faces_dir: str) -> list[dict]:
    """
    Load reference photos and profiles from known_faces/.
    Returns a list of dicts: { "name": str, "image_b64": str, "profile": dict }
    """
    known = []
    faces_path = Path(known_faces_dir)

    if not faces_path.exists():
        log.warning(f"Known faces directory not found: {known_faces_dir}")
        return known

    image_extensions = {".jpg", ".jpeg", ".png"}
    for img_file in faces_path.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue

        profile_file = img_file.with_suffix(".json")
        if not profile_file.exists():
            log.warning(f"No profile JSON for {img_file.name}, skipping.")
            continue

        with open(img_file, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        with open(profile_file) as f:
            profile = json.load(f)

        known.append({
            "name": profile.get("name", img_file.stem),
            "image_b64": image_b64,
            "profile": profile
        })
        log.info(f"Loaded reference: {profile.get('name', img_file.stem)}")

    log.info(f"Total known faces loaded: {len(known)}")
    return known

# ── Gemini face matching ──────────────────────────────────────────────────────

def match_face_with_gemini(frame, known_faces: list[dict]) -> tuple[dict | None, float]:
    """
    Send the current frame + all reference photos to Gemini.
    Ask if the person in the frame matches any known person.

    Returns: (matched_profile, confidence) or (None, 0.0)
    """
    if not known_faces:
        return None, 0.0

    # Encode current frame as base64
    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    # Build the list of known names for the prompt
    names = [p["name"] for p in known_faces]

    # Build prompt
    prompt = f"""You are a strict face verification system.
I will show you a live camera frame followed by {len(known_faces)} reference photo(s).
The reference photos are of: {', '.join(names)}.

Does the person in the FIRST image (live frame) match any of the reference photos?
Only say matched=true if you are highly confident (90%+) it is the same person.
If the face is unclear, partially visible, or you are unsure, say matched=false.

Reply in this exact JSON format only, no extra text:
{{"matched": true, "name": "PersonName", "confidence": 0.95}}
or
{{"matched": false, "name": null, "confidence": 0.0}}"""

    # Build image parts: first the live frame, then all reference photos
    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
    ]
    for person in known_faces:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": person["image_b64"]}})

    parts.append(prompt)

    try:
        response = model.generate_content(parts)
        text = response.text.strip()

        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        result = json.loads(text)

        if result.get("matched"):
            matched_name = result.get("name")
            confidence = float(result.get("confidence", 0.8))
            # Only accept high-confidence matches
            if confidence < 0.90:
                log.info(f"Match rejected — confidence too low: {confidence:.0%}")
                return None, 0.0
            for person in known_faces:
                if person["name"].lower() == matched_name.lower():
                    return person["profile"], confidence
        return None, 0.0

    except Exception as e:
        log.warning(f"Gemini matching failed: {e}")
        return None, 0.0

# ── MongoDB event logger ──────────────────────────────────────────────────────

def build_voice_script(profile: dict) -> str:
    patient_name = os.getenv("PATIENT_NAME", "there")
    name = profile.get("name", "someone")
    relationship = profile.get("relationship", "someone you know")
    background = profile.get("background", "")
    last_convo = profile.get("last_conversation", "")

    script = f"{patient_name}, your {relationship} {name} is here."
    if background:
        script += f" {background}."
    if last_convo:
        script += f" Last time you spoke, {last_convo}."
    return script


def log_recognition_event(collection, profile: dict, confidence: float):
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": os.getenv("PATIENT_ID", "unknown"),
        "type": "identity",
        "subtype": "face_recognized",
        "confidence": round(confidence, 3),
        "metadata": {"person_profile": profile},
        "source": "vision_engine_v1",
        "verified": True,
        "voice_script": build_voice_script(profile),
        "processing_status": "success",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    collection.insert_one(event)
    log.info(f"Logged to MongoDB: {profile.get('name')} (confidence: {confidence:.0%})")
    return event["event_id"]

# ── New person enrollment ─────────────────────────────────────────────────────

def record_audio(duration=8, sample_rate=16000) -> str:
    """Record audio from mic for `duration` seconds, save to temp file, return path."""
    log.info(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                        channels=1, dtype="float32")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_data, sample_rate)
    log.info("Recording complete.")
    return tmp.name


def transcribe_and_extract_profile(audio_path: str) -> dict | None:
    """
    Send recorded audio to Gemini and ask it to extract a person profile.
    Returns a profile dict or None if extraction fails.
    """
    try:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = """This is a short audio recording of a person introducing themselves to an Alzheimer's patient.
Extract the following information from what they said and return ONLY valid JSON:
{
  "name": "their first name",
  "relationship": "how they know the patient (e.g. son, daughter, doctor, friend)",
  "background": "one sentence about who they are",
  "last_conversation": "what they last talked about with the patient, or empty string if not mentioned"
}
If you cannot determine a field, use an empty string."""

        response = model.generate_content([
            {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
            prompt
        ])

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        profile = json.loads(text)
        log.info(f"Extracted profile: {profile}")
        return profile

    except Exception as e:
        log.warning(f"Profile extraction failed: {e}")
        return None


def enroll_new_person(frame, known_faces_dir: str, known_faces: list) -> bool:
    """
    Full enrollment flow for an unknown face:
    1. Speak a greeting prompt
    2. Record their introduction
    3. Extract profile via Gemini
    4. Save their photo + profile to known_faces/
    5. Add them to the in-memory known_faces list
    Returns True if enrollment succeeded.
    """
    log.info("Unknown face detected — starting enrollment.")

    # Step 1: Ask them to introduce themselves
    speak("Hi there! I don't recognize you yet. Please tell me your name and how you know the patient. You have 8 seconds.")

    # Step 2: Record their response
    audio_path = record_audio(duration=8)

    # Step 3: Extract profile from audio
    profile = transcribe_and_extract_profile(audio_path)
    os.remove(audio_path)

    if not profile or not profile.get("name"):
        speak("Sorry, I couldn't understand that. Please try again later.")
        return False

    name = profile["name"].strip()
    safe_name = name.lower().replace(" ", "_")

    # Step 4: Save their photo
    faces_path = Path(known_faces_dir)
    img_path = faces_path / f"{safe_name}.jpg"
    cv2.imwrite(str(img_path), frame)

    # Step 5: Save their profile JSON
    profile_path = faces_path / f"{safe_name}.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    # Step 6: Add to in-memory list
    with open(img_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    known_faces.append({
        "name": name,
        "image_b64": image_b64,
        "profile": profile
    })

    log.info(f"Enrolled new person: {name}")
    speak(f"Got it! I'll remember you, {name}. Nice to meet you.")
    return True


# ── Audio playback via ElevenLabs ────────────────────────────────────────────

def speak(voice_script: str):
    """
    Synthesize voice_script using ElevenLabs and play it out loud via pygame.
    If ElevenLabs fails, falls back to a simple print.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    if not api_key or not voice_id or api_key == "your_elevenlabs_api_key_here":
        log.warning("ElevenLabs not configured — skipping audio.")
        log.info(f"Voice script: {voice_script}")
        return

    try:
        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=voice_script,
            model_id="eleven_monolingual_v1",
        )

        # Save to temp file and play with pygame
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in audio:
                f.write(chunk)
            tmp_path = f.name

        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        os.remove(tmp_path)
        log.info("Audio played successfully.")

    except Exception as e:
        log.warning(f"Audio playback failed: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(known_faces_dir="services/vision/known_faces", video_source=0):
    collection = connect_to_mongo()
    known_faces = load_known_faces(known_faces_dir)

    if not known_faces:
        log.warning("No known faces loaded. Add photos + JSON profiles to known_faces/")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        log.error(f"Could not open video source: {video_source}")
        return

    log.info("Starting face recognition. Press Q to quit.")

    recently_logged = {}   # name -> last logged timestamp
    COOLDOWN_SECONDS = 30
    GEMINI_EVERY_N_FRAMES = 15  # Call Gemini every 15 frames to save API calls
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to read frame, retrying...")
            continue

        frame_count += 1

        # Only call Gemini every N frames
        if frame_count % GEMINI_EVERY_N_FRAMES == 0 and known_faces:
            profile, confidence = match_face_with_gemini(frame, known_faces)

            if profile:
                name = profile.get("name", "Unknown")
                now = datetime.now(timezone.utc).timestamp()
                last_logged = recently_logged.get(name, 0)

                cv2.putText(frame, f"{name} ({confidence:.0%})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                if now - last_logged > COOLDOWN_SECONDS:
                    log_recognition_event(collection, profile, confidence)
                    recently_logged[name] = now
                    speak(build_voice_script(profile))
            else:
                cv2.putText(frame, "Unknown - enrolling...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

                # Only attempt enrollment once every 60 seconds to avoid spamming
                now = datetime.now(timezone.utc).timestamp()
                last_enrolled = recently_logged.get("__unknown__", 0)
                if now - last_enrolled > 60:
                    recently_logged["__unknown__"] = now
                    enroll_new_person(frame, known_faces_dir, known_faces)
        cv2.imshow("AuraGuard - Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Stopped.")


if __name__ == "__main__":
    run()
