"""
Face Recognition Engine
-----------------------
How it works:
1. At startup, it loads reference images from the known_faces/ folder.
   Each person needs:
     - a photo:   known_faces/hussain.jpg
     - a profile: known_faces/hussain.json

2. For each video frame, it:
   - Detects faces using face_recognition library
   - Compares each detected face against all known encodings
   - If a match is found, logs an identity event to MongoDB

3. MongoDB stores every recognition event so caregivers can
   see a history of who the patient has seen.
"""

import os
import json
import uuid
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import face_recognition
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── MongoDB setup ────────────────────────────────────────────────────────────

def connect_to_mongo():
    """Connect to MongoDB Atlas and return the events collection."""
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "aura_guard")
    collection_name = os.getenv("MONGODB_COLLECTION", "events")

    if not uri:
        raise ValueError("MONGODB_URI is not set in your .env file")

    client = MongoClient(uri, tlsCAFile=certifi.where())
    collection = client[db_name][collection_name]
    log.info(f"Connected to MongoDB: {db_name}.{collection_name}")
    return collection


# ── Known faces loader ───────────────────────────────────────────────────────

def load_known_faces(known_faces_dir: str) -> tuple[list, list]:
    """
    Load face encodings and profiles from the known_faces/ directory.

    For each person you want recognized, add:
      known_faces/name.jpg   <- clear photo of their face
      known_faces/name.json  <- their profile info

    Returns:
        encodings: list of 128-dimension face vectors
        profiles:  list of matching profile dicts
    """
    encodings = []
    profiles = []

    faces_path = Path(known_faces_dir)
    if not faces_path.exists():
        log.warning(f"Known faces directory not found: {known_faces_dir}")
        return encodings, profiles

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png"}
    for img_file in faces_path.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue

        # Look for matching .json profile
        profile_file = img_file.with_suffix(".json")
        if not profile_file.exists():
            log.warning(f"No profile JSON found for {img_file.name}, skipping.")
            continue

        # Load the image and compute face encoding
        image = face_recognition.load_image_file(str(img_file))
        face_encs = face_recognition.face_encodings(image)

        if not face_encs:
            log.warning(f"No face detected in {img_file.name}, skipping.")
            continue

        # Load the profile
        with open(profile_file) as f:
            profile = json.load(f)

        encodings.append(face_encs[0])  # Use first face found in image
        profiles.append(profile)
        log.info(f"Loaded face: {profile.get('name', img_file.stem)}")

    log.info(f"Total known faces loaded: {len(encodings)}")
    return encodings, profiles


# ── MongoDB event logger ─────────────────────────────────────────────────────

def log_recognition_event(collection, profile: dict, confidence: float, frame):
    """
    Save a face recognition event to MongoDB.

    The document structure follows the JSON Contract from the spec.
    We store everything EXCEPT the raw image (to keep documents small).
    """
    # Encode frame as base64 (stored temporarily, not in MongoDB per spec)
    _, buffer = cv2.imencode(".jpg", frame)
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    event = {
        "event_id": str(uuid.uuid4()),          # Unique ID for this event
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": os.getenv("PATIENT_ID", "unknown"),
        "type": "identity",
        "subtype": "face_recognized",
        "confidence": round(confidence, 3),
        "metadata": {
            "person_profile": profile           # Full profile stored here
        },
        "source": "vision_engine_v1",
        # Enrichment fields (normally added by Brain, added here for standalone use)
        "verified": True,
        "voice_script": build_voice_script(profile),
        "processing_status": "success",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    collection.insert_one(event)
    log.info(f"Logged to MongoDB: {profile.get('name')} (confidence: {confidence:.2f})")
    return event["event_id"]


def build_voice_script(profile: dict) -> str:
    """Build a personalized voice script from the person's profile."""
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


# ── Main recognition loop ────────────────────────────────────────────────────

def run(known_faces_dir="services/vision/known_faces", video_source=0):
    """
    Main loop:
    - Opens webcam (or video file)
    - Every frame: detects faces, compares to known faces
    - On match: logs event to MongoDB, draws box on screen
    """
    collection = connect_to_mongo()
    known_encodings, known_profiles = load_known_faces(known_faces_dir)

    if not known_encodings:
        log.warning("No known faces loaded. Recognition will not trigger any events.")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        log.error(f"Could not open video source: {video_source}")
        return

    log.info("Starting face recognition loop. Press Q to quit.")

    # Track recently seen faces to avoid spamming MongoDB
    recently_logged = {}   # name -> last logged timestamp
    COOLDOWN_SECONDS = 30  # Only log same person once every 30 seconds

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to read frame, retrying...")
            continue

        frame_count += 1

        # Only process every 3rd frame for performance
        if frame_count % 3 != 0:
            cv2.imshow("AuraGuard - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Resize frame for faster processing (face_recognition is slow on full res)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and compute encodings
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            if not known_encodings:
                continue

            # Compare against all known faces
            # Returns list of True/False for each known face
            matches = face_recognition.compare_faces(known_encodings, face_encoding)

            # Get distance scores (lower = more similar, 0.0 = perfect match)
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            best_match_idx = distances.argmin()
            best_distance = distances[best_match_idx]
            confidence = 1.0 - best_distance  # Convert distance to confidence score

            if matches[best_match_idx] and confidence > 0.5:
                profile = known_profiles[best_match_idx]
                name = profile.get("name", "Unknown")

                # Draw green box around recognized face
                # Scale coordinates back up (we resized by 0.25 earlier)
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.0%})",
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # Log to MongoDB (with cooldown to avoid duplicates)
                now = datetime.now(timezone.utc).timestamp()
                last_logged = recently_logged.get(name, 0)

                if now - last_logged > COOLDOWN_SECONDS:
                    log_recognition_event(collection, profile, confidence, frame)
                    recently_logged[name] = now
            else:
                # Unknown face — draw red box
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown",
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

        cv2.imshow("AuraGuard - Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Face recognition loop stopped.")


if __name__ == "__main__":
    run()
