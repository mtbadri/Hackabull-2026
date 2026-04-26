"""
Face Recognition Engine (Gemini-powered)
------------------------------------------
1. Loads known faces + profiles from known_faces/ at startup
2. Every N frames, takes a snapshot and sends it to Gemini
3. Gemini compares the snapshot against ALL reference photos
4. Only fires if Gemini is certain — strict prompt, no guessing
5. Logs event to MongoDB and speaks via ElevenLabs
"""

import os
import json
import uuid
import base64
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import certifi
import google.generativeai as genai
from pymongo import MongoClient
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import pygame

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_PARALLEL_CALLS = 5          # parallel votes per match attempt
CHECK_EVERY_N_FRAMES = 10          # frames between Gemini checks
COOLDOWN_SECONDS = 60
CONFIDENCE_THRESHOLD = 0.65        # minimum confidence to accept a match
KNOWN_FACES_DIR = Path(__file__).parent / "known_faces"

# ── Gemini ────────────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview")

# ── MongoDB ───────────────────────────────────────────────────────────────────

def connect_to_mongo():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "aura_guard")
    col_name = os.getenv("MONGODB_COLLECTION", "events")
    if not uri:
        raise ValueError("MONGODB_URI not set in .env")
    client = MongoClient(uri, tlsCAFile=certifi.where())
    log.info(f"Connected to MongoDB: {db_name}.{col_name}")
    return client[db_name][col_name]

# ── Known faces loader ────────────────────────────────────────────────────────

def load_known_faces() -> list[dict]:
    """
    Load reference images and profiles.
    Reads the raw image bytes — no face detection needed at load time.
    Returns list of { name, image_b64, mime_type, profile }
    """
    known = []

    if not KNOWN_FACES_DIR.exists():
        log.warning(f"Known faces directory not found: {KNOWN_FACES_DIR}")
        return known

    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

    for img_file in sorted(KNOWN_FACES_DIR.iterdir()):
        if img_file.suffix.lower() not in mime_map:
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
            "mime_type": mime_map[img_file.suffix.lower()],
            "profile": profile,
        })
        log.info(f"Loaded reference: {profile.get('name', img_file.stem)}")

    log.info(f"Total known faces loaded: {len(known)}")
    return known

# ── Gemini matching ───────────────────────────────────────────────────────────

def is_frame_usable(frame_bgr) -> bool:
    """Reject frames that are too dark or too blurry."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    log.info(f"Frame quality — brightness={brightness:.1f}, blur={blur_score:.1f}")
    if brightness < 30:
        log.info("Frame too dark, skipping")
        return False
    if blur_score < 2:
        log.info("Frame too blurry, skipping")
        return False
    return True


def _parse_gemini_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from Gemini response."""
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _verify_one(frame_b64: str, person: dict) -> float:
    """
    Ask Gemini: is the live frame the same person as this ONE reference photo?
    Returns a confidence float 0.0–1.0.
    """
    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
        {"inline_data": {"mime_type": person["mime_type"], "data": person["image_b64"]}},
        f"""You are a strict face verification system.

Image 1 is a live camera frame.
Image 2 is a reference photo of {person['name']}.

Task: Determine whether the person in Image 1 is the SAME individual as in Image 2.

Focus only on permanent facial features: eye spacing, nose shape, jawline, face geometry, brow ridge.
Ignore lighting, angle, expression, glasses, or hair differences.

Be very conservative — only say yes if you are highly confident.

Respond ONLY with this exact JSON (no extra text):
{{"same_person": true, "confidence": 0.95}}
or
{{"same_person": false, "confidence": 0.10}}

confidence must be a float between 0.0 and 1.0 representing how certain you are.""",
    ]

    response = model.generate_content(parts)
    result = _parse_gemini_json(response.text.strip())
    confidence = float(result.get("confidence", 0.0))
    same = bool(result.get("same_person", False))
    log.info(f"  [{person['name']}] same={same}, confidence={confidence:.2f}")
    return confidence if same else 0.0


def match_with_gemini(frame_bgr, known_faces: list[dict]) -> dict | None:
    """
    For each known person, fire GEMINI_PARALLEL_CALLS verification calls in
    parallel and tally the votes. The person with the most confident majority
    wins — only accepted if confidence >= CONFIDENCE_THRESHOLD.
    """
    if not known_faces:
        return None

    if not is_frame_usable(frame_bgr):
        log.debug("Frame rejected (quality check failed)")
        return None

    _, buf = cv2.imencode(".jpg", frame_bgr)
    frame_b64 = base64.b64encode(buf).decode("utf-8")

    # Build one task per (person, vote_index)
    tasks = [(person, i) for person in known_faces for i in range(GEMINI_PARALLEL_CALLS)]

    # votes[name] = list of confidence scores where same_person=True
    votes: dict[str, list[float]] = {p["name"]: [] for p in known_faces}

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(_verify_one, frame_b64, person): person for person, _ in tasks}
        for future in as_completed(futures):
            person = futures[future]
            try:
                confidence = future.result()
                if confidence > 0:
                    votes[person["name"]].append(confidence)
            except Exception as e:
                log.warning(f"Vote failed for {person['name']}: {e}")

    # Pick the person with the most positive votes, break ties by avg confidence
    best_person = None
    best_score = (0, 0.0)  # (vote_count, avg_confidence)

    for person in known_faces:
        name = person["name"]
        positive_votes = votes[name]
        count = len(positive_votes)
        avg_conf = sum(positive_votes) / count if count else 0.0
        log.info(f"  [{name}] votes={count}/{GEMINI_PARALLEL_CALLS}, avg_confidence={avg_conf:.2f}")
        if (count, avg_conf) > best_score:
            best_score = (count, avg_conf)
            best_person = person

    majority = GEMINI_PARALLEL_CALLS // 2 + 1  # need more than half
    if best_person and best_score[0] >= majority and best_score[1] >= CONFIDENCE_THRESHOLD:
        log.info(f"Matched: {best_person['name']} ({best_score[0]}/{GEMINI_PARALLEL_CALLS} votes, avg={best_score[1]:.2f})")
        return best_person["profile"]

    log.info(f"No confident majority match (best: {best_person['name'] if best_person else 'none'}, votes={best_score[0]}, avg={best_score[1]:.2f})")
    return None

# ── Voice script ──────────────────────────────────────────────────────────────

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

# ── MongoDB logger ────────────────────────────────────────────────────────────

def log_event(collection, profile: dict) -> str:
    voice_script = build_voice_script(profile)
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": os.getenv("PATIENT_ID", "unknown"),
        "type": "identity",
        "subtype": "face_recognized",
        "confidence": 1.0,
        "metadata": {"person_profile": profile},
        "source": "vision_engine_v1",
        "verified": True,
        "voice_script": voice_script,
        "processing_status": "success",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    collection.insert_one(event)
    log.info(f"Logged: {profile.get('name')}")
    return voice_script

# ── Audio ─────────────────────────────────────────────────────────────────────

def speak(voice_script: str):
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

    if not api_key or api_key == "your_elevenlabs_api_key_here":
        log.info(f"[Voice] {voice_script}")
        return

    try:
        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=voice_script,
            model_id="eleven_monolingual_v1",
        )
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
    except Exception as e:
        log.warning(f"Audio failed: {e}")

# ── Face presence detector ────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def has_face(frame_bgr) -> bool:
    """Quick local check — is there a face in frame at all?"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    return len(faces) > 0

# ── Main loop ─────────────────────────────────────────────────────────────────

def run(video_source=0):
    collection = connect_to_mongo()
    known_faces = load_known_faces()

    if not known_faces:
        log.warning("No known faces loaded — add <name>.jpg + <name>.json to known_faces/")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        log.error(f"Could not open video source: {video_source}")
        return

    log.info("Running. Press Q to quit.")

    frame_count = 0
    last_label = ("Scanning...", (200, 200, 200))
    current_match = None      # name of whoever is currently in frame
    face_absent_frames = 0    # consecutive frames with no face detected
    ABSENT_THRESHOLD = 10     # frames without a face before we reset

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed, retrying...")
            continue

        frame_count += 1

        # ── Detect face presence every frame (cheap, local) ──
        face_present = has_face(frame)

        if not face_present:
            face_absent_frames += 1
            if face_absent_frames >= ABSENT_THRESHOLD and current_match is not None:
                log.info(f"Face left frame — resetting (was: {current_match})")
                current_match = None
                last_label = ("Scanning...", (200, 200, 200))
        else:
            face_absent_frames = 0

        # ── Only call Gemini if a face is present and not already matched ──
        if (face_present
                and current_match is None
                and frame_count % CHECK_EVERY_N_FRAMES == 0
                and known_faces):

            profile = match_with_gemini(frame, known_faces)

            if profile:
                name = profile.get("name", "Unknown")
                current_match = name
                voice_script = log_event(collection, profile)
                speak(voice_script)
                last_label = (f"Matched: {name}", (0, 255, 0))
            else:
                last_label = ("No match", (0, 0, 255))

        cv2.putText(frame, last_label[0], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_label[1], 2)
        cv2.imshow("AuraGuard - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Stopped.")


if __name__ == "__main__":
    run()
