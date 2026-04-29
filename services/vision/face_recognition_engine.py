"""
Face Recognition Engine (YuNet + SFace with ONNXRuntime)
---------------------------------------------------------
1. Loads known-face embeddings from local known_faces/*.jpg + *.json files.
2. Uses OpenCV's YuNet for face detection (via DNN).
3. Uses ONNXRuntime for SFace recognition (workaround for OpenCV ONNX bug).
4. No external API calls - all processing is local.
5. Logs events to local JSONL and speaks via edge-tts.
"""

import os
import json
import uuid
import base64
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import queue
import subprocess
import threading
import time
import av
import cv2
import numpy as np
import onnxruntime as ort
import requests as http_requests
from dotenv import load_dotenv
import pygame

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
KNOWN_FACES_DIR = _ROOT / "known_faces"
_MODELS_DIR = _ROOT.parent.parent / "tests" / "vision" / "models"
YUNET_MODEL = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = _MODELS_DIR / "face_recognition_sface_2021dec.onnx"
_EVENT_LOG_DIR = Path(__file__).parent.parent.parent / "tempfiles"
EVENTS_JSONL = _EVENT_LOG_DIR / "events.jsonl"
DETECTION_PAUSED_FLAG = _EVENT_LOG_DIR / "detection_paused"

# ── Config ────────────────────────────────────────────────────────────────────

RECOGNITION_INTERVAL = 2.0  # seconds between recognition attempts
COOLDOWN_SECONDS = 10  # seconds before the same person can be re-announced
CONFIDENCE_THRESHOLD = 0.363  # SFace cosine threshold
ENABLE_HEALTH_DETECTION: bool = os.getenv("ENABLE_HEALTH_DETECTION", "false").lower() == "true"
HEALTH_CHECK_INTERVAL_SECONDS = 5
HEALTH_COOLDOWN_SECONDS = 120

HEALTH_SUBTYPE_MAP = {
    "WATER BOTTLE": "drinking", "SODA CAN": "drinking", "CAN": "drinking",
    "CUP": "drinking", "GLASS": "drinking", "MUG": "drinking",
    "BOTTLE": "drinking", "DRINKING": "drinking",
    "FOOD": "eating", "FORK": "eating", "SPOON": "eating",
    "SANDWICH": "eating", "EATING": "eating",
    "PILL": "medicine_taken", "PILLS": "medicine_taken", "TABLET": "medicine_taken",
    "MEDICINE": "medicine_taken", "MEDICATION": "medicine_taken",
}

_last_health_event: dict[str, float] = {}

# ── Thread state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()
_speak_lock = threading.Lock()
_recognizing = False
_pending_profile = False
_health_running = False

# ── YuNet + SFace initialization ─────────────────────────────────────────────

def _load_face_models():
    """Load YuNet detector and SFace recognizer using ONNXRuntime."""
    if not YUNET_MODEL.exists():
        log.error("YuNet model not found!")
        raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}")
    if not SFACE_MODEL.exists():
        log.error("SFace model not found!")
        raise FileNotFoundError(f"SFace model not found: {SFACE_MODEL}")

    # Load YuNet detector using OpenCV DNN
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3,
    )

    # Load SFace using ONNXRuntime (workaround for OpenCV bug)
    sess = ort.InferenceSession(
        str(SFACE_MODEL),
        providers=['CPUExecutionProvider']
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    log.info("YuNet + SFace models loaded (SFace via ONNXRuntime)")
    return detector, sess, input_name, output_name


_detector, _sface_session, _sface_input_name, _sface_output_name = _load_face_models()
_detector_lock = threading.Lock()


def _get_face_feature_onnx(img_bgr: np.ndarray) -> np.ndarray | None:
    """Detect face with YuNet and get SFace feature using ONNXRuntime."""
    h, w = img_bgr.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    # Detect faces with YuNet
    with _detector_lock:
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(img_bgr)

    if faces is None or len(faces) == 0:
        return None

    # Get the highest-confidence face
    best = max(faces, key=lambda f: float(f[14]))
    
    # Align the face using OpenCV (only if recognizer was successfully created)
    aligned = _recognizer.alignCrop(img_bgr, best) if _recognizer is not None else None
    
    # If OpenCV alignment fails, manually crop
    if aligned is None:
        x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        # Add some margin
        margin = int(w * 0.1)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
        aligned = img_bgr[y1:y2, x1:x2]
        aligned = cv2.resize(aligned, (112, 112))

    # Preprocess for SFace: normalize to [-1, 1]
    aligned = aligned.astype(np.float32) / 255.0
    aligned = (aligned - 0.5) / 0.5
    aligned = aligned.transpose(2, 0, 1).flatten()
    aligned = aligned.reshape(1, 3, 112, 112)

    # Run SFace inference
    features = _sface_session.run([_sface_output_name], {_sface_input_name: aligned})[0]
    return features[0]


# ── Also keep OpenCV recognizer for alignment ────────────────────────────────

try:
    _recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    _use_onnx_alignment = False
except Exception as e:
    log.warning(f"OpenCV SFace failed: {e}, using manual alignment")
    _recognizer = None
    _use_onnx_alignment = True


def _get_face_feature(img_bgr: np.ndarray) -> np.ndarray | None:
    """Detect the largest face and return SFace feature vector."""
    h, w = img_bgr.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    with _detector_lock:
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(img_bgr)

    if faces is None or len(faces) == 0:
        return None

    # Pick the highest-confidence face
    best = max(faces, key=lambda f: float(f[14]))

    # Align the face
    if _recognizer is not None:
        try:
            aligned = _recognizer.alignCrop(img_bgr, best)
            feature = _recognizer.feature(aligned)
            return feature
        except Exception as e:
            log.warning(f"OpenCV alignment failed: {e}")

    # Fallback: manual alignment
    x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    margin = int(w * 0.1)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
    aligned = img_bgr[y1:y2, x1:x2]
    aligned = cv2.resize(aligned, (112, 112))

    # Preprocess for SFace
    aligned = aligned.astype(np.float32) / 255.0
    aligned = (aligned - 0.5) / 0.5
    aligned = aligned.transpose(2, 0, 1).flatten()
    aligned = aligned.reshape(1, 3, 112, 112)

    # Run SFace via ONNXRuntime
    features = _sface_session.run([_sface_output_name], {_sface_input_name: aligned})[0]
    return features[0]


# ── Known-face loader ─────────────────────────────────────────────────────────

def load_known_faces() -> list[dict]:
    """Load known faces from local known_faces/*.jpg + *.json files."""
    known: list[dict] = []

    if not KNOWN_FACES_DIR.exists():
        log.warning("Known faces directory not found: %s", KNOWN_FACES_DIR)
        return known

    for jpg_file in sorted(KNOWN_FACES_DIR.glob("*.jpg")):
        name = jpg_file.stem
        json_file = jpg_file.with_suffix(".json")
        profile: dict = {}
        if json_file.exists():
            try:
                profile = json.loads(json_file.read_text())
            except Exception as e:
                log.warning("Failed to parse %s: %s", json_file.name, e)

        img = cv2.imread(str(jpg_file))
        if img is None:
            log.warning("Could not read %s", jpg_file)
            continue

        feature = _get_face_feature(img)
        if feature is None:
            log.warning("No face detected in %s", jpg_file)
            continue

        known.append({"name": name, "profile": profile, "feature": feature})
        log.info("Loaded face: %s", name)

    log.info("Loaded %d known faces", len(known))
    return known


# ── SFace matching ────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Flatten to 1D if needed
    a = a.flatten()
    b = b.flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


def match_with_sface(frame_bgr: np.ndarray, known_faces: list[dict]) -> dict | None:
    """Run YuNet + SFace on frame_bgr and match against known faces."""
    if not known_faces:
        return None

    query_feature = _get_face_feature(frame_bgr)
    if query_feature is None:
        log.debug("No face detected in frame")
        return None

    best_score = 0.0
    best_person = None

    for person in known_faces:
        score = _cosine_similarity(query_feature, person["feature"])
        log.debug("  %s -> cosine=%.3f", person["name"], score)
        if score > best_score:
            best_score = score
            best_person = person

    if best_score >= CONFIDENCE_THRESHOLD and best_person is not None:
        log.info("Matched: %s (cosine=%.3f)", best_person["name"], best_score)
        return best_person["profile"]

    log.debug("No confident match (best=%.3f)", best_score)
    return None


# ── Local event store ─────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def _append_event(event: dict) -> None:
    """Append one event as a JSON line to the local events.jsonl store."""
    _EVENT_LOG_DIR.mkdir(exist_ok=True)
    payload = {k: v for k, v in event.items() if k != "_id"}
    with _write_lock:
        with EVENTS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def read_events(n: int = 50) -> list[dict]:
    """Read the most recent n events from the local JSONL store."""
    if not EVENTS_JSONL.exists():
        return []
    lines = EVENTS_JSONL.read_text(encoding="utf-8").splitlines()
    events = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(events) >= n:
            break
    return events


# ── Frame quality check ───────────────────────────────────────────────────────

def is_frame_usable(frame_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 30:
        log.debug("Frame too dark")
        return False
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 2:
        log.debug("Frame too blurry")
        return False
    return True


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


# ── Event saver ───────────────────────────────────────────────────────────────

_event_log_counter = 0


def save_event_json(event: dict) -> None:
    global _event_log_counter
    _EVENT_LOG_DIR.mkdir(exist_ok=True)
    _event_log_counter += 1
    payload = {k: v for k, v in event.items() if k not in ("image_b64", "_id")}
    path = _EVENT_LOG_DIR / f"event_{_event_log_counter:04d}.json"
    path.write_text(json.dumps(payload, indent=2))
    log.debug("Event JSON saved -> %s", path)
    try:
        http_requests.post("http://localhost:8502/ingest", json=payload, timeout=2)
    except Exception:
        pass


# ── Event logger ──────────────────────────────────────────────────────────────

def log_event(profile: dict, frame_bgr: np.ndarray | None = None) -> str:
    """Append an identity event to the local JSONL store."""
    voice_script = build_voice_script(profile)

    image_b64 = ""
    if frame_bgr is not None:
        _, buf = cv2.imencode(".jpg", frame_bgr)
        image_b64 = base64.b64encode(buf).decode("utf-8")

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": os.getenv("PATIENT_ID", "unknown"),
        "type": "identity",
        "subtype": "face_recognized",
        "confidence": 1.0,
        "image_b64": image_b64,
        "metadata": {"person_profile": profile},
        "source": "vision_engine_yunet_sface",
        "verified": True,
        "voice_script": voice_script,
        "processing_status": "success",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_event(event)
    save_event_json(event)
    log.info("Event logged for: %s", profile.get("name"))
    return voice_script


# ── Audio ─────────────────────────────────────────────────────────────────────

def _play_mp3(tmp_path: str):
    import platform
    played = False
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        played = True
    except Exception:
        pass
    if not played and platform.system() == "Darwin":
        subprocess.run(["afplay", tmp_path], check=False)


def speak(voice_script: str):
    if not _speak_lock.acquire(blocking=False):
        log.info("[Voice skipped - already speaking]")
        return
    tmp_path = None
    try:
        import asyncio, edge_tts

        async def _synth():
            chunks: list[bytes] = []
            async for chunk in edge_tts.Communicate(voice_script, voice="en-US-AriaNeural").stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        audio_bytes = asyncio.run(_synth())
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        _play_mp3(tmp_path)
    except Exception as e:
        log.warning("edge-tts failed: %s", e)
        log.info("[Voice] %s", voice_script)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        _speak_lock.release()


# ── Health activity detector (Gemini - optional) ──────────────────────────────

def detect_health_activity(frame_bgr: np.ndarray) -> None:
    """Ask Gemini if the wearer is eating, drinking, or taking medicine."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        log.warning("Gemini unavailable for health detection: %s", e)
        return

    h, w = frame_bgr.shape[:2]
    if w > 640:
        frame_bgr = cv2.resize(frame_bgr, (640, int(h * 640 / w)))
    _, buf = cv2.imencode(".jpg", frame_bgr)
    frame_b64 = base64.b64encode(buf).decode("utf-8")

    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
        (
            "This image is from a first-person point-of-view camera worn on glasses. "
            "Look carefully at what objects are visible and being held or used. "
            "If you see any of the following, name it: "
            "water bottle, soda can, cup, glass, mug, bottle, food, fork, spoon, sandwich, pill, pills, tablet, medicine, medication. "
            "Reply with ONLY the object name from that list (e.g. 'water bottle', 'cup', 'pills'). "
            "If none of those objects are visible, reply with exactly: NONE."
        ),
    ]

    try:
        response = _model.generate_content(parts)
        answer = response.text.strip().upper()
        log.info("Health check response: %r", answer)
        subtype = next(
            (st for kw, st in HEALTH_SUBTYPE_MAP.items() if kw in answer), None
        )
        if subtype is None:
            return

        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts - _last_health_event.get(subtype, 0) < HEALTH_COOLDOWN_SECONDS:
            log.debug("Health event %s suppressed (cooldown)", subtype)
            return
        _last_health_event[subtype] = now_ts

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": os.getenv("PATIENT_ID", "unknown"),
            "type": "health",
            "subtype": subtype,
            "confidence": 0.9,
            "image_b64": frame_b64,
            "metadata": {"detected_item": answer.title()},
            "source": "vision_engine_yunet_sface",
        }
        brain_url = f"http://{os.getenv('BRAIN_HOST', 'localhost')}:{os.getenv('BRAIN_PORT', '8000')}/event"
        resp = http_requests.post(brain_url, json=event, timeout=5)
        save_event_json(event)
        log.info("Health event sent: %s -> Brain %d", subtype, resp.status_code)
    except Exception as e:
        log.warning("Health detection failed: %s", e)


# ── Frame generator ───────────────────────────────────────────────────────────

_NET_PREFIXES = ("rtmp://", "rtsp://", "http://", "https://")


# Global video capture for cleanup
_video_capture = None
_video_capture_lock = threading.Lock()


def _cleanup_on_exit():
    """Cleanup function to release webcam on exit."""
    global _video_capture
    if _video_capture is not None:
        with _video_capture_lock:
            if _video_capture is not None:
                _video_capture.release()
                _video_capture = None
        cv2.destroyAllWindows()
        print("Webcam released")


import atexit
atexit.register(_cleanup_on_exit)


def _yield_frames(video_source):
    """Yield BGR numpy frames with automatic reconnect on failure."""
    global _video_capture
    
    use_av = isinstance(video_source, str) and any(
        video_source.startswith(p) for p in _NET_PREFIXES
    )

    if use_av:
        while True:
            try:
                log.info("Connecting to stream: %s", video_source)
                container = av.open(
                    video_source,
                    options={"fflags": "nobuffer", "flags": "low_delay"},
                )
                log.info("Stream connected.")
                for av_frame in container.decode(video=0):
                    yield av_frame.to_ndarray(format="bgr24")
                log.info("Stream ended - reconnecting in 5 s...")
            except Exception as e:
                log.warning("Stream error: %s - reconnecting in 5 s...", e)
            time.sleep(5)
    else:
        global _video_capture
        with _video_capture_lock:
            _video_capture = cv2.VideoCapture(video_source)
        if not _video_capture.isOpened():
            log.error("Could not open video source: %s", video_source)
            return
        while True:
            with _video_capture_lock:
                ret, frame = _video_capture.read()
            if ret:
                yield frame
            else:
                log.warning("Frame read failed - reconnecting in 3 s...")
                with _video_capture_lock:
                    _video_capture.release()
                    _video_capture = None
                time.sleep(3)
                with _video_capture_lock:
                    _video_capture = cv2.VideoCapture(video_source)
                if not _video_capture.isOpened():
                    log.error("Failed to reopen: %s", video_source)
                    return


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(video_source=None):
    try:
        known_faces = load_known_faces()

        if not known_faces:
            log.warning("No known faces loaded - add <name>.jpg + <name>.json to known_faces/")

        if video_source is None:
            video_source = os.getenv("RTMP_STREAM_URL", "rtmp://localhost/live/stream")

        log.info("Opening video source: %s", video_source)
        log.info("Recognition engine: YuNet + SFace (ONNXRuntime)")
        log.info("Running. Press Q to quit.")

        frame_count = 0
        last_label = ("Scanning...", (200, 200, 200))
        current_match = None
        last_matched_name = None
        last_match_time = 0.0
        last_recognition_time = 0.0
        last_health_time = 0.0

        global _recognizing, _pending_profile, _health_running

        def _recognition_worker(frame_copy):
            global _recognizing, _pending_profile
            profile = match_with_sface(frame_copy, known_faces)
            with _lock:
                _pending_profile = profile
                _recognizing = False

        def _health_worker(frame_copy):
            global _health_running
            detect_health_activity(frame_copy)
            with _lock:
                _health_running = False

        # Keep only the latest frame
        _frame_q: queue.Queue = queue.Queue(maxsize=1)

        def _reader():
            for f in _yield_frames(video_source):
                if _frame_q.full():
                    try:
                        _frame_q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    _frame_q.put_nowait(f)
                except queue.Full:
                    pass

        threading.Thread(target=_reader, daemon=True).start()

        while True:
            try:
                frame = _frame_q.get(timeout=5)
            except queue.Empty:
                log.warning("No frames for 5 s - waiting...")
                continue

            frame_count += 1
            now = time.time()

            # Reset after cooldown
            if current_match is not None and (now - last_match_time) >= COOLDOWN_SECONDS:
                log.info("Cooldown expired - ready to scan again")
                current_match = None
                last_matched_name = None
                last_label = ("Scanning...", (200, 200, 200))

            # Collect result from background recognition worker
            with _lock:
                has_result = _pending_profile is not False
                profile = _pending_profile
                _pending_profile = False

            if has_result and profile is not None and current_match is None:
                name = profile.get("name", "Unknown")
                current_match = name
                last_match_time = now
                if name != last_matched_name:
                    last_matched_name = name
                    frame_snapshot = frame.copy()
                    threading.Thread(
                        target=lambda p=profile, f=frame_snapshot: speak(log_event(p, f)),
                        daemon=True,
                    ).start()
                    last_label = (f"Matched: {name}", (0, 255, 0))
                else:
                    last_label = (f"Matched: {name}", (0, 200, 100))
            elif has_result and profile is None and current_match is None:
                last_label = ("Scanning...", (200, 200, 200))

            # Fire recognition on a timer
            detection_paused = DETECTION_PAUSED_FLAG.exists()
            if detection_paused:
                last_label = ("Manual mode - detection paused", (200, 140, 0))
            else:
                with _lock:
                    should_recognise = (
                        current_match is None
                        and not _recognizing
                        and (now - last_match_time) >= COOLDOWN_SECONDS
                        and (now - last_recognition_time) >= RECOGNITION_INTERVAL
                        and known_faces
                        and is_frame_usable(frame)
                    )
                    if should_recognise:
                        _recognizing = True
                        last_recognition_time = now

                if should_recognise:
                    threading.Thread(
                        target=_recognition_worker,
                        args=(frame.copy(),),
                        daemon=True,
                    ).start()

            # Optional health scan
            if ENABLE_HEALTH_DETECTION:
                with _lock:
                    should_health = (
                        not _health_running
                        and (now - last_health_time) >= HEALTH_CHECK_INTERVAL_SECONDS
                    )
                    if should_health:
                        _health_running = True
                        last_health_time = now
                if should_health:
                    threading.Thread(target=_health_worker, args=(frame.copy(),), daemon=True).start()

            # Display
            cv2.putText(
                frame, last_label[0], (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_label[1], 2,
            )
            cv2.imshow("AuraGuard - Face Recognition (YuNet+SFace)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        log.info("Stopped.")
    finally:
        _cleanup_on_exit()


if __name__ == "__main__":
    run()