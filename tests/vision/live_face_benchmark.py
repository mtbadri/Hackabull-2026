"""
Face Recognition Algorithm Benchmark — Live Webcam Edition
===========================================================
Compares accuracy and speed of four algorithms against a live webcam frame,
using the known_faces/ reference images as the enrollment database.

Algorithms tested:
  1. LBPH (OpenCV)         – classic, zero extra deps
  2. ArcFace (buffalo_l)   – insightface, highest accuracy
  3. SFace (buffalo_s)     – insightface, lightweight/fast
  4. Gemini (shirt color)  – current cloud-based approach

Usage:
    python tests/vision/live_face_benchmark.py
    python tests/vision/live_face_benchmark.py --skip-gemini
    python tests/vision/live_face_benchmark.py --camera 1
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent
KNOWN_FACES_DIR = ROOT / "services" / "vision" / "known_faces"
MODELS_DIR = Path(__file__).parent / "models"

# OpenCV Zoo model paths (downloaded once)
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_CV_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# ── Webcam capture ────────────────────────────────────────────────────────────

def capture_webcam_frame(camera_index: int = 0, warmup_frames: int = 10) -> np.ndarray:
    """
    Open the default webcam, discard a few warm-up frames so the sensor
    auto-exposure settles, then return one BGR frame.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    log.info("Warming up camera (%d frames)…", warmup_frames)
    for _ in range(warmup_frames):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to read frame from webcam")

    log.info("Captured webcam frame: %dx%d", frame.shape[1], frame.shape[0])
    return frame


# ── Base class ────────────────────────────────────────────────────────────────

class FaceRecognitionAlgorithm:
    """
    Base class for all face recognition algorithms.

    Subclasses must implement:
      - load_known_faces(faces_dir)  – enroll reference images
      - match_face(frame_bgr)        – identify the person in a live frame
    """

    def __init__(self, name: str):
        self.name = name
        self.available: bool = True   # set False in __init__ if deps missing
        self.known_names: List[str] = []

    def load_known_faces(self, faces_dir: Path) -> None:
        raise NotImplementedError

    def match_face(
        self, frame_bgr: np.ndarray, threshold: float = 0.6
    ) -> Tuple[Optional[str], float]:
        """Return (matched_name_or_None, confidence 0-1)."""
        raise NotImplementedError


# ── Algorithm 1: Haar Cascade + Face Histogram (zero deps) ───────────────────

class HaarHistogramAlgorithm(FaceRecognitionAlgorithm):
    """
    Haar Cascade detection + HSV colour histogram comparison.
    Zero extra dependencies — pure OpenCV.
    Not a true identity recognizer; included as a speed/accuracy baseline.
    """

    def __init__(self):
        super().__init__("Haar+Histogram (OpenCV)")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.known_histograms: Dict[str, np.ndarray] = {}

    def _face_histogram(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect face, crop it, return normalised HSV histogram."""
        # Resize large images so Haar cascade can find faces reliably
        h, w = img_bgr.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face_roi = img_bgr[y : y + h, x : x + w]
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def load_known_faces(self, faces_dir: Path) -> None:
        if not faces_dir.exists():
            return
        for jpg_file in sorted(faces_dir.glob("*.jpg")):
            name = jpg_file.stem
            img = cv2.imread(str(jpg_file))
            if img is None:
                continue
            hist = self._face_histogram(img)
            if hist is None:
                log.warning("No face detected in %s", jpg_file)
                continue
            self.known_histograms[name] = hist
            self.known_names.append(name)
            log.info("Loaded %s (histogram)", name)

    def match_face(
        self, frame_bgr: np.ndarray, threshold: float = 0.5
    ) -> Tuple[Optional[str], float]:
        query_hist = self._face_histogram(frame_bgr)
        if query_hist is None or len(self.known_histograms) == 0:
            return None, 0.0

        best_match = None
        best_score = 0.0
        for name, known_hist in self.known_histograms.items():
            # Bhattacharyya distance: 0 = identical, 1 = completely different
            dist = cv2.compareHist(
                query_hist.reshape(-1, 1).astype(np.float32),
                known_hist.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            score = 1.0 - dist  # convert to similarity
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score


# ── Algorithm 2: OpenCV YuNet + SFace (built-in deep learning) ───────────────

class OpenCVSFaceAlgorithm(FaceRecognitionAlgorithm):
    """
    YuNet face detector + SFace recognizer — both bundled in opencv-python ≥ 4.8.
    No extra installs required beyond opencv-python.
    """

    def __init__(self):
        super().__init__("YuNet+SFace (OpenCV DNN)")
        if not YUNET_MODEL.exists() or not SFACE_CV_MODEL.exists():
            log.error(
                "Missing model files. Run: python tests/vision/download_models.py"
            )
            self.available = False
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL), "", (320, 320), score_threshold=0.6, nms_threshold=0.3
            )
            self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_CV_MODEL), "")
            self.known_features: Dict[str, np.ndarray] = {}
            log.info("OpenCV YuNet+SFace initialized")
        except Exception as e:
            log.error("OpenCV SFace init failed: %s", e)
            self.available = False

    def _get_face_feature(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and extract SFace feature vector."""
        h, w = img_bgr.shape[:2]
        # YuNet works best when the image is resized to a reasonable size
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
            h, w = img_bgr.shape[:2]

        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        # Use the highest-confidence face
        face = faces[0]
        aligned = self.recognizer.alignCrop(img_bgr, face)
        feature = self.recognizer.feature(aligned)
        return feature

    def load_known_faces(self, faces_dir: Path) -> None:
        if not self.available or not faces_dir.exists():
            return
        for jpg_file in sorted(faces_dir.glob("*.jpg")):
            name = jpg_file.stem
            img = cv2.imread(str(jpg_file))
            if img is None:
                continue
            feature = self._get_face_feature(img)
            if feature is None:
                log.warning("No face detected in %s", jpg_file)
                continue
            self.known_features[name] = feature
            self.known_names.append(name)
            log.info("Loaded %s (SFace feature)", name)

    def match_face(
        self, frame_bgr: np.ndarray, threshold: float = 0.363
    ) -> Tuple[Optional[str], float]:
        """
        Cosine similarity threshold of 0.363 is the recommended value from
        the OpenCV SFace paper for 1e-4 FAR.
        """
        if not self.available:
            return None, 0.0

        query_feature = self._get_face_feature(frame_bgr)
        if query_feature is None or len(self.known_features) == 0:
            return None, 0.0

        best_match = None
        best_score = 0.0
        for name, known_feat in self.known_features.items():
            score = self.recognizer.match(
                query_feature, known_feat, cv2.FACE_RECOGNIZER_SF_FR_COSINE
            )
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= threshold:
            return best_match, float(best_score)
        return None, float(best_score)


# ── Algorithm 2: ArcFace (insightface buffalo_l) ──────────────────────────────

class ArcFaceAlgorithm(FaceRecognitionAlgorithm):
    """ArcFace: Additive Angular Margin Loss (insightface buffalo_l)."""

    def __init__(self):
        super().__init__("ArcFace (buffalo_l)")
        try:
            import insightface

            self.app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
            self.known_embeddings: Dict[str, np.ndarray] = {}
            log.info("ArcFace initialized (buffalo_l)")
        except Exception as e:
            log.error("ArcFace init failed: %s", e)
            self.available = False
            self.app = None

    def load_known_faces(self, faces_dir: Path) -> None:
        """Encode .jpg images into ArcFace embeddings."""
        if not self.available or not faces_dir.exists():
            return

        for jpg_file in sorted(faces_dir.glob("*.jpg")):
            name = jpg_file.stem
            img = cv2.imread(str(jpg_file))
            if img is None:
                log.warning("Failed to load %s", jpg_file)
                continue

            faces = self.app.get(img)
            if len(faces) == 0:
                log.warning("No face detected in %s", jpg_file)
                continue

            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            self.known_embeddings[name] = face.embedding.astype(np.float32)
            self.known_names.append(name)
            log.info("Loaded %s (embedding dim=%d)", name, len(face.embedding))

    def match_face(
        self, frame_bgr: np.ndarray, threshold: float = 0.4
    ) -> Tuple[Optional[str], float]:
        """
        Extract embedding from frame, compare via cosine similarity.
        Typical threshold ~0.4 (higher = stricter).
        """
        if not self.available:
            return None, 0.0

        faces = self.app.get(frame_bgr)
        if len(faces) == 0:
            return None, 0.0

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        embedding = face.embedding.astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-5)

        best_match = None
        best_score = 0.0

        for name, known_emb in self.known_embeddings.items():
            known_emb_norm = known_emb / (np.linalg.norm(known_emb) + 1e-5)
            similarity = np.dot(embedding, known_emb_norm)
            if similarity > best_score:
                best_score = similarity
                best_match = name

        if best_score >= threshold:
            return best_match, float(best_score)
        return None, float(best_score)


# ── Algorithm 3: SFace (insightface buffalo_s) ────────────────────────────────

class SFaceAlgorithm(FaceRecognitionAlgorithm):
    """SFace: Efficient lightweight model (insightface buffalo_s)."""

    def __init__(self):
        super().__init__("SFace (buffalo_s)")
        try:
            import insightface

            self.app = insightface.app.FaceAnalysis(
                name="buffalo_s",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
            self.known_embeddings: Dict[str, np.ndarray] = {}
            log.info("SFace initialized (buffalo_s)")
        except Exception as e:
            log.error("SFace init failed: %s", e)
            self.available = False
            self.app = None

    def load_known_faces(self, faces_dir: Path) -> None:
        """Encode .jpg images into SFace embeddings."""
        if not self.available or not faces_dir.exists():
            return

        for jpg_file in sorted(faces_dir.glob("*.jpg")):
            name = jpg_file.stem
            img = cv2.imread(str(jpg_file))
            if img is None:
                log.warning("Failed to load %s", jpg_file)
                continue

            faces = self.app.get(img)
            if len(faces) == 0:
                log.warning("No face detected in %s", jpg_file)
                continue

            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            self.known_embeddings[name] = face.embedding.astype(np.float32)
            self.known_names.append(name)
            log.info("Loaded %s (embedding dim=%d)", name, len(face.embedding))

    def match_face(
        self, frame_bgr: np.ndarray, threshold: float = 0.4
    ) -> Tuple[Optional[str], float]:
        """Extract embedding from frame, compare via cosine similarity."""
        if not self.available:
            return None, 0.0

        faces = self.app.get(frame_bgr)
        if len(faces) == 0:
            return None, 0.0

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        embedding = face.embedding.astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-5)

        best_match = None
        best_score = 0.0

        for name, known_emb in self.known_embeddings.items():
            known_emb_norm = known_emb / (np.linalg.norm(known_emb) + 1e-5)
            similarity = np.dot(embedding, known_emb_norm)
            if similarity > best_score:
                best_score = similarity
                best_match = name

        if best_score >= threshold:
            return best_match, float(best_score)
        return None, float(best_score)


# ── Algorithm 4: Gemini (shirt color) ────────────────────────────────────────

class GeminiShirtColorAlgorithm(FaceRecognitionAlgorithm):
    """Current implementation: Gemini vision API with shirt-color matching."""

    def __init__(self):
        super().__init__("Gemini (Shirt Color)")
        self.profiles: Dict[str, dict] = {}
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            log.info("Gemini initialized")
        except Exception as e:
            log.error("Gemini init failed: %s", e)
            self.available = False
            self.model = None

    def load_known_faces(self, faces_dir: Path) -> None:
        """Load profiles with shirt colors from .json files."""
        if not self.available or not faces_dir.exists():
            return

        for profile_file in sorted(faces_dir.glob("*.json")):
            try:
                profile = json.loads(profile_file.read_text())
                name = profile.get("name", profile_file.stem)
                self.known_names.append(name)
                self.profiles[name] = profile
                log.info(
                    "Loaded %s (shirt: %s)", name, profile.get("shirt_color", "?")
                )
            except Exception as e:
                log.warning("Failed to load %s: %s", profile_file.name, e)

    def match_face(
        self, frame_bgr: np.ndarray, threshold: float = 0.65
    ) -> Tuple[Optional[str], float]:
        """Match using Gemini shirt-color detection."""
        if not self.available or len(self.profiles) == 0:
            return None, 0.0

        try:
            # Resize for faster upload
            h, w = frame_bgr.shape[:2]
            if w > 640:
                frame_bgr = cv2.resize(frame_bgr, (640, int(h * 640 / w)))

            _, buf = cv2.imencode(".jpg", frame_bgr)
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            known_colors = ", ".join(
                f"{p.get('shirt_color')} ({name})"
                for name, p in self.profiles.items()
            )

            parts = [
                {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
                f"""You are identifying a person by their shirt color.

The known people and their shirt colors are:
{known_colors}

Look at the person visible in this image. What color is their shirt or top?

Respond ONLY with this exact JSON (no extra text):
{{"shirt_color": "green", "confidence": 0.95, "person_visible": true}}

- shirt_color: the single best-matching color from the known list above
- confidence: float 0.0–1.0 of how certain you are
- person_visible: true if a person with a visible shirt is in frame, false otherwise""",
            ]

            response = self.model.generate_content(parts)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text.strip())

            if not result.get("person_visible", False):
                return None, 0.0

            detected_color = result.get("shirt_color", "").lower().strip()
            confidence = float(result.get("confidence", 0.0))

            # Match against known colors
            for name, profile in self.profiles.items():
                shirt_color = profile.get("shirt_color", "").lower()
                for word in shirt_color.split("/"):
                    if word.strip() in detected_color or detected_color in word.strip():
                        if confidence >= threshold:
                            return name, confidence

            return None, confidence
        except Exception as e:
            log.warning("Gemini matching failed: %s", e)
            return None, 0.0


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    algorithms: List[FaceRecognitionAlgorithm],
    test_frame: np.ndarray,
    ground_truth: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run all algorithms against a single test frame.
    Returns dict of {algo_name: {match, confidence, time_ms, correct}}.
    """
    results = {}

    for algo in algorithms:
        if not algo.available:
            log.warning("Skipping %s (not available)", algo.name)
            continue

        log.info("\n" + "=" * 60)
        log.info("Testing: %s", algo.name)
        log.info("=" * 60)

        algo.load_known_faces(KNOWN_FACES_DIR)
        if len(algo.known_names) == 0:
            log.warning("No known faces loaded for %s", algo.name)
            continue

        log.info("Loaded %d known faces: %s", len(algo.known_names), algo.known_names)

        start = time.time()
        matched_name, confidence = algo.match_face(test_frame)
        elapsed_ms = (time.time() - start) * 1000

        correct = matched_name == ground_truth if ground_truth else None

        status = "✓" if correct else ("✗" if correct is False else "?")
        log.info(
            "%s Matched: %s | Confidence: %.3f | Time: %.2f ms",
            status,
            matched_name or "None",
            confidence,
            elapsed_ms,
        )

        results[algo.name] = {
            "match": matched_name,
            "confidence": confidence,
            "time_ms": elapsed_ms,
            "correct": correct,
        }

    return results


def print_summary(results: Dict[str, Dict]) -> None:
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Algorithm':<25} {'Match':<15} {'Confidence':<12} {'Time (ms)':<12} {'Correct':<8}"
    )
    print("-" * 80)

    # Sort by time (ascending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["time_ms"])

    for name, stats in sorted_results:
        correct_str = (
            "✓" if stats["correct"] else ("✗" if stats["correct"] is False else "?")
        )
        print(
            f"{name:<25} {stats['match'] or 'None':<15} {stats['confidence']:>10.3f} "
            f"{stats['time_ms']:>10.2f} {correct_str:>7}"
        )

    print("=" * 80)


def save_results(results: Dict[str, Dict], output_file: Path) -> None:
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "ranking_by_speed": [
            name for name, _ in sorted(results.items(), key=lambda x: x[1]["time_ms"])
        ],
    }

    output_file.write_text(json.dumps(output, indent=2))
    log.info("Results saved to %s", output_file)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face recognition benchmark")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--skip-gemini", action="store_true", help="Skip Gemini (slow/API cost)"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Expected person name in frame (for accuracy check)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a test image instead of live webcam",
    )
    args = parser.parse_args()

    log.info("Starting face recognition benchmark")
    log.info("Known faces directory: %s", KNOWN_FACES_DIR)

    # Capture live frame or load from file
    if args.image:
        test_frame = cv2.imread(args.image)
        if test_frame is None:
            log.error("Failed to load image: %s", args.image)
            sys.exit(1)
        log.info("Loaded test image: %s (%dx%d)", args.image, test_frame.shape[1], test_frame.shape[0])
    else:
        try:
            test_frame = capture_webcam_frame(camera_index=args.camera)
        except Exception as e:
            log.error("Failed to capture webcam frame: %s", e)
            log.error(
                "If running from a terminal without camera permission, try:\n"
                "  python tests/vision/live_face_benchmark.py --image path/to/photo.jpg\n"
                "Or grant camera access to Terminal in System Settings > Privacy & Security > Camera"
            )
            sys.exit(1)

    # Initialize algorithms
    algorithms = [
        HaarHistogramAlgorithm(),
        OpenCVSFaceAlgorithm(),
        ArcFaceAlgorithm(),
        SFaceAlgorithm(),
    ]

    if not args.skip_gemini:
        algorithms.append(GeminiShirtColorAlgorithm())

    # Run benchmark
    results = run_benchmark(algorithms, test_frame, ground_truth=args.ground_truth)

    # Print summary
    print_summary(results)

    # Save results
    output_file = ROOT / "tests" / "vision" / "benchmark_results.json"
    save_results(results, output_file)

    log.info("Benchmark complete!")


if __name__ == "__main__":
    main()
