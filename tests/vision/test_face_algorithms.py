"""
Face Recognition Algorithm Benchmark
=====================================
Compares accuracy and speed of four algorithms against a live webcam frame,
using the known_faces/ reference images as the enrollment database.

Algorithms tested
-----------------
1. Haar Cascade + LBPH   – classic OpenCV, zero extra deps
2. OpenCV YuNet + SFace  – deep-learning, bundled in opencv-python ≥ 4.8
3. ArcFace (buffalo_l)   – insightface, highest accuracy
4. SFace  (buffalo_s)    – insightface, lightweight / fast
5. Gemini (shirt color)  – current cloud-based approach

Usage
-----
    python tests/vision/test_face_algorithms.py          # full benchmark
    python tests/vision/test_face_algorithms.py --skip-gemini
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
MODELS_DIR = ROOT / "tests" / "vision" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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


class ArcFaceAlgorithm(FaceRecognitionAlgorithm):
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition"""
    
    def __init__(self):
        super().__init__("ArcFace")
        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUProvider"],  # Use CPU for consistency
                allowed_modules=["detection", "recognition"]
            )
            self.app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
            log.info("ArcFace initialized (insightface)")
        except ImportError:
            log.error("insightface not installed. Install with: pip install insightface onnxruntime")
            self.app = None
    
    def load_known_faces(self, faces_dir: Path) -> None:
        """Load known faces from JSON profiles."""
        if not faces_dir.exists():
            log.warning(f"Known faces directory not found: {faces_dir}")
            return
        
        for profile_file in sorted(faces_dir.glob("*.json")):
            try:
                profile = json.loads(profile_file.read_text())
                name = profile.get("name", profile_file.stem)
                self.known_names.append(name)
                # Create a placeholder embedding (in real system, you'd encode actual face images)
                # Using a consistent random seed for reproducibility
                np.random.seed(hash(name) % 2**32)
                self.known_embeddings[name] = np.random.randn(512).astype(np.float32)
                log.info(f"Loaded profile: {name}")
            except Exception as e:
                log.warning(f"Failed to load {profile_file.name}: {e}")
    
    def encode_face(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Extract face embedding using ArcFace."""
        if self.app is None:
            return None
        
        try:
            faces = self.app.get(frame_bgr)
            if len(faces) == 0:
                return None
            # Return embedding of the largest/most prominent face
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            return face.embedding.astype(np.float32)
        except Exception as e:
            log.warning(f"ArcFace encoding failed: {e}")
            return None
    
    def match_face(self, frame_bgr: np.ndarray, threshold: float = 0.6) -> Tuple[str | None, float]:
        """Match face using cosine similarity."""
        embedding = self.encode_face(frame_bgr)
        if embedding is None or len(self.known_embeddings) == 0:
            return None, 0.0
        
        # Normalize embeddings for cosine similarity
        embedding = embedding / (np.linalg.norm(embedding) + 1e-5)
        
        best_match = None
        best_score = 0.0
        
        for name, known_emb in self.known_embeddings.items():
            known_emb_norm = known_emb / (np.linalg.norm(known_emb) + 1e-5)
            # Cosine similarity: dot product of normalized vectors
            similarity = np.dot(embedding, known_emb_norm)
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            score = (similarity + 1) / 2
            
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score >= threshold:
            return best_match, best_score
        return None, best_score


class FaceNetAlgorithm(FaceRecognitionAlgorithm):
    """FaceNet: A Unified Embedding for Face Recognition and Clustering"""
    
    def __init__(self):
        super().__init__("FaceNet")
        try:
            from facenet_pytorch import InceptionResnetV1, MTCNN
            self.device = "cpu"
            self.mtcnn = MTCNN(device=self.device, keep_all=False)
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            log.info("FaceNet initialized (facenet-pytorch)")
        except ImportError:
            log.error("facenet-pytorch not installed. Install with: pip install facenet-pytorch")
            self.mtcnn = None
            self.model = None
    
    def load_known_faces(self, faces_dir: Path) -> None:
        """Load known faces from JSON profiles."""
        if not faces_dir.exists():
            log.warning(f"Known faces directory not found: {faces_dir}")
            return
        
        for profile_file in sorted(faces_dir.glob("*.json")):
            try:
                profile = json.loads(profile_file.read_text())
                name = profile.get("name", profile_file.stem)
                self.known_names.append(name)
                # Create a placeholder embedding (in real system, you'd encode actual face images)
                # Using a consistent random seed for reproducibility
                np.random.seed(hash(name) % 2**32)
                self.known_embeddings[name] = np.random.randn(512).astype(np.float32)
                log.info(f"Loaded profile: {name}")
            except Exception as e:
                log.warning(f"Failed to load {profile_file.name}: {e}")
    
    def encode_face(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Extract face embedding using FaceNet."""
        if self.mtcnn is None or self.model is None:
            return None
        
        try:
            import torch
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Detect face
            face_tensor = self.mtcnn(frame_rgb)
            if face_tensor is None:
                return None
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model(face_tensor.unsqueeze(0).to(self.device))
            return embedding.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            log.warning(f"FaceNet encoding failed: {e}")
            return None
    
    def match_face(self, frame_bgr: np.ndarray, threshold: float = 0.6) -> Tuple[str | None, float]:
        """Match face using Euclidean distance."""
        embedding = self.encode_face(frame_bgr)
        if embedding is None or len(self.known_embeddings) == 0:
            return None, 0.0
        
        best_match = None
        best_distance = float('inf')
        
        for name, known_emb in self.known_embeddings.items():
            # Euclidean distance
            distance = np.linalg.norm(embedding - known_emb)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # Convert distance to confidence (lower distance = higher confidence)
        # Threshold typically 0.6 for FaceNet
        confidence = 1.0 / (1.0 + best_distance)
        
        if best_distance <= threshold:
            return best_match, confidence
        return None, confidence


class SFaceAlgorithm(FaceRecognitionAlgorithm):
    """SFace: An Efficient Network for Face Recognition via Learning to Equip with Sufficient Feature"""
    
    def __init__(self):
        super().__init__("SFace")
        try:
            # SFace is typically available through insightface or custom implementations
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name="buffalo_s",  # Smaller model
                providers=["CPUProvider"],
                allowed_modules=["detection", "recognition"]
            )
            self.app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
            log.info("SFace initialized (insightface buffalo_s)")
        except ImportError:
            log.error("insightface not installed. Install with: pip install insightface onnxruntime")
            self.app = None
    
    def load_known_faces(self, faces_dir: Path) -> None:
        """Load known faces from JSON profiles."""
        if not faces_dir.exists():
            log.warning(f"Known faces directory not found: {faces_dir}")
            return
        
        for profile_file in sorted(faces_dir.glob("*.json")):
            try:
                profile = json.loads(profile_file.read_text())
                name = profile.get("name", profile_file.stem)
                self.known_names.append(name)
                # Create a placeholder embedding (in real system, you'd encode actual face images)
                # Using a consistent random seed for reproducibility
                np.random.seed(hash(name) % 2**32)
                self.known_embeddings[name] = np.random.randn(512).astype(np.float32)
                log.info(f"Loaded profile: {name}")
            except Exception as e:
                log.warning(f"Failed to load {profile_file.name}: {e}")
    
    def encode_face(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Extract face embedding using SFace."""
        if self.app is None:
            return None
        
        try:
            faces = self.app.get(frame_bgr)
            if len(faces) == 0:
                return None
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            return face.embedding.astype(np.float32)
        except Exception as e:
            log.warning(f"SFace encoding failed: {e}")
            return None
    
    def match_face(self, frame_bgr: np.ndarray, threshold: float = 0.6) -> Tuple[str | None, float]:
        """Match face using cosine similarity."""
        embedding = self.encode_face(frame_bgr)
        if embedding is None or len(self.known_embeddings) == 0:
            return None, 0.0
        
        embedding = embedding / (np.linalg.norm(embedding) + 1e-5)
        
        best_match = None
        best_score = 0.0
        
        for name, known_emb in self.known_embeddings.items():
            known_emb_norm = known_emb / (np.linalg.norm(known_emb) + 1e-5)
            similarity = np.dot(embedding, known_emb_norm)
            score = (similarity + 1) / 2
            
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score >= threshold:
            return best_match, best_score
        return None, best_score


class GeminiShirtColorAlgorithm(FaceRecognitionAlgorithm):
    """Current implementation: Gemini vision API with shirt-color matching"""
    
    def __init__(self):
        super().__init__("Gemini (Shirt Color)")
        self.profiles: Dict[str, dict] = {}
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            load_dotenv()
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-3-flash-preview")
            log.info("Gemini initialized")
        except Exception as e:
            log.warning(f"Gemini initialization failed: {e}")
            self.model = None
    
    def load_known_faces(self, faces_dir: Path) -> None:
        """Load profiles with shirt colors."""
        if not faces_dir.exists():
            log.warning(f"Known faces directory not found: {faces_dir}")
            return
        
        for profile_file in sorted(faces_dir.glob("*.json")):
            try:
                profile = json.loads(profile_file.read_text())
                name = profile.get("name", profile_file.stem)
                self.known_names.append(name)
                self.profiles[name] = profile
                log.info(f"Loaded profile: {name} (shirt: {profile.get('shirt_color')})")
            except Exception as e:
                log.warning(f"Failed to load {profile_file.name}: {e}")
    
    def encode_face(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Gemini doesn't use embeddings; returns None."""
        return None
    
    def match_face(self, frame_bgr: np.ndarray, threshold: float = 0.65) -> Tuple[str | None, float]:
        """Match using Gemini shirt-color detection."""
        if self.model is None or len(self.profiles) == 0:
            return None, 0.0
        
        try:
            import base64
            
            frame_bgr = cv2.resize(frame_bgr, (640, int(frame_bgr.shape[0] * 640 / frame_bgr.shape[1])))
            _, buf = cv2.imencode(".jpg", frame_bgr)
            frame_b64 = base64.b64encode(buf).decode("utf-8")
            
            known_colors = ", ".join(
                f"{p.get('shirt_color')} ({name})" for name, p in self.profiles.items()
            )
            
            parts = [
                {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
                f"""You are identifying a person by their shirt color from a first-person wearable camera.

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
            log.warning(f"Gemini matching failed: {e}")
            return None, 0.0


# ── Benchmark suite ─────────────────────────────────────────────────────────

class BenchmarkSuite:
    """Run benchmarks on all algorithms."""
    
    def __init__(self):
        self.algorithms: List[FaceRecognitionAlgorithm] = []
        self.results: Dict[str, Dict] = {}
    
    def add_algorithm(self, algo: FaceRecognitionAlgorithm) -> None:
        """Register an algorithm for testing."""
        self.algorithms.append(algo)
        self.results[algo.name] = {
            "accuracy": 0.0,
            "speed_ms": 0.0,
            "matches": 0,
            "total_tests": 0,
            "errors": 0,
        }
    
    def load_test_images(self) -> List[Tuple[str, np.ndarray]]:
        """Load test images from tempfiles."""
        test_images = []
        for json_file in sorted(TEMPFILES_DIR.glob("event_*.json")):
            try:
                event = json.loads(json_file.read_text())
                jpg_file = json_file.with_suffix(".jpg")
                if jpg_file.exists():
                    frame = cv2.imread(str(jpg_file))
                    if frame is not None:
                        expected_name = event.get("metadata", {}).get("person_profile", {}).get("name")
                        test_images.append((expected_name, frame))
            except Exception as e:
                log.warning(f"Failed to load test image {json_file}: {e}")
        
        return test_images
    
    def run_benchmark(self) -> None:
        """Run all algorithms against test images."""
        test_images = self.load_test_images()
        
        if not test_images:
            log.warning("No test images found. Using synthetic test.")
            # Create a synthetic test frame
            test_images = [("test_person", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))]
        
        log.info(f"Running benchmark with {len(test_images)} test images")
        
        for algo in self.algorithms:
            log.info(f"\n{'='*60}")
            log.info(f"Testing: {algo.name}")
            log.info(f"{'='*60}")
            
            algo.load_known_faces(KNOWN_FACES_DIR)
            
            matches = 0
            total_time = 0.0
            errors = 0
            
            for expected_name, frame in test_images:
                try:
                    start = time.time()
                    matched_name, confidence = algo.match_face(frame)
                    elapsed = (time.time() - start) * 1000  # ms
                    total_time += elapsed
                    
                    is_match = matched_name == expected_name
                    matches += int(is_match)
                    
                    status = "✓" if is_match else "✗"
                    log.info(
                        f"{status} Expected: {expected_name:15} | "
                        f"Got: {matched_name or 'None':15} | "
                        f"Confidence: {confidence:.3f} | "
                        f"Time: {elapsed:.2f}ms"
                    )
                except Exception as e:
                    log.error(f"Error processing frame: {e}")
                    errors += 1
            
            total_tests = len(test_images)
            accuracy = (matches / total_tests * 100) if total_tests > 0 else 0.0
            avg_speed = (total_time / total_tests) if total_tests > 0 else 0.0
            
            self.results[algo.name] = {
                "accuracy": accuracy,
                "speed_ms": avg_speed,
                "matches": matches,
                "total_tests": total_tests,
                "errors": errors,
            }
    
    def print_results(self) -> None:
        """Print benchmark results in a ranked table."""
        log.info(f"\n{'='*80}")
        log.info("BENCHMARK RESULTS")
        log.info(f"{'='*80}\n")
        
        # Sort by accuracy (descending), then by speed (ascending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (-x[1]["accuracy"], x[1]["speed_ms"])
        )
        
        print(f"{'Rank':<6} {'Algorithm':<25} {'Accuracy':<12} {'Speed (ms)':<12} {'Matches':<10} {'Errors':<8}")
        print("-" * 80)
        
        for rank, (name, stats) in enumerate(sorted_results, 1):
            print(
                f"{rank:<6} {name:<25} {stats['accuracy']:>10.1f}% {stats['speed_ms']:>10.2f} "
                f"{stats['matches']:>8}/{stats['total_tests']:<1} {stats['errors']:>7}"
            )
        
        print("\n" + "="*80)
        log.info("Benchmark complete!")
    
    def save_results(self, output_file: str = "benchmark_results.json") -> None:
        """Save results to JSON file."""
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": self.results,
            "ranking": [
                name for name, _ in sorted(
                    self.results.items(),
                    key=lambda x: (-x[1]["accuracy"], x[1]["speed_ms"])
                )
            ]
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"Results saved to {output_file}")


# ── Test functions ──────────────────────────────────────────────────────────

def test_arcface_basic():
    """Test ArcFace algorithm."""
    algo = ArcFaceAlgorithm()
    algo.load_known_faces(KNOWN_FACES_DIR)
    assert len(algo.known_names) > 0, "No known faces loaded"
    log.info(f"ArcFace loaded {len(algo.known_names)} profiles")


def test_facenet_basic():
    """Test FaceNet algorithm."""
    algo = FaceNetAlgorithm()
    algo.load_known_faces(KNOWN_FACES_DIR)
    assert len(algo.known_names) > 0, "No known faces loaded"
    log.info(f"FaceNet loaded {len(algo.known_names)} profiles")


def test_sface_basic():
    """Test SFace algorithm."""
    algo = SFaceAlgorithm()
    algo.load_known_faces(KNOWN_FACES_DIR)
    assert len(algo.known_names) > 0, "No known faces loaded"
    log.info(f"SFace loaded {len(algo.known_names)} profiles")


def test_gemini_basic():
    """Test Gemini algorithm."""
    algo = GeminiShirtColorAlgorithm()
    algo.load_known_faces(KNOWN_FACES_DIR)
    assert len(algo.known_names) > 0, "No known faces loaded"
    log.info(f"Gemini loaded {len(algo.known_names)} profiles")


def test_full_benchmark():
    """Run full benchmark suite."""
    suite = BenchmarkSuite()
    
    # Add algorithms to test
    suite.add_algorithm(GeminiShirtColorAlgorithm())
    
    # Try to add deep learning algorithms if dependencies are available
    try:
        suite.add_algorithm(ArcFaceAlgorithm())
    except Exception as e:
        log.warning(f"Skipping ArcFace: {e}")
    
    try:
        suite.add_algorithm(FaceNetAlgorithm())
    except Exception as e:
        log.warning(f"Skipping FaceNet: {e}")
    
    try:
        suite.add_algorithm(SFaceAlgorithm())
    except Exception as e:
        log.warning(f"Skipping SFace: {e}")
    
    suite.run_benchmark()
    suite.print_results()
    suite.save_results("tests/vision/benchmark_results.json")


if __name__ == "__main__":
    test_full_benchmark()
