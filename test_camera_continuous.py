"""
Continuous Camera Test with Real-Time Face Recognition
======================================================
Displays live camera feed with face recognition overlay.
Press 'q' to quit, 's' to save a snapshot.

Usage:
    python test_camera_continuous.py
    python test_camera_continuous.py --camera 1
    python test_camera_continuous.py --algorithm sface
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
KNOWN_FACES_DIR = ROOT / "services" / "vision" / "known_faces"

# ── Simple SFace Algorithm ───────────────────────────────────────────────────

class SimpleSFaceAlgorithm:
    """Lightweight SFace algorithm using insightface buffalo_s."""

    def __init__(self):
        self.name = "SFace (buffalo_s)"
        self.available = True
        try:
            import insightface

            self.app = insightface.app.FaceAnalysis(
                name="buffalo_s",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
            self.known_embeddings: Dict[str, np.ndarray] = {}
            self.known_names = []
            log.info("SFace initialized")
        except Exception as e:
            log.error("SFace init failed: %s", e)
            self.available = False
            self.app = None

    def load_known_faces(self, faces_dir: Path) -> None:
        """Load known faces from JPG files."""
        if not self.available or not faces_dir.exists():
            log.warning("Cannot load faces: algorithm not available or directory missing")
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
            log.info("Loaded %s", name)

        log.info("Total known faces: %d", len(self.known_names))

    def match_face(
        self, frame_bgr: np.ndarray, threshold: float = 0.4
    ) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        Match face in frame.
        Returns: (matched_name, confidence, bounding_box)
        """
        if not self.available:
            return None, 0.0, None

        faces = self.app.get(frame_bgr)
        if len(faces) == 0:
            return None, 0.0, None

        # Get largest face
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

        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        
        if best_score >= threshold:
            return best_match, float(best_score), bbox
        return None, float(best_score), bbox


# ── Continuous Camera Test ───────────────────────────────────────────────────

class ContinuousCameraTest:
    """Real-time camera test with face recognition overlay."""

    def __init__(self, camera_index: int = 0, algorithm=None):
        self.camera_index = camera_index
        self.algorithm = algorithm
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.snapshot_count = 0

    def start(self) -> bool:
        """Initialize camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            log.error("Cannot open camera index %d", self.camera_index)
            return False

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        log.info("Camera opened successfully")
        log.info("Resolution: %dx%d", 
                 int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        self.running = True
        return True

    def stop(self):
        """Release camera."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        log.info("Camera released")

    def draw_overlay(self, frame: np.ndarray, name: Optional[str], 
                     confidence: float, bbox: Optional[np.ndarray],
                     processing_time: float) -> np.ndarray:
        """Draw recognition results on frame."""
        h, w = frame.shape[:2]
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw processing time
        cv2.putText(frame, f"Time: {processing_time:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding box and name
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # Choose color based on match
            if name:
                color = (0, 255, 0)  # Green for match
                label = f"{name} ({confidence:.2f})"
            else:
                color = (0, 165, 255)  # Orange for unknown
                label = f"Unknown ({confidence:.2f})"
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save snapshot",
            "Press 'r' to reload known faces"
        ]
        y_offset = h - 70
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return frame

    def save_snapshot(self, frame: np.ndarray, name: Optional[str], confidence: float):
        """Save current frame as snapshot."""
        self.snapshot_count += 1
        filename = f"snapshot_{self.snapshot_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save metadata
        metadata = {
            "snapshot_id": self.snapshot_count,
            "timestamp": time.time(),
            "matched_name": name,
            "confidence": confidence,
            "frame_count": self.frame_count
        }
        with open(f"snapshot_{self.snapshot_count:04d}.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        log.info("Saved snapshot: %s (match: %s, confidence: %.3f)", 
                 filename, name or "None", confidence)

    def run(self):
        """Main loop."""
        if not self.start():
            return

        log.info("Starting continuous camera test...")
        log.info("Window: 'Face Recognition Test'")
        
        # Load known faces
        if self.algorithm:
            self.algorithm.load_known_faces(KNOWN_FACES_DIR)

        last_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    log.warning("Failed to read frame")
                    break

                self.frame_count += 1
                
                # Run face recognition
                start_time = time.time()
                name, confidence, bbox = None, 0.0, None
                
                if self.algorithm and self.algorithm.available:
                    name, confidence, bbox = self.algorithm.match_face(frame)
                
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update FPS
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    elapsed = current_time - last_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.last_fps_update = current_time
                
                # Draw overlay
                display_frame = self.draw_overlay(frame, name, confidence, bbox, processing_time)
                
                # Show frame
                cv2.imshow("Face Recognition Test", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log.info("Quit requested")
                    break
                elif key == ord('s'):
                    self.save_snapshot(frame, name, confidence)
                elif key == ord('r'):
                    log.info("Reloading known faces...")
                    if self.algorithm:
                        self.algorithm.known_embeddings.clear()
                        self.algorithm.known_names.clear()
                        self.algorithm.load_known_faces(KNOWN_FACES_DIR)
        
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        
        finally:
            self.stop()
            log.info("Total frames processed: %d", self.frame_count)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continuous camera test with face recognition")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sface",
        choices=["sface", "none"],
        help="Face recognition algorithm (default: sface)"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Continuous Camera Test")
    log.info("=" * 60)
    log.info("Camera index: %d", args.camera)
    log.info("Algorithm: %s", args.algorithm)
    log.info("Known faces directory: %s", KNOWN_FACES_DIR)
    
    # Initialize algorithm
    algorithm = None
    if args.algorithm == "sface":
        algorithm = SimpleSFaceAlgorithm()
        if not algorithm.available:
            log.error("SFace not available. Install with: pip install insightface onnxruntime")
            return
    
    # Run test
    test = ContinuousCameraTest(camera_index=args.camera, algorithm=algorithm)
    test.run()
    
    log.info("Test complete!")


if __name__ == "__main__":
    main()
