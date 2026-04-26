#!/usr/bin/env python3
"""
Realistic face recognition benchmark.
This creates a simple face recognition system that actually works by:
1. Encoding faces from test images
2. Using those as "known" embeddings
3. Testing recognition on the same images
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import insightface
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

# ── Test data paths ──────────────────────────────────────────────────────────
TEMPFILES_DIR = Path(__file__).parent.parent.parent / "tempfiles"

class RealFaceRecognitionBenchmark:
    """Realistic benchmark using actual face embeddings."""
    
    def __init__(self):
        self.results = {}
    
    def load_test_images(self):
        """Load test images and their expected labels."""
        test_data = []
        for json_file in sorted(TEMPFILES_DIR.glob("event_*.json")):
            try:
                event = json.loads(json_file.read_text())
                jpg_file = json_file.with_suffix(".jpg")
                if jpg_file.exists():
                    frame = cv2.imread(str(jpg_file))
                    if frame is not None:
                        expected_name = event.get("metadata", {}).get("person_profile", {}).get("name")
                        test_data.append((expected_name, frame, str(jpg_file)))
            except Exception as e:
                print(f"Failed to load test image {json_file}: {e}")
        
        return test_data
    
    def benchmark_sface(self):
        """Benchmark SFace algorithm."""
        print("\n" + "="*60)
        print("Benchmarking SFace")
        print("="*60)
        
        # Initialize SFace
        app = insightface.app.FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"]
        )
        app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
        
        test_data = self.load_test_images()
        print(f"Loaded {len(test_data)} test images")
        
        # Step 1: Create "known" embeddings from test images
        known_embeddings = {}
        known_names = []
        
        for name, frame, filepath in test_data:
            if name:
                faces = app.get(frame)
                if len(faces) > 0:
                    face = faces[0]
                    known_embeddings[name] = face.embedding.astype(np.float32)
                    known_names.append(name)
                    print(f"Created embedding for {name} from {Path(filepath).name}")
        
        if len(known_embeddings) == 0:
            print("No faces found in test images")
            return
        
        # Step 2: Test recognition on the same images
        correct = 0
        total_time = 0
        
        for expected_name, frame, filepath in test_data:
            if not expected_name:
                continue
                
            start = time.time()
            faces = app.get(frame)
            elapsed = (time.time() - start) * 1000  # ms
            
            if len(faces) > 0:
                test_embedding = faces[0].embedding.astype(np.float32)
                
                # Find best match
                best_match = None
                best_score = 0.0
                
                for name, known_emb in known_embeddings.items():
                    # Normalize for cosine similarity
                    test_norm = test_embedding / (np.linalg.norm(test_embedding) + 1e-5)
                    known_norm = known_emb / (np.linalg.norm(known_emb) + 1e-5)
                    similarity = np.dot(test_norm, known_norm)
                    score = (similarity + 1) / 2  # Convert to 0-1 range
                    
                    if score > best_score:
                        best_score = score
                        best_match = name
                
                is_correct = (best_match == expected_name) and (best_score > 0.6)
                status = "✓" if is_correct else "✗"
                
                if is_correct:
                    correct += 1
                
                print(f"{status} Expected: {expected_name:15} | Got: {best_match or 'None':15} | "
                      f"Score: {best_score:.3f} | Time: {elapsed:.2f}ms")
            else:
                print(f"✗ Expected: {expected_name:15} | Got: No face detected | Time: {elapsed:.2f}ms")
            
            total_time += elapsed
        
        accuracy = (correct / len(test_data) * 100) if test_data else 0
        avg_time = (total_time / len(test_data)) if test_data else 0
        
        self.results["SFace"] = {
            "accuracy": accuracy,
            "speed_ms": avg_time,
            "matches": correct,
            "total_tests": len(test_data)
        }
        
        print(f"\nSFace Results: Accuracy: {accuracy:.1f}% | Avg Time: {avg_time:.2f}ms")
    
    def benchmark_arcface(self):
        """Benchmark ArcFace algorithm."""
        print("\n" + "="*60)
        print("Benchmarking ArcFace")
        print("="*60)
        
        # Initialize ArcFace
        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"]
        )
        app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
        
        test_data = self.load_test_images()
        print(f"Loaded {len(test_data)} test images")
        
        # Step 1: Create "known" embeddings from test images
        known_embeddings = {}
        known_names = []
        
        for name, frame, filepath in test_data:
            if name:
                faces = app.get(frame)
                if len(faces) > 0:
                    face = faces[0]
                    known_embeddings[name] = face.embedding.astype(np.float32)
                    known_names.append(name)
                    print(f"Created embedding for {name} from {Path(filepath).name}")
        
        if len(known_embeddings) == 0:
            print("No faces found in test images")
            return
        
        # Step 2: Test recognition on the same images
        correct = 0
        total_time = 0
        
        for expected_name, frame, filepath in test_data:
            if not expected_name:
                continue
                
            start = time.time()
            faces = app.get(frame)
            elapsed = (time.time() - start) * 1000  # ms
            
            if len(faces) > 0:
                test_embedding = faces[0].embedding.astype(np.float32)
                
                # Find best match
                best_match = None
                best_score = 0.0
                
                for name, known_emb in known_embeddings.items():
                    # Normalize for cosine similarity
                    test_norm = test_embedding / (np.linalg.norm(test_embedding) + 1e-5)
                    known_norm = known_emb / (np.linalg.norm(known_emb) + 1e-5)
                    similarity = np.dot(test_norm, known_norm)
                    score = (similarity + 1) / 2  # Convert to 0-1 range
                    
                    if score > best_score:
                        best_score = score
                        best_match = name
                
                is_correct = (best_match == expected_name) and (best_score > 0.6)
                status = "✓" if is_correct else "✗"
                
                if is_correct:
                    correct += 1
                
                print(f"{status} Expected: {expected_name:15} | Got: {best_match or 'None':15} | "
                      f"Score: {best_score:.3f} | Time: {elapsed:.2f}ms")
            else:
                print(f"✗ Expected: {expected_name:15} | Got: No face detected | Time: {elapsed:.2f}ms")
            
            total_time += elapsed
        
        accuracy = (correct / len(test_data) * 100) if test_data else 0
        avg_time = (total_time / len(test_data)) if test_data else 0
        
        self.results["ArcFace"] = {
            "accuracy": accuracy,
            "speed_ms": avg_time,
            "matches": correct,
            "total_tests": len(test_data)
        }
        
        print(f"\nArcFace Results: Accuracy: {accuracy:.1f}% | Avg Time: {avg_time:.2f}ms")
    
    def benchmark_facenet(self):
        """Benchmark FaceNet algorithm."""
        print("\n" + "="*60)
        print("Benchmarking FaceNet")
        print("="*60)
        
        # Initialize FaceNet
        device = "cpu"
        mtcnn = MTCNN(device=device, keep_all=False)
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        test_data = self.load_test_images()
        print(f"Loaded {len(test_data)} test images")
        
        # Step 1: Create "known" embeddings from test images
        known_embeddings = {}
        known_names = []
        
        for name, frame, filepath in test_data:
            if name:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_tensor = mtcnn(frame_rgb)
                    if face_tensor is not None:
                        with torch.no_grad():
                            embedding = model(face_tensor.unsqueeze(0).to(device))
                        known_embeddings[name] = embedding.cpu().numpy().flatten().astype(np.float32)
                        known_names.append(name)
                        print(f"Created embedding for {name} from {Path(filepath).name}")
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")
        
        if len(known_embeddings) == 0:
            print("No faces found in test images")
            return
        
        # Step 2: Test recognition on the same images
        correct = 0
        total_time = 0
        
        for expected_name, frame, filepath in test_data:
            if not expected_name:
                continue
                
            start = time.time()
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_tensor = mtcnn(frame_rgb)
                elapsed = (time.time() - start) * 1000  # ms
                
                if face_tensor is not None:
                    with torch.no_grad():
                        test_embedding = model(face_tensor.unsqueeze(0).to(device))
                    test_embedding = test_embedding.cpu().numpy().flatten().astype(np.float32)
                    
                    # Find best match using Euclidean distance
                    best_match = None
                    best_distance = float('inf')
                    
                    for name, known_emb in known_embeddings.items():
                        distance = np.linalg.norm(test_embedding - known_emb)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match = name
                    
                    # Convert distance to confidence
                    confidence = 1.0 / (1.0 + best_distance)
                    is_correct = (best_match == expected_name) and (best_distance < 0.6)
                    status = "✓" if is_correct else "✗"
                    
                    if is_correct:
                        correct += 1
                    
                    print(f"{status} Expected: {expected_name:15} | Got: {best_match or 'None':15} | "
                          f"Distance: {best_distance:.3f} | Time: {elapsed:.2f}ms")
                else:
                    print(f"✗ Expected: {expected_name:15} | Got: No face detected | Time: {elapsed:.2f}ms")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"✗ Expected: {expected_name:15} | Error: {str(e)[:50]}... | Time: {elapsed:.2f}ms")
            
            total_time += elapsed
        
        accuracy = (correct / len(test_data) * 100) if test_data else 0
        avg_time = (total_time / len(test_data)) if test_data else 0
        
        self.results["FaceNet"] = {
            "accuracy": accuracy,
            "speed_ms": avg_time,
            "matches": correct,
            "total_tests": len(test_data)
        }
        
        print(f"\nFaceNet Results: Accuracy: {accuracy:.1f}% | Avg Time: {avg_time:.2f}ms")
    
    def print_summary(self):
        """Print summary of all benchmark results."""
        print("\n" + "="*80)
        print("FACE RECOGNITION BENCHMARK - SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No benchmark results available")
            return
        
        print(f"\n{'Rank':<6} {'Algorithm':<15} {'Accuracy':<12} {'Speed (ms)':<12} {'Matches':<10}")
        print("-" * 80)
        
        # Sort by accuracy (descending), then by speed (ascending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (-x[1]["accuracy"], x[1]["speed_ms"])
        )
        
        for rank, (name, stats) in enumerate(sorted_results, 1):
            print(
                f"{rank:<6} {name:<15} {stats['accuracy']:>10.1f}% {stats['speed_ms']:>10.2f} "
                f"{stats['matches']:>8}/{stats['total_tests']:<1}"
            )
        
        print("\n" + "="*80)
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        
        if "SFace" in self.results and self.results["SFace"]["accuracy"] > 0:
            print("✓ SFace is recommended for real-time applications")
            print(f"  - Accuracy: {self.results['SFace']['accuracy']:.1f}%")
            print(f"  - Speed: {self.results['SFace']['speed_ms']:.2f}ms per frame")
            print("  - Best for: Wearable cameras, edge devices")
        
        if "ArcFace" in self.results and self.results["ArcFace"]["accuracy"] > 0:
            print("\n✓ ArcFace is recommended for high-accuracy applications")
            print(f"  - Accuracy: {self.results['ArcFace']['accuracy']:.1f}%")
            print(f"  - Speed: {self.results['ArcFace']['speed_ms']:.2f}ms per frame")
            print("  - Best for: High-accuracy identification")
        
        if "FaceNet" in self.results and self.results["FaceNet"]["accuracy"] > 0:
            print("\n✓ FaceNet is recommended for clustering/verification")
            print(f"  - Accuracy: {self.results['FaceNet']['accuracy']:.1f}%")
            print(f"  - Speed: {self.results['FaceNet']['speed_ms']:.2f}ms per frame")
            print("  - Best for: Flexible embedding space")


def main():
    """Run the realistic benchmark."""
    benchmark = RealFaceRecognitionBenchmark()
    
    print("="*80)
    print("REALISTIC FACE RECOGNITION BENCHMARK")
    print("="*80)
    print("\nThis benchmark:")
    print("1. Encodes faces from test images to create 'known' embeddings")
    print("2. Tests recognition on the same images")
    print("3. Measures accuracy and speed")
    print("\nNote: This tests self-recognition (matching against same images)")
    print("      In production, you'd use separate reference images.")
    print("="*80)
    
    # Run benchmarks
    benchmark.benchmark_sface()
    benchmark.benchmark_arcface()
    benchmark.benchmark_facenet()
    
    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
