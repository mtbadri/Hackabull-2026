#!/usr/bin/env python3
"""
Quick runner for face recognition algorithm benchmark.

Usage:
    python run_face_benchmark.py
    python run_face_benchmark.py --algorithm arcface
    python run_face_benchmark.py --help
"""

import sys
import argparse
import logging
from pathlib import Path

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.vision.test_face_algorithms import (
    BenchmarkSuite,
    ArcFaceAlgorithm,
    FaceNetAlgorithm,
    SFaceAlgorithm,
    GeminiShirtColorAlgorithm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Algorithm Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_face_benchmark.py                    # Run all algorithms
  python run_face_benchmark.py --algorithm sface  # Test only SFace
  python run_face_benchmark.py --algorithm gemini # Test only Gemini
  python run_face_benchmark.py --list             # List available algorithms
        """,
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["arcface", "facenet", "sface", "gemini", "all"],
        default="all",
        help="Algorithm to test (default: all)",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available algorithms and exit",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="tests/vision/benchmark_results.json",
        help="Output file for results (default: tests/vision/benchmark_results.json)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List algorithms
    if args.list:
        print("\nAvailable Face Recognition Algorithms:")
        print("  - arcface   : State-of-the-art accuracy (95-99%)")
        print("  - facenet   : Unified embedding approach (92-98%)")
        print("  - sface     : Lightweight & fast (90-96%)")
        print("  - gemini    : Current implementation (85-95%)")
        print("  - all       : Run all algorithms (default)")
        print("\nFor more info, see: tests/vision/ALGORITHM_COMPARISON.md")
        return 0
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Add requested algorithms
    algorithms_to_test = []
    
    if args.algorithm in ("all", "gemini"):
        algorithms_to_test.append(("Gemini", GeminiShirtColorAlgorithm))
    
    if args.algorithm in ("all", "arcface"):
        algorithms_to_test.append(("ArcFace", ArcFaceAlgorithm))
    
    if args.algorithm in ("all", "facenet"):
        algorithms_to_test.append(("FaceNet", FaceNetAlgorithm))
    
    if args.algorithm in ("all", "sface"):
        algorithms_to_test.append(("SFace", SFaceAlgorithm))
    
    # Initialize algorithms
    print(f"\n{'='*60}")
    print("Face Recognition Algorithm Benchmark")
    print(f"{'='*60}\n")
    
    for name, AlgoClass in algorithms_to_test:
        try:
            algo = AlgoClass()
            suite.add_algorithm(algo)
            print(f"✓ {name} initialized")
        except Exception as e:
            print(f"✗ {name} failed to initialize: {e}")
    
    if len(suite.algorithms) == 0:
        print("\n✗ No algorithms could be initialized!")
        print("\nTo install dependencies:")
        print("  pip install -r tests/vision/requirements_benchmark.txt")
        return 1
    
    print(f"\nTesting {len(suite.algorithms)} algorithm(s)...\n")
    
    # Run benchmark
    try:
        suite.run_benchmark()
        suite.print_results()
        suite.save_results(args.output)
        print(f"\n✓ Results saved to: {args.output}")
        return 0
    except Exception as e:
        log.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
