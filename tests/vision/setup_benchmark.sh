#!/bin/bash
# Setup script for face recognition benchmark

set -e

echo "=========================================="
echo "Face Recognition Benchmark Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install benchmark dependencies
echo ""
echo "Installing benchmark dependencies..."
pip install -r tests/vision/requirements_benchmark.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the benchmark:"
echo "  python -m pytest tests/vision/test_face_algorithms.py::test_full_benchmark -v -s"
echo ""
echo "Or run directly:"
echo "  python tests/vision/test_face_algorithms.py"
echo ""
echo "For more information, see:"
echo "  tests/vision/BENCHMARK_README.md"
echo ""
