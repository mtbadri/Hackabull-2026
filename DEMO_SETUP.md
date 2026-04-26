# Demo Setup Guide

## Files Committed for Demo

### Core Demo Scripts
- **test_camera_continuous.py** - Real-time camera test with live face recognition overlay
- **run_face_benchmark.py** - Face recognition algorithm benchmark runner
- **integration_example.py** - Integration example showing full system workflow
- **collect_profile_images.py** - Utility to collect and organize known face profiles

### Documentation
- **BENCHMARK_SUMMARY.md** - Summary of face recognition algorithm benchmarks
- **FINAL_BENCHMARK_REPORT.md** - Detailed benchmark report with performance metrics
- **tests/vision/BENCHMARK_README.md** - Vision testing documentation
- **tests/vision/QUICK_REFERENCE.md** - Quick reference for vision algorithms

### Vision Models
- **tests/vision/models/face_detection_yunet_2023mar.onnx** - YuNet face detection model
- **tests/vision/models/face_recognition_sface_2021dec.onnx** - SFace recognition model

### Test Scripts
- **tests/vision/real_benchmark.py** - Real-world benchmark testing
- **tests/vision/setup_benchmark.sh** - Benchmark setup script
- **tests/vision/benchmark_results.json** - Latest benchmark results

## Files Removed (Cleanup)

### Duplicate/Old Directories
- **brain/** - Empty duplicate directory
- **brain-new-with/** - Old brain service (consolidated into services/brain/)

### Temporary/Cache Files
- **.hypothesis/** - Hypothesis testing cache
- **.pytest_cache/** - Pytest cache
- **__pycache__/** - Python bytecode cache (all instances)
- **\*.pyc** - Compiled Python files
- **.DS_Store** - macOS metadata files

### Unnecessary Files
- **Hackabull-2026-main 614.zip** - Old zip archive
- **test_sface_dimension.py** - Test dimension script (not needed)
- **test_sface_dimension2.py** - Test dimension script (not needed)

## Current Project Structure

```
Hackabull-2026/
├── services/
│   ├── brain/          # AI Brain service (FastAPI)
│   ├── vision/         # Vision Engine (Flask)
│   ├── dashboard/      # Caregiver Portal (Streamlit)
│   └── webapp/         # Web application
├── tests/
│   ├── brain/          # Brain service tests
│   ├── vision/         # Vision tests and benchmarks
│   └── integration/    # Integration tests
├── shared/             # Shared contracts and utilities
├── tempfiles/          # Temporary event files
├── dashboard/          # Dashboard components
├── test_camera_continuous.py  # Live camera demo
├── run_all.py          # Launch all services
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Running the Demo

### 1. Continuous Camera Test
```bash
python test_camera_continuous.py
```
- Shows live camera feed with face recognition
- Press 'q' to quit, 's' to save snapshot, 'r' to reload faces

### 2. Face Recognition Benchmark
```bash
python run_face_benchmark.py
```
- Tests multiple face recognition algorithms
- Compares accuracy and speed

### 3. Full System Demo
```bash
python run_all.py
```
- Launches all three services:
  - Vision Engine: http://localhost:5000
  - AI Brain: http://localhost:8000
  - Caregiver Portal: http://localhost:8501

## Git Status

**Branch:** test  
**Last Commit:** Add demo files and cleanup unnecessary directories  
**Status:** Clean working tree - all changes committed and pushed

## Next Steps for Demo

1. Ensure all API keys are set in `.env`
2. Verify known faces are in `services/vision/known_faces/`
3. Test camera access with `test_camera_continuous.py`
4. Run full system with `run_all.py`
5. Open dashboard at http://localhost:8501

## Notes

- `.env` file is gitignored (contains API keys)
- Known face images are gitignored (privacy)
- Virtual environment `.venv/` is gitignored
- All Python cache files are now excluded
