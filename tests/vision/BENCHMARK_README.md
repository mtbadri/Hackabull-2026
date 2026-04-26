# Face Recognition Algorithm Benchmark

This benchmark suite compares four face recognition approaches:

1. **Gemini (Shirt Color)** - Your current implementation using Google's Gemini vision API
2. **ArcFace** - State-of-the-art deep learning approach with additive angular margin loss
3. **FaceNet** - Google's unified embedding approach for face recognition
4. **SFace** - Efficient lightweight model optimized for speed

## Quick Start

### 1. Install Dependencies

```bash
# Install benchmark dependencies
pip install -r tests/vision/requirements_benchmark.txt
```

### 2. Prepare Test Data

The benchmark uses images from `tempfiles/event_*.jpg` paired with metadata in `event_*.json`.

If you don't have test images yet, the benchmark will create synthetic test data.

### 3. Run the Benchmark

```bash
# Run full benchmark
python -m pytest tests/vision/test_face_algorithms.py::test_full_benchmark -v -s

# Or run directly
python tests/vision/test_face_algorithms.py
```

### 4. View Results

Results are saved to `tests/vision/benchmark_results.json` with a ranking table printed to console.

## Algorithm Comparison

### Gemini (Shirt Color) - Current Implementation
**Pros:**
- No model training required
- Works with any camera angle
- Contextual understanding (can detect health activities)
- No GPU required
- Handles partial faces well

**Cons:**
- Requires API calls (latency, cost)
- Depends on visible shirt color
- Rate-limited by API
- ~500-1000ms per call

**Best for:** Contextual awareness, health monitoring, flexible scenarios

---

### ArcFace
**Pros:**
- State-of-the-art accuracy (99%+ on LFW benchmark)
- Fast inference (~50-100ms)
- Robust to pose variations
- No API calls needed
- Highly optimized models available

**Cons:**
- Requires GPU for real-time performance
- Needs training data for new people
- Sensitive to lighting conditions
- Embedding dimension: 512

**Best for:** High-accuracy identification, offline operation, performance-critical systems

---

### FaceNet
**Pros:**
- Excellent accuracy (99.6% on LFW)
- Unified embedding space
- Good generalization
- Multiple model sizes available

**Cons:**
- Slower than ArcFace (~100-200ms on CPU)
- Requires GPU for real-time use
- Larger model size
- Embedding dimension: 128-512

**Best for:** Flexible embedding space, clustering, verification tasks

---

### SFace
**Pros:**
- Lightweight and fast (~30-50ms)
- Good accuracy (98%+ on benchmarks)
- Lower memory footprint
- Efficient for edge devices

**Cons:**
- Smaller embedding dimension (128)
- Less robust to extreme poses
- Fewer pre-trained models available
- Newer, less battle-tested

**Best for:** Edge devices, real-time systems, resource-constrained environments

## Benchmark Metrics

### Accuracy
- Percentage of correct matches against known faces
- Higher is better

### Speed
- Average inference time in milliseconds
- Lower is better
- Measured on CPU (no GPU acceleration)

### Matches
- Number of correct identifications / total tests

### Errors
- Number of processing failures

## Expected Results (CPU Baseline)

| Algorithm | Accuracy | Speed (ms) | Notes |
|-----------|----------|-----------|-------|
| Gemini | 85-95% | 500-1000 | API-dependent |
| ArcFace | 95-99% | 80-150 | Requires GPU for <50ms |
| FaceNet | 92-98% | 120-200 | Slower on CPU |
| SFace | 90-96% | 40-80 | Fastest option |

*Note: Actual results depend on image quality, lighting, pose, and test dataset.*

## Customizing the Benchmark

### Add Custom Test Images

Place face images in `tempfiles/` with corresponding JSON metadata:

```json
{
  "metadata": {
    "person_profile": {
      "name": "person_name"
    }
  }
}
```

### Adjust Thresholds

Edit the `match_face()` threshold parameter in each algorithm class:

```python
matched_name, confidence = algo.match_face(frame, threshold=0.6)
```

### Test Specific Algorithm

```python
from tests.vision.test_face_algorithms import BenchmarkSuite, ArcFaceAlgorithm

suite = BenchmarkSuite()
suite.add_algorithm(ArcFaceAlgorithm())
suite.run_benchmark()
suite.print_results()
```

## Recommendations

### For Your Use Case (Wearable Camera + Health Monitoring)

**Hybrid Approach:**
1. Use **SFace** for fast, local face detection (no API calls)
2. Fall back to **Gemini** for contextual health activity detection
3. Use **ArcFace** for high-confidence verification when GPU available

**Implementation:**
```python
# Fast local check
sface_match, sface_conf = sface.match_face(frame)

if sface_conf > 0.8:
    # High confidence - use it
    return sface_match

# Lower confidence - ask Gemini for context
gemini_match, gemini_conf = gemini.match_face(frame)
if gemini_conf > 0.65:
    return gemini_match

# No match
return None
```

### Production Deployment

**Recommended Stack:**
- **Primary:** ArcFace (with GPU) for accuracy
- **Secondary:** SFace (CPU) for fallback
- **Tertiary:** Gemini for contextual verification

**Cost Optimization:**
- Cache embeddings for known faces
- Use SFace for initial screening
- Only call Gemini when confidence is low

## Troubleshooting

### ImportError: No module named 'insightface'

```bash
pip install insightface onnxruntime
```

### ImportError: No module named 'facenet_pytorch'

```bash
pip install facenet-pytorch torch torchvision
```

### CUDA/GPU Issues

The benchmark runs on CPU by default. To use GPU:

```python
# In algorithm classes, change:
self.device = "cuda"  # or "cpu"
```

### Gemini API Errors

Ensure `GEMINI_API_KEY` is set in `.env`:

```bash
echo "GEMINI_API_KEY=your_key_here" >> .env
```

## References

- **ArcFace:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **FaceNet:** [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- **SFace:** [SFace: An Efficient Network for Face Recognition via Learning to Equip with Sufficient Feature](https://arxiv.org/abs/2104.12225)
- **InsightFace:** [InsightFace: 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface)

## Next Steps

1. Run the benchmark with your actual test data
2. Review `benchmark_results.json` for detailed metrics
3. Choose the best algorithm(s) for your use case
4. Integrate into `services/vision/face_recognition_engine.py`
5. Monitor performance in production

---

**Last Updated:** April 2026
