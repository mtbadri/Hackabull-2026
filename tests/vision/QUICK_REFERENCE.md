# Face Recognition Benchmark - Quick Reference

## 🚀 Get Started in 3 Steps

### Step 1: Install
```bash
pip install -r tests/vision/requirements_benchmark.txt
```

### Step 2: Run
```bash
python run_face_benchmark.py
```

### Step 3: Review
Check `tests/vision/benchmark_results.json` for detailed results.

---

## 📊 Algorithm Comparison at a Glance

```
┌─────────────┬──────────┬──────────┬──────────┬──────────┐
│ Algorithm   │ Accuracy │ Speed    │ Cost     │ Best For │
├─────────────┼──────────┼──────────┼──────────┼──────────┤
│ ArcFace     │ 95-99%   │ 80-150ms │ Free     │ Accuracy │
│ FaceNet     │ 92-98%   │ 120-200ms│ Free     │ Flexible │
│ SFace       │ 90-96%   │ 40-80ms  │ Free     │ Speed ⭐ │
│ Gemini      │ 85-95%   │ 500-1000ms│ $0.15/1K│ Context  │
└─────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 🎯 Recommendation for Your Wearable Camera

### Hybrid Approach (Best)
```
Frame → SFace (40-80ms) → Confidence > 0.85?
                              ├─ YES → Use match
                              └─ NO → Gemini (500-1000ms) + Context
```

**Why?**
- ✓ Fast response (40-80ms most of the time)
- ✓ Contextual awareness (health monitoring)
- ✓ Privacy-first (local processing by default)
- ✓ Cost-effective (minimal API calls)
- ✓ Offline capability

---

## 📁 File Structure

```
tests/vision/
├── test_face_algorithms.py          # Main benchmark suite (600+ lines)
├── BENCHMARK_README.md              # Setup & usage guide
├── ALGORITHM_COMPARISON.md          # Detailed technical comparison
├── QUICK_REFERENCE.md               # This file
├── requirements_benchmark.txt       # Dependencies
└── setup_benchmark.sh               # Automated setup

Root:
├── run_face_benchmark.py            # Easy CLI runner
└── BENCHMARK_SUMMARY.md             # Overview & recommendations
```

---

## 🔧 Common Commands

### Run All Algorithms
```bash
python run_face_benchmark.py
```

### Test Specific Algorithm
```bash
python run_face_benchmark.py --algorithm sface
python run_face_benchmark.py --algorithm arcface
python run_face_benchmark.py --algorithm gemini
```

### List Available Algorithms
```bash
python run_face_benchmark.py --list
```

### Run with Pytest
```bash
python -m pytest tests/vision/test_face_algorithms.py::test_full_benchmark -v -s
```

### Run Individual Tests
```bash
pytest tests/vision/test_face_algorithms.py::test_sface_basic -v
pytest tests/vision/test_face_algorithms.py::test_arcface_basic -v
pytest tests/vision/test_face_algorithms.py::test_facenet_basic -v
pytest tests/vision/test_face_algorithms.py::test_gemini_basic -v
```

---

## 📈 Expected Results

### Accuracy Ranking
1. **ArcFace** - 95-99% (Best accuracy)
2. **FaceNet** - 92-98%
3. **SFace** - 90-96% (Best for wearables)
4. **Gemini** - 85-95% (Current)

### Speed Ranking
1. **SFace** - 40-80ms (Fastest) ⭐
2. **ArcFace** - 80-150ms
3. **FaceNet** - 120-200ms
4. **Gemini** - 500-1000ms (API overhead)

### Cost Ranking
1. **SFace** - Free (local)
2. **ArcFace** - Free (local)
3. **FaceNet** - Free (local)
4. **Gemini** - $0.15 per 1000 calls

---

## 🔍 Algorithm Details

### ArcFace
- **What:** Deep CNN with angular margin loss
- **Accuracy:** 99.83% on LFW benchmark
- **Speed:** 10-20ms (GPU), 80-150ms (CPU)
- **Use:** High-accuracy identification
- **Pros:** State-of-the-art, robust, well-tested
- **Cons:** Requires GPU for real-time, needs training data

### FaceNet
- **What:** Inception-ResNet with triplet loss
- **Accuracy:** 99.6% on LFW benchmark
- **Speed:** 20-40ms (GPU), 120-200ms (CPU)
- **Use:** Clustering, verification
- **Pros:** Flexible embeddings, excellent accuracy
- **Cons:** Slower than ArcFace, requires GPU

### SFace
- **What:** Lightweight CNN optimized for edge
- **Accuracy:** 98%+ on benchmarks
- **Speed:** 5-15ms (GPU), 40-80ms (CPU)
- **Use:** Wearable devices, edge computing
- **Pros:** Fastest, smallest model, good accuracy
- **Cons:** Lower accuracy than ArcFace/FaceNet

### Gemini (Current)
- **What:** Google's vision API with shirt-color matching
- **Accuracy:** 85-95% (depends on shirt color)
- **Speed:** 500-1000ms (API call)
- **Use:** Contextual awareness, health monitoring
- **Pros:** No training, contextual understanding
- **Cons:** API calls, latency, privacy concerns

---

## 💡 Implementation Tips

### For Maximum Speed
```python
# Use SFace only
sface = SFaceAlgorithm()
name, conf = sface.match_face(frame)
```

### For Maximum Accuracy
```python
# Use ArcFace (requires GPU)
arcface = ArcFaceAlgorithm()
name, conf = arcface.match_face(frame)
```

### For Best Balance (Recommended)
```python
# Hybrid approach
sface = SFaceAlgorithm()
gemini = GeminiShirtColorAlgorithm()

# Try fast local first
name, conf = sface.match_face(frame, threshold=0.75)
if conf > 0.85:
    return name, conf

# Fall back to Gemini for context
name, conf = gemini.match_face(frame)
return name, conf
```

### For Privacy
```python
# Use SFace only (no API calls)
sface = SFaceAlgorithm()
name, conf = sface.match_face(frame)
```

---

## 🐛 Troubleshooting

### ImportError: No module named 'insightface'
```bash
pip install insightface onnxruntime
```

### ImportError: No module named 'facenet_pytorch'
```bash
pip install facenet-pytorch torch torchvision
```

### Gemini API errors
```bash
# Check .env file
echo "GEMINI_API_KEY=your_key_here" >> .env
```

### CUDA/GPU not found
- Benchmark runs on CPU by default
- To use GPU, edit algorithm classes and set `device = "cuda"`

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **BENCHMARK_README.md** | Complete setup & usage guide |
| **ALGORITHM_COMPARISON.md** | Detailed technical comparison |
| **QUICK_REFERENCE.md** | This file - quick lookup |
| **test_face_algorithms.py** | Full implementation |

---

## 🎓 Learning Resources

- **ArcFace Paper:** [Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698)
- **FaceNet Paper:** [Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)
- **SFace Paper:** [Efficient Network for Face Recognition](https://arxiv.org/abs/2104.12225)
- **InsightFace:** [GitHub Repository](https://github.com/deepinsight/insightface)

---

## ✅ Checklist

- [ ] Install dependencies: `pip install -r tests/vision/requirements_benchmark.txt`
- [ ] Run benchmark: `python run_face_benchmark.py`
- [ ] Review results: `tests/vision/benchmark_results.json`
- [ ] Read comparison: `tests/vision/ALGORITHM_COMPARISON.md`
- [ ] Choose approach: Hybrid (recommended)
- [ ] Integrate into production
- [ ] Test end-to-end latency
- [ ] Monitor accuracy in production

---

## 🚀 Next Steps

1. **Run the benchmark** with your test data
2. **Review the results** and comparison
3. **Choose your approach** (hybrid recommended)
4. **Integrate into production**
5. **Monitor performance** in real-world use

---

**Quick Links:**
- 📖 [Full Documentation](BENCHMARK_README.md)
- 🔬 [Technical Comparison](ALGORITHM_COMPARISON.md)
- 🏃 [Run Benchmark](../../../run_face_benchmark.py)
- 📊 [Results](benchmark_results.json)

---

**Last Updated:** April 2026
**Status:** Ready to use
