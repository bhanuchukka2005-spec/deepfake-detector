# 🛡️ DeepShield — AI Face-Swap Deepfake Detector

> Detect face-swap deepfakes in images and videos using a pretrained transformer model, OpenCV face detection, and a FastAPI backend with a drag-and-drop frontend.

[![CI](https://github.com/bhanuchukka2005-spec/deepfake-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/bhanuchukka2005-spec/deepfake-detector/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=flat-square&logo=opencv)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Project Structure

```
deepfake-detector/
├── model/
│   └── detector.py          # Core pipeline: face detection + classification + heatmap
├── api/
│   └── main.py              # FastAPI backend (REST endpoints + LRU cache)
├── frontend/
│   └── index.html           # Single-page drag-and-drop UI (no build step)
├── tests/
│   └── test_api.py          # pytest test suite (mock-based, no model download)
├── .github/
│   └── workflows/
│       └── ci.yml           # CI: lint + security + tests + docker build
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Classification Model | `prithivMLmods/Deep-Fake-Detector-Model` (HuggingFace Transformers) |
| Face Detection | OpenCV Haar Cascade (no dlib/cmake required) |
| Heatmap | Weighted face-region overlay based on fake probability |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| CI/CD | GitHub Actions |
| Container | Docker |
| Runtime | Python 3.11 |

---

## How It Works

```
Input (Image or Video)
        ↓
Face Detection — OpenCV Haar Cascade
        ↓
Crop each face region (+ 20px padding)
        ↓
Transformer Classifier — prithivMLmods/Deep-Fake-Detector-Model
        ↓
REAL / FAKE probability per face
        ↓
Heatmap overlay (JET colormap — blue=real, red=fake)
        ↓
Verdict aggregation
  Image: highest fake confidence across all faces
  Video: FAKE if >30% of sampled frames are fake
        ↓
JSON response + heatmap URL → UI renders verdict
```

---

## Key Engineering Decisions

**Why Haar Cascade instead of MTCNN or RetinaFace?**
Haar Cascade requires zero additional dependencies and runs in milliseconds on CPU. MTCNN is more accurate but needs a GPU for reasonable speed. Given the target of sub-5s inference on CPU, Haar is the right trade-off. A full-frame fallback handles cases where no face is detected.

**Why delete files immediately after analysis?**
Privacy by design. The system only needs the inference result — not the original media. Processing and deleting on the fly means no sensitive media accumulates on the server.

**Why LRU cache instead of a database?**
Results are small JSON objects identified by a short UUID. Bounded in-memory LRU (200 entries) gives O(1) lookups with no database overhead. For production with persistence requirements, swap for Redis.

**Why majority vote at >30% fake threshold for video?**
A video where 30% of frames are fake is already highly suspicious. Setting the threshold at 50% would miss videos where only some scenes are manipulated — which is how real deepfake attacks work.

---

## Requirements

- Python 3.10 or 3.11 (recommended)
- No C++ compiler needed — OpenCV replaces dlib entirely
- ~400MB disk space for model download (first run only, cached automatically)

---

## Setup & Installation

### Local (without Docker)

**Step 1 — Clone the project**

```bash
git clone https://github.com/bhanuchukka2005-spec/deepfake-detector
cd deepfake-detector
```

**Step 2 — Create a virtual environment**

```powershell
# Windows
py -3.11 -m venv venv
venv\Scripts\activate
```

```bash
# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

**Step 3 — Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4 — Start the backend**

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> First run downloads the model (~400MB) from HuggingFace automatically. Cached after that.

Wait until you see:
```
[INFO] Model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Step 5 — Serve the frontend**

Open a second terminal:

```bash
cd deepfake-detector
python -m http.server 3000 -d frontend
```

Open **http://localhost:3000** in your browser.

---

### With Docker

```bash
git clone https://github.com/bhanuchukka2005-spec/deepfake-detector
cd deepfake-detector
docker-compose up --build
```

Open **http://localhost:8000**

> First run downloads the model inside the container (~400MB). Subsequent starts are instant.

---

## Using the App

1. Select **Image** or **Video** tab
2. Drag and drop a file or click to browse
3. Click **Analyze for Deepfake**
4. Results show:
   - **Verdict** — REAL or FAKE
   - **Risk Score** — 0–100% manipulation likelihood
   - **Confidence** — model certainty
   - **Heatmap** — visual overlay showing artifact regions
   - **Frame timeline** — per-frame breakdown (video only)

---

## API Reference

Interactive docs: **http://localhost:8000/docs**

### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2025-04-21T10:30:00"
}
```

### `POST /analyze/image`

**Request:** `multipart/form-data` with field `file` (JPG, PNG, WebP)

**Response:**
```json
{
  "analysis_id": "a3f8c2b1",
  "input_type": "image",
  "final_verdict": "FAKE",
  "confidence": 0.891,
  "risk_score": 89.1,
  "faces_detected": 1,
  "processing_time_ms": 432.1,
  "explanation": "Strong indicators of face-swap manipulation detected...",
  "heatmap_url": "/results_static/a3f8c2b1_heatmap.jpg",
  "face_results": [
    {
      "bbox": [120, 340, 310, 150],
      "label": "FAKE",
      "fake_probability": 0.891,
      "real_probability": 0.109
    }
  ]
}
```

### `POST /analyze/video`

**Request:** `multipart/form-data` with field `file` (MP4, AVI, MOV — max 100MB)

Optional query param: `?sample_rate=10` (analyze every Nth frame, default 10)

**Additional response fields:**
```json
{
  "total_frames": 720,
  "analyzed_frames": 72,
  "fake_frame_ratio": 0.61,
  "duration_seconds": 24.3
}
```

### `GET /results/{analysis_id}`

Retrieve a previously cached result by its ID.

---

## CI/CD Pipeline

Every push to `main` or pull request triggers:

| Job | What it does |
|-----|-------------|
| Lint | ruff checks `api/` and `model/` for code quality |
| Security | bandit scans for vulnerabilities, safety audits dependencies |
| Tests | pytest runs mock-based test suite — no model download needed |
| Docker Build | builds image and verifies it exists |

---

## Running Tests Locally

```bash
pip install pytest httpx
pytest tests/ -v
```

Tests use mocks — the real model is never downloaded during testing.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Cannot connect to backend` | API not running | Start uvicorn in terminal |
| `Model not loaded` error | Import failed on startup | Check terminal for traceback |
| `503 Service Unavailable` | Model load error | Check terminal — usually a missing package |
| `400 Unsupported type` | Wrong file format | Use JPG/PNG for images, MP4/MOV for video |
| No face detected | Face too small or side-on | Use a clear frontal face image |
| Slow analysis on video | CPU inference, long video | Increase `sample_rate` (e.g. `?sample_rate=30`) |

---

## Model Details

**Model:** `prithivMLmods/Deep-Fake-Detector-Model` (HuggingFace)

Trained to distinguish real human faces from AI-generated or face-swapped images. Loaded via the `transformers` library, runs inference on cropped face regions.

**Face detection:** OpenCV Haar Cascade — lightweight, no compilation needed, works on all platforms.

**Heatmap generation:** Each detected face region is weighted by its fake probability score, normalised to 0–255, coloured with JET colormap (blue=low, red=high), and blended onto the original frame at 40% opacity.

---

## Limitations

- Works best on clear, frontal, well-lit faces
- Detection accuracy decreases on heavily compressed video
- Audio-visual deepfakes (lip-sync manipulation) are not detected — visual only
- Performance on very novel GAN architectures may vary

---

## Interview Q&A

**Q: What is the inference pipeline step by step?**
Haar Cascade detects face bounding boxes → each face is cropped with 20px padding → PIL converts to RGB → HuggingFace processor tokenises the image → transformer outputs logit scores → softmax gives P(FAKE) and P(REAL) → highest fake confidence determines the frame verdict.

**Q: How does the heatmap work exactly?**
A zero float32 numpy array is initialised the same size as the frame. For each detected face rectangle, those pixels are assigned the face's fake probability. The array is normalised to 0–255, passed through `cv2.applyColorMap(COLORMAP_JET)`, and blended with the original frame using `cv2.addWeighted` at 40% opacity.

**Q: Why >30% threshold for video?**
Real deepfake attacks are often partial — only some scenes are swapped. A 50% threshold would classify a video with every other frame faked as REAL. 30% catches partial manipulation that matters in practice.

**Q: What would you change for production?**
Replace Haar Cascade with a GPU-accelerated face detector, swap the in-memory LRU cache for Redis, add JWT auth and rate limiting, implement async video processing via a task queue, and extend to audio deepfake detection.

---

## Future Work

**Short-Term**
- GPU acceleration — CUDA batching for video frames
- Audio deepfake detection — extend pipeline to voice-cloning artifacts
- CORS hardening — restrict `allow_origins` to trusted domains

**Medium-Term**
- Persistent storage — replace LRU cache with Redis for multi-instance deployments
- Batch API — accept zip archives for bulk content moderation
- User authentication — JWT auth with per-user rate limiting

**Long-Term**
- Adversarial robustness — harden against perturbations designed to fool detectors
- Multi-modal analysis — combine face, background, and audio signals
- Mobile deployment — export to ONNX / TensorFlow Lite for on-device inference

---

## License

MIT License. See `LICENSE` for details.

---

## Author

**Ch. Bhanu Prakash** — CS Engineering, Presidency University, Bengaluru

[GitHub](https://github.com/bhanuchukka2005-spec) · [LinkedIn](https://linkedin.com/in/chukka-bhanu-prakash) · [LeetCode](https://leetcode.com/u/Bhanu_heroo7)