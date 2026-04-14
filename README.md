# DeepShield — AI Face-Swap Deepfake Detector

> Detect face-swap deepfakes in images and videos using a pretrained transformer model with spatial heatmap explainability.

---

## Project Structure

```
deepfake-detector/
├── model/
│   └── detector.py       # Core pipeline: face detection + classification + heatmap
├── api/
│   └── main.py           # FastAPI backend (REST endpoints)
├── frontend/
│   └── index.html        # Single-page browser UI (no build step needed)
├── requirements.txt      # Python dependencies
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
| Runtime | Python 3.10+ |

---

## How It Works

```
Input (Image or Video)
        ↓
Face Detection — OpenCV Haar Cascade
        ↓
Crop each face region (+ padding)
        ↓
Transformer Classifier — prithivMLmods/Deep-Fake-Detector-Model
        ↓
REAL / FAKE probability per face
        ↓
Heatmap overlay (red = high fake probability regions)
        ↓
JSON response → UI renders verdict + heatmap + frame timeline
```

For **videos**, every Nth frame is sampled and classified. The final verdict is FAKE if more than 30% of analyzed frames are classified as fake.

---

## Requirements

- Python 3.10 or 3.11 (recommended)
- No C++ compiler needed — OpenCV replaces dlib entirely
- ~400MB disk space for model download (first run only, cached automatically)

---

## Setup & Installation

### Step 1 — Clone / download the project

Make sure your folder looks like this:
```
deepfake_detector/
├── model/detector.py
├── api/main.py
├── frontend/index.html
└── requirements.txt
```

### Step 2 — Create a virtual environment

```powershell
# Windows (PowerShell)
py -3.10 -m venv venv
venv\Scripts\activate
```

```bash
# macOS / Linux
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch, Transformers, OpenCV, FastAPI, and all other dependencies. No cmake or Visual C++ needed.

---

## Running the Project

You need **two terminals** running simultaneously.

### Terminal 1 — Start the backend API

```powershell
cd deepfake_detector/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**First run:** The model (~400MB) will be downloaded from HuggingFace automatically. This happens once and is cached.

Wait until you see:
```
[INFO] Model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify the API is working:**

Open in browser → `http://localhost:8000/health`

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

Interactive API docs → `http://localhost:8000/docs`

### Terminal 2 — Serve the frontend

```powershell
cd deepfake_detector
python -m http.server 3000 -d frontend
```

Then open your browser at: **`http://localhost:3000`**

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

### `GET /health`
Returns server status and model load state.

### `POST /analyze/image`
Upload a JPG, PNG, or WebP image.

**Request:** `multipart/form-data` with field `file`

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
Upload an MP4, AVI, or MOV file (max 100MB).

Optional query param: `?sample_rate=10` (analyze every Nth frame, default 10)

**Additional response fields for video:**
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

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Cannot connect to backend` | API not running | Start uvicorn in Terminal 1 |
| `Model not loaded` error | Import failed on startup | Check terminal for traceback |
| `503 Service Unavailable` | Model load error | Check terminal — usually a missing package |
| `400 Unsupported type` | Wrong file format | Use JPG/PNG for images, MP4/MOV for video |
| No face detected | Face too small or side-on | Use a clear frontal face image |
| Slow analysis on video | CPU inference, long video | Increase `sample_rate` param (e.g. `?sample_rate=30`) |

---

## Model Details

**Model:** `prithivMLmods/Deep-Fake-Detector-Model` (HuggingFace)

Trained to distinguish real human faces from AI-generated or face-swapped images. The model is loaded via the `transformers` library and runs inference on cropped face regions.

**Face detection:** OpenCV's Haar Cascade (`haarcascade_frontalface_default.xml`) — lightweight, requires no compilation, works on all platforms.

**Heatmap generation:** Each detected face region is highlighted with intensity proportional to its fake probability score, then blended onto the original frame using OpenCV's JET colormap.

---

## Limitations

- Works best on clear, frontal, well-lit faces
- Detection accuracy decreases on heavily compressed video (e.g. after social media re-encoding)
- Audio-visual deepfakes (lip-sync manipulation) are not detected — visual only
- Performance on very novel GAN architectures may vary

---

## Future Work

The following improvements and extensions are planned or recommended for future development iterations:

- Short-Term Improvements

• GPU Acceleration — add CUDA batching for video frames to reduce video analysis time from minutes to seconds on NVIDIA hardware.
• Audio Deepfake Detection — extend the pipeline to analyse audio tracks for voice-cloning artifacts using a separate audio classifier.
• Confidence Calibration — apply temperature scaling post-training to ensure predicted probabilities are well-calibrated and not overconfident.
• CORS Hardening — restrict allow_origins in main.py to specific trusted domains before production deployment.

- Medium-Term Extensions

• Persistent Storage — replace the in-memory LRU cache with a PostgreSQL or Redis backend to survive server restarts and support multi-instance deployments.
• Model Versioning — add an endpoint to report the active model version and support hot-swapping models without restarting the server.
• Batch API — accept zip archives of images for bulk analysis, useful for content moderation at scale.
• User Authentication — add JWT-based auth and per-user rate limiting for public-facing deployments.

- Long-Term Research Directions

• Adversarial Robustness — evaluate and harden the model against adversarial perturbations specifically designed to fool deepfake detectors.
• Multi-Modal Analysis — combine face, background, and audio signals into a unified multi-modal confidence score.
• Continual Learning — implement a feedback loop where flagged results reviewed by human moderators are periodically used to fine-tune the model on emerging manipulation techniques.
• Mobile Deployment — export the model to ONNX or TensorFlow Lite for on-device inference in iOS/Android applications, eliminating the need for a server.
---

## License

MIT License. See `LICENSE` for details.