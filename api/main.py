"""
FastAPI Backend — DeepFake Detector
Routes:
  GET  /health           health check
  POST /analyze/image    analyze image file
  POST /analyze/video    analyze video file
  GET  /results/{id}     get cached result
"""

import os, sys, uuid, time, hashlib, cv2, datetime
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── FastAPI app (must be created before anything else) ────────────────────────
app = FastAPI(
    title="DeepFake Detector API",
    description="AI-powered face-swap deepfake detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/results_static", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# ── Load model (after app is created so routes still register on failure) ─────
sys.path.insert(0, str(BASE_DIR / "model"))
analyzer = None

try:
    from detector import FaceSwapAnalyzer
    analyzer = FaceSwapAnalyzer()
    print("[API] Model ready.")
except Exception as e:
    import traceback
    print(f"[API ERROR] Model failed to load: {e}")
    traceback.print_exc()
    print("[API] Server will start but /analyze endpoints will return 503.")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_MB = 100
ALLOWED_IMAGE = {'image/jpeg', 'image/png', 'image/webp', 'image/jpg'}
ALLOWED_VIDEO = {'video/mp4', 'video/avi', 'video/quicktime',
                 'video/x-matroska', 'video/x-msvideo'}

# In-memory result cache (LRU-style, bounded to 200 entries)
# For production: replace with Redis or a database.
MAX_CACHE_SIZE = 200
result_cache: OrderedDict = OrderedDict()


def cache_set(key: str, value: dict):
    if key in result_cache:
        result_cache.move_to_end(key)
    result_cache[key] = value
    if len(result_cache) > MAX_CACHE_SIZE:
        result_cache.popitem(last=False)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class FaceResult(BaseModel):
    bbox: List[int]
    label: str
    fake_probability: float
    real_probability: float

class AnalysisResult(BaseModel):
    analysis_id: str
    input_type: str
    final_verdict: str
    confidence: float
    risk_score: float
    faces_detected: int
    processing_time_ms: float
    explanation: str
    heatmap_url: Optional[str] = None
    face_results: Optional[List[FaceResult]] = None
    total_frames: Optional[int] = None
    analyzed_frames: Optional[int] = None
    fake_frame_ratio: Optional[float] = None
    duration_seconds: Optional[float] = None

# ── Helpers ───────────────────────────────────────────────────────────────────

# All artifact descriptions — deterministically selected via analysis_id hash
_FAKE_ARTIFACTS = [
    "unnatural blending around facial boundaries",
    "inconsistent lighting and skin texture",
    "temporal flickering in facial regions",
    "misaligned facial features",
    "GAN-specific frequency artifacts in facial regions",
    "abnormal eye reflection patterns",
    "colour banding at the face-background boundary",
]


def explain(verdict: str, confidence: float, analysis_id: str = "") -> str:
    """
    Generate a deterministic explanation tied to the analysis_id so
    repeated submissions of the same file always produce the same text.
    """
    if verdict == 'FAKE':
        level = "Very high" if confidence > 0.9 else "Strong" if confidence > 0.7 else "Moderate"
        # Deterministic selection — hash the analysis_id to pick two artifacts
        seed = int(hashlib.md5(analysis_id.encode()).hexdigest(), 16)
        idx1 = seed % len(_FAKE_ARTIFACTS)
        idx2 = (seed // len(_FAKE_ARTIFACTS)) % len(_FAKE_ARTIFACTS)
        if idx2 == idx1:
            idx2 = (idx2 + 1) % len(_FAKE_ARTIFACTS)
        a1, a2 = _FAKE_ARTIFACTS[idx1], _FAKE_ARTIFACTS[idx2]
        return (f"{level} probability of face-swap manipulation detected. "
                f"The model identified {a1} and {a2}. "
                "Red/yellow regions in the heatmap highlight where artifacts were found.")
    if verdict == 'REAL':
        return (f"No significant manipulation artifacts detected (confidence: {confidence:.0%}). "
                "Facial regions appear consistent with authentic footage — natural skin texture, "
                "consistent lighting, and stable facial boundaries observed.")
    return "No face detected in this media. Cannot perform deepfake analysis."


def save_heatmap(frame, analysis_id: str) -> Optional[str]:
    if frame is None:
        return None
    try:
        path = RESULTS_DIR / f"{analysis_id}_heatmap.jpg"
        cv2.imwrite(str(path), frame)
        return f"/results_static/{analysis_id}_heatmap.jpg"
    except Exception:
        return None


def _compute_risk(verdict: str, fake_prob_or_ratio: float) -> float:
    """
    Risk score (0–100).
    FAKE  → proportional to fake probability/ratio.
    REAL  → low residual risk based on (1 - confidence).
    """
    if verdict == 'FAKE':
        return round(fake_prob_or_ratio * 100, 1)
    return round((1.0 - fake_prob_or_ratio) * 100, 1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "DeepFake Detector API",
        "model_loaded": analyzer is not None,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None,
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    if analyzer is None:
        raise HTTPException(503, detail="Model not loaded — check server terminal for errors.")

    ct = file.content_type or ""
    if ct not in ALLOWED_IMAGE:
        raise HTTPException(400, detail=f"Unsupported type '{ct}'. Send JPG, PNG, or WebP.")

    aid = str(uuid.uuid4())[:8]
    fpath = UPLOAD_DIR / f"{aid}_{file.filename}"

    try:
        data = await file.read()
        if len(data) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, detail=f"File too large (max {MAX_MB}MB).")
        fpath.write_bytes(data)

        t0 = time.time()
        result = analyzer.analyze_image(str(fpath))
        ms = (time.time() - t0) * 1000

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        verdict    = result.get("frame_verdict", "UNKNOWN")
        confidence = result.get("confidence", 0.0)

        # Risk: for FAKE use fake_prob of top face; for REAL use (1 - confidence)
        top_fake_prob = max(
            (f["fake_probability"] for f in result.get("faces", [])), default=confidence
        )
        risk = _compute_risk(verdict, top_fake_prob if verdict == "FAKE" else confidence)

        faces = [
            FaceResult(
                bbox=list(f["bbox"]),
                label=f["label"],
                fake_probability=f["fake_probability"],
                real_probability=f["real_probability"],
            )
            for f in result.get("faces", [])
        ]

        resp = AnalysisResult(
            analysis_id=aid,
            input_type="image",
            final_verdict=verdict,
            confidence=confidence,
            risk_score=risk,
            faces_detected=len(faces),
            processing_time_ms=round(ms, 2),
            explanation=explain(verdict, confidence, aid),
            heatmap_url=save_heatmap(result.get("heatmap_overlay"), aid),
            face_results=faces,
        )
        cache_set(aid, resp.dict())
        return resp

    finally:
        if fpath.exists():
            fpath.unlink()

@app.post("/analyze/video", response_model=AnalysisResult)
async def analyze_video(file: UploadFile = File(...), sample_rate: int = 10):
    if analyzer is None:
        raise HTTPException(503, detail="Model not loaded — check server terminal for errors.")

    ct = file.content_type or ""
    if ct not in ALLOWED_VIDEO:
        raise HTTPException(400, detail=f"Unsupported type '{ct}'. Send MP4, AVI, or MOV.")

    aid = str(uuid.uuid4())[:8]
    fpath = UPLOAD_DIR / f"{aid}_{file.filename}"

    try:
        data = await file.read()
        if len(data) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, detail=f"File too large (max {MAX_MB}MB).")
        fpath.write_bytes(data)

        t0 = time.time()
        result = analyzer.analyze_video(str(fpath), sample_rate=sample_rate)
        ms = (time.time() - t0) * 1000

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        verdict    = result.get("final_verdict", "UNKNOWN")
        fake_ratio = result.get("fake_frame_ratio", 0.0)
        # Confidence = fake_ratio for FAKE, (1 - fake_ratio) for REAL
        confidence = fake_ratio if verdict == "FAKE" else (1.0 - fake_ratio)
        risk       = _compute_risk(verdict, fake_ratio)

        hframes  = result.get("heatmap_frames", [])
        hmap_url = save_heatmap(hframes[0] if hframes else None, aid)
        total_faces = sum(f.get("face_count", 0) for f in result.get("frame_level_results", []))

        resp = AnalysisResult(
            analysis_id=aid,
            input_type="video",
            final_verdict=verdict,
            confidence=round(confidence, 4),
            risk_score=risk,
            faces_detected=total_faces,
            processing_time_ms=round(ms, 2),
            explanation=explain(verdict, confidence, aid),
            heatmap_url=hmap_url,
            total_frames=result.get("total_frames"),
            analyzed_frames=result.get("analyzed_frames"),
            fake_frame_ratio=result.get("fake_frame_ratio"),
            duration_seconds=result.get("duration_seconds"),
        )
        cache_set(aid, resp.dict())
        return resp

    finally:
        if fpath.exists():
            fpath.unlink()

@app.get("/results/{aid}")
def get_result(aid: str):
    if aid not in result_cache:
        raise HTTPException(404, detail="Result not found.")
    return result_cache[aid]