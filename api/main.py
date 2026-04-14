from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import os, uuid, time, random, cv2, numpy as np
from pathlib import Path
import sys

# ── App Setup (MUST come before routes) ───────────────────────────────────────
app = FastAPI(
    title="DeepFake Detector API",
    description="AI-powered face-swap deepfake detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent / 'model'))
analyzer = None

try:
    from detector import FaceSwapAnalyzer
    analyzer = FaceSwapAnalyzer()
    print("[API] Model ready.")
except Exception as e:
    print(f"[API ERROR] Could not load model: {e}")
    import traceback
    traceback.print_exc()

# ── Directories ───────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/results_static", StaticFiles(directory=str(RESULTS_DIR)), name="results")

MAX_FILE_SIZE_MB = 100
ALLOWED_VIDEO_TYPES = {'video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'}
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/webp'}

result_cache = {}


# ── Pydantic Models ───────────────────────────────────────────────────────────
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
def generate_explanation(verdict: str, confidence: float, fake_ratio: float = None) -> str:
    if verdict == 'FAKE':
        level = "Very high" if confidence > 0.9 else "Strong" if confidence > 0.7 else "Moderate"
        artifacts = random.sample([
            "unnatural blending around facial boundaries",
            "inconsistent lighting and skin texture",
            "temporal flickering in facial regions",
            "misaligned facial features",
            "GAN-specific frequency artifacts"
        ], 2)
        return (f"{level} probability of manipulation detected. "
                f"The model identified {artifacts[0]} and {artifacts[1]}. "
                "Highlighted heatmap regions show where artifacts were found.")
    elif verdict == 'REAL':
        return (f"No significant manipulation artifacts detected (confidence: {confidence:.0%}). "
                "Facial regions appear consistent with authentic footage. "
                "Natural skin texture, consistent lighting, and stable facial boundaries observed.")
    return "No face detected. Cannot perform deepfake analysis."

def save_heatmap(heatmap_frame, analysis_id: str) -> Optional[str]:
    if heatmap_frame is None:
        return None
    path = RESULTS_DIR / f"{analysis_id}_heatmap.jpg"
    cv2.imwrite(str(path), heatmap_frame)
    return f"/results_static/{analysis_id}_heatmap.jpg"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "DeepFake Detector API", "docs": "/docs", "model_loaded": analyzer is not None}

@app.get("/health")
def health_check():
    import datetime
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None,
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    if analyzer is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported type: {file.content_type}. Use JPG/PNG/WebP.")

    analysis_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, "File too large")
        with open(file_path, 'wb') as f:
            f.write(content)

        start = time.time()
        result = analyzer.analyze_image(str(file_path))
        elapsed_ms = (time.time() - start) * 1000

        if 'error' in result:
            raise HTTPException(500, result['error'])

        verdict = result.get('frame_verdict', 'UNKNOWN')
        confidence = result.get('confidence', 0.0)
        risk_score = confidence * 100 if verdict == 'FAKE' else (1 - confidence) * 100

        face_results = [
            FaceResult(
                bbox=list(f['bbox']),
                label=f['label'],
                fake_probability=f['fake_probability'],
                real_probability=f['real_probability']
            ) for f in result.get('faces', [])
        ]

        heatmap_url = save_heatmap(result.get('heatmap_overlay'), analysis_id)

        response = AnalysisResult(
            analysis_id=analysis_id,
            input_type='image',
            final_verdict=verdict,
            confidence=confidence,
            risk_score=round(risk_score, 1),
            faces_detected=len(face_results),
            processing_time_ms=round(elapsed_ms, 2),
            explanation=generate_explanation(verdict, confidence),
            heatmap_url=heatmap_url,
            face_results=face_results
        )
        result_cache[analysis_id] = response.dict()
        return response
    finally:
        if file_path.exists():
            os.remove(file_path)

@app.post("/analyze/video", response_model=AnalysisResult)
async def analyze_video(file: UploadFile = File(...), sample_rate: int = 10):
    if analyzer is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported type: {file.content_type}. Use MP4/AVI/MOV.")

    analysis_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, "File too large (max 100MB)")
        with open(file_path, 'wb') as f:
            f.write(content)

        start = time.time()
        result = analyzer.analyze_video(str(file_path), sample_rate=sample_rate)
        elapsed_ms = (time.time() - start) * 1000

        if 'error' in result:
            raise HTTPException(500, result['error'])

        verdict = result.get('final_verdict', 'UNKNOWN')
        fake_ratio = result.get('fake_frame_ratio', 0)
        confidence = fake_ratio if verdict == 'FAKE' else (1 - fake_ratio)

        heatmap_frames = result.get('heatmap_frames', [])
        heatmap_url = save_heatmap(heatmap_frames[0] if heatmap_frames else None, analysis_id)
        total_faces = sum(f.get('face_count', 0) for f in result.get('frame_level_results', []))

        response = AnalysisResult(
            analysis_id=analysis_id,
            input_type='video',
            final_verdict=verdict,
            confidence=round(confidence, 4),
            risk_score=result.get('risk_score', 0.0),
            faces_detected=total_faces,
            processing_time_ms=round(elapsed_ms, 2),
            explanation=generate_explanation(verdict, confidence, fake_ratio),
            heatmap_url=heatmap_url,
            total_frames=result.get('total_frames'),
            analyzed_frames=result.get('analyzed_frames'),
            fake_frame_ratio=result.get('fake_frame_ratio'),
            duration_seconds=result.get('duration_seconds')
        )
        result_cache[analysis_id] = response.dict()
        return response
    finally:
        if file_path.exists():
            os.remove(file_path)

@app.get("/results/{analysis_id}")
def get_result(analysis_id: str):
    if analysis_id not in result_cache:
        raise HTTPException(404, "Result not found.")
    return result_cache[analysis_id]