"""
Tests for DeepFake Detector API.
Run with: pytest tests/ -v

All tests use mocks — no real model download needed.
"""

import io
import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add api/ to path so we can import main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

# Mock the heavy model imports BEFORE importing main
# This prevents the test from trying to download the transformer model
sys.modules['detector'] = MagicMock()

# Mock FaceSwapAnalyzer so it doesn't load on import
mock_analyzer = MagicMock()

with patch.dict('sys.modules', {'detector': MagicMock()}):
    with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
               MagicMock() if name == 'detector' else __import__(name, *args, **kwargs)):
        pass

from fastapi.testclient import TestClient


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Create test client with analyzer mocked.
    We patch 'analyzer' in main so tests never touch the real model.
    """
    import main
    main.analyzer = MagicMock()
    return TestClient(main.app)


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_root_returns_service_info(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "DeepFake Detector API"


# ── Image analysis ────────────────────────────────────────────────────────────

def test_analyze_image_fake_result(client):
    """Analyzer returns FAKE — API should return correct structure."""
    import main
    main.analyzer.analyze_image.return_value = {
        "frame_verdict": "FAKE",
        "confidence": 0.93,
        "heatmap_overlay": None,
        "faces": [
            {
                "bbox": (100, 200, 300, 50),
                "label": "FAKE",
                "fake_probability": 0.93,
                "real_probability": 0.07,
            }
        ]
    }

    img_bytes = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
    response = client.post(
        "/analyze/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_verdict"] == "FAKE"
    assert data["confidence"] == 0.93
    assert data["input_type"] == "image"
    assert "analysis_id" in data
    assert "explanation" in data
    assert "risk_score" in data
    assert data["faces_detected"] == 1


def test_analyze_image_real_result(client):
    """Analyzer returns REAL — verify verdict and risk_score are correct."""
    import main
    main.analyzer.analyze_image.return_value = {
        "frame_verdict": "REAL",
        "confidence": 0.88,
        "heatmap_overlay": None,
        "faces": [
            {
                "bbox": (100, 200, 300, 50),
                "label": "REAL",
                "fake_probability": 0.12,
                "real_probability": 0.88,
            }
        ]
    }

    img_bytes = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
    response = client.post(
        "/analyze/image",
        files={"file": ("real.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_verdict"] == "REAL"
    assert data["risk_score"] < 50   # REAL should have low risk score


def test_analyze_image_wrong_type(client):
    """Non-image file should return 400."""
    response = client.post(
        "/analyze/image",
        files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")}
    )
    assert response.status_code == 400


def test_analyze_image_no_file(client):
    """Missing file should return 422 (FastAPI validation)."""
    response = client.post("/analyze/image")
    assert response.status_code == 422


# ── Video analysis ────────────────────────────────────────────────────────────

def test_analyze_video_fake_result(client):
    """Video analysis returning FAKE verdict."""
    import main
    main.analyzer.analyze_video.return_value = {
        "final_verdict": "FAKE",
        "fake_frame_ratio": 0.65,
        "total_frames": 300,
        "analyzed_frames": 30,
        "duration_seconds": 10.0,
        "frame_level_results": [{"face_count": 1}],
        "heatmap_frames": [],
        "risk_score": 65.0
    }

    video_bytes = io.BytesIO(b"\x00" * 500)
    response = client.post(
        "/analyze/video",
        files={"file": ("test.mp4", video_bytes, "video/mp4")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_verdict"] == "FAKE"
    assert data["input_type"] == "video"
    assert data["total_frames"] == 300
    assert data["fake_frame_ratio"] == 0.65


def test_analyze_video_wrong_type(client):
    """Non-video file should return 400."""
    response = client.post(
        "/analyze/video",
        files={"file": ("img.jpg", b"fake jpg", "image/jpeg")}
    )
    assert response.status_code == 400


# ── Result cache ──────────────────────────────────────────────────────────────

def test_get_result_from_cache(client):
    """After analyzing, result should be retrievable by ID."""
    import main
    main.analyzer.analyze_image.return_value = {
        "frame_verdict": "REAL",
        "confidence": 0.80,
        "heatmap_overlay": None,
        "faces": []
    }

    img_bytes = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
    post_response = client.post(
        "/analyze/image",
        files={"file": ("cached.jpg", img_bytes, "image/jpeg")}
    )
    assert post_response.status_code == 200
    aid = post_response.json()["analysis_id"]

    # Now retrieve by ID
    get_response = client.get(f"/results/{aid}")
    assert get_response.status_code == 200
    assert get_response.json()["analysis_id"] == aid


def test_get_nonexistent_result(client):
    """Unknown analysis ID should return 404."""
    response = client.get("/results/nonexistent123")
    assert response.status_code == 404


# ── Explanation logic ─────────────────────────────────────────────────────────

def test_explanation_for_fake():
    """FAKE explanation should mention manipulation."""
    import main
    result = main.explain("FAKE", 0.95, "abc12345")
    assert "manipulation" in result.lower() or "fake" in result.lower() or "artifact" in result.lower()


def test_explanation_for_real():
    """REAL explanation should mention no artifacts."""
    import main
    result = main.explain("REAL", 0.88, "abc12345")
    assert "no" in result.lower() or "authentic" in result.lower() or "natural" in result.lower()


def test_explanation_deterministic():
    """Same analysis_id should always produce same explanation."""
    import main
    r1 = main.explain("FAKE", 0.90, "fixed_id_123")
    r2 = main.explain("FAKE", 0.90, "fixed_id_123")
    assert r1 == r2