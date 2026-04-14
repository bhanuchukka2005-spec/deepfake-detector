"""
DeepFake Face-Swap Detector
Uses a pretrained transformer model from HuggingFace for accurate detection.
Face detection via OpenCV Haar Cascade (no dlib/cmake required).
Probability heatmap overlaid on face regions weighted by fake confidence score.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Optional


class FaceSwapAnalyzer:
    """
    Full pipeline: face detection -> crop -> classify -> explain
    Uses: prithivMLmods/Deep-Fake-Detector-Model from HuggingFace
    """

    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[INFO] Loading deepfake detector on {self.device}...")

        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(
            "prithivMLmods/Deep-Fake-Detector-Model"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "prithivMLmods/Deep-Fake-Detector-Model"
        )
        self.model.to(self.device)
        self.model.eval()

        # Identify which label index means FAKE
        id2label = self.model.config.id2label
        print(f"[INFO] Model labels: {id2label}")
        self.fake_idx = next(
            (i for i, l in id2label.items()
             if any(w in l.lower() for w in ['fake', 'ai', 'artificial', 'generated', 'deepfake'])),
            1
        )
        self.real_idx = next(
            (i for i, l in id2label.items()
             if any(w in l.lower() for w in ['real', 'authentic', 'genuine', 'human'])),
            0
        )
        print(f"[INFO] FAKE index={self.fake_idx}, REAL index={self.real_idx}")
        print("[INFO] Model ready.")

        # Face detector — OpenCV Haar (no dlib needed)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect face bounding boxes using OpenCV Haar Cascade.
        Returns list of (top, right, bottom, left) tuples.
        Falls back to full-frame analysis if no face detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try standard detection first
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Relax constraints if nothing found (profile faces, partial occlusion)
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(40, 40)
            )

        locations = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Convert (x,y,w,h) -> (top, right, bottom, left)
                locations.append((y, x + w, y + h, x))

        return locations

    def classify_face(self, face_crop_rgb: np.ndarray) -> dict:
        """Run the transformer model on a cropped face region."""
        pil_img = Image.fromarray(face_crop_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        fake_prob = probs[self.fake_idx].item()
        real_prob = probs[self.real_idx].item()

        # Normalize in case of >2 classes
        total = fake_prob + real_prob
        if total > 0:
            fake_prob = fake_prob / total
            real_prob = real_prob / total

        label = 'FAKE' if fake_prob > 0.5 else 'REAL'
        return {
            'label': label,
            'fake_probability': round(fake_prob, 4),
            'real_probability': round(real_prob, 4),
        }

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Analyze a single frame. Returns verdict, confidence, faces list,
        and a probability heatmap overlay highlighting regions by fake score.
        If no face is detected, falls back to full-frame classification.
        """
        results = {
            'faces': [],
            'frame_verdict': 'REAL',
            'confidence': 0.0,
            'heatmap_overlay': frame.copy()
        }

        face_locations = self.detect_faces(frame)

        # Fallback: run classifier on the full frame when no face is found
        if not face_locations:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            classification = self.classify_face(frame_rgb)
            results['frame_verdict'] = classification['label']
            results['confidence'] = (
                classification['fake_probability']
                if classification['label'] == 'FAKE'
                else classification['real_probability']
            )
            results['faces'] = [{
                'bbox': (0, frame.shape[1], frame.shape[0], 0),
                'label': classification['label'],
                'fake_probability': classification['fake_probability'],
                'real_probability': classification['real_probability'],
                'full_frame_fallback': True,
            }]
            return results

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        highest_fake_conf = 0.0
        heatmap = np.zeros((h, w), dtype=np.float32)

        for (top, right, bottom, left) in face_locations:
            pad = 20
            t = max(0, top - pad)
            b = min(h, bottom + pad)
            l = max(0, left - pad)
            r = min(w, right + pad)

            face_crop = frame_rgb[t:b, l:r]
            if face_crop.size == 0:
                continue

            classification = self.classify_face(face_crop)
            fake_prob = classification['fake_probability']

            # Build probability heatmap over face region weighted by fake score
            heatmap[t:b, l:r] = fake_prob

            results['faces'].append({
                'bbox': (top, right, bottom, left),
                'label': classification['label'],
                'fake_probability': classification['fake_probability'],
                'real_probability': classification['real_probability'],
            })

            if fake_prob > highest_fake_conf:
                highest_fake_conf = fake_prob

        results['frame_verdict'] = 'FAKE' if highest_fake_conf > 0.5 else 'REAL'
        results['confidence'] = round(
            highest_fake_conf if highest_fake_conf > 0.5 else (1 - highest_fake_conf), 4
        )

        # Overlay probability heatmap on frame (JET colormap: blue=low, red=high fake prob)
        if heatmap.max() > 0:
            heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            results['heatmap_overlay'] = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        return results

    def analyze_image(self, image_path: str) -> dict:
        """Analyze a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': f'Cannot read image: {image_path}'}
        return self.analyze_frame(frame)

    def analyze_video(self, video_path: str, sample_rate: int = 10) -> dict:
        """
        Analyze a video file for deepfake content.
        Samples every Nth frame, aggregates results by voting.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        frame_results = []
        heatmap_frames = []
        fake_votes = 0
        real_votes = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                result = self.analyze_frame(frame)

                frame_results.append({
                    'frame_idx': frame_idx,
                    'timestamp': round(frame_idx / fps, 2) if fps > 0 else 0,
                    'verdict': result['frame_verdict'],
                    'confidence': result['confidence'],
                    'face_count': len(result['faces'])
                })

                if result['frame_verdict'] == 'FAKE':
                    fake_votes += 1
                    if len(heatmap_frames) < 3:
                        heatmap_frames.append(result['heatmap_overlay'])
                elif result['frame_verdict'] == 'REAL':
                    real_votes += 1

            frame_idx += 1

        cap.release()

        total_votes = fake_votes + real_votes
        fake_ratio = fake_votes / total_votes if total_votes > 0 else 0

        return {
            'video_path': video_path,
            'total_frames': total_frames,
            'analyzed_frames': len(frame_results),
            'duration_seconds': round(duration, 2),
            'fps': round(fps, 2),
            'final_verdict': 'FAKE' if fake_ratio > 0.3 else 'REAL',
            'fake_frame_ratio': round(fake_ratio, 4),
            'fake_votes': fake_votes,
            'real_votes': real_votes,
            'frame_level_results': frame_results,
            'heatmap_frames': heatmap_frames,
            'risk_score': round(fake_ratio * 100, 1)
        }