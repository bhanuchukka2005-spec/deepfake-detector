# At top of detector.py, replace the model class with this:

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import numpy as np
import cv2

class FaceSwapAnalyzer:
    def __init__(self, model_path=None, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print("[INFO] Loading pretrained deepfake detector...")
        self.processor = AutoImageProcessor.from_pretrained(
            "prithivMLmods/Deep-Fake-Detector-Model"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "prithivMLmods/Deep-Fake-Detector-Model"
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Model ready on {self.device}")

    def detect_faces(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        locations = []
        for (x, y, w, h) in faces:
            locations.append((y, x + w, y + h, x))
        return locations

    def analyze_frame(self, frame: np.ndarray) -> dict:
        results = {
            'faces': [],
            'frame_verdict': 'REAL',
            'confidence': 0.0,
            'heatmap_overlay': frame.copy()
        }

        face_locations = self.detect_faces(frame)

        if not face_locations:
            results['frame_verdict'] = 'NO_FACE'
            return results

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        highest_fake_conf = 0.0

        for (top, right, bottom, left) in face_locations:
            h, w = frame.shape[:2]
            pad = 20
            face_crop = frame_rgb[
                max(0, top-pad):min(h, bottom+pad),
                max(0, left-pad):min(w, right+pad)
            ]
            if face_crop.size == 0:
                continue

            pil_img = Image.fromarray(face_crop)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]

            # Model labels: check which index is Fake vs Real
            id2label = self.model.config.id2label
            fake_idx = next(
                (i for i, l in id2label.items() if 'fake' in l.lower() or 'ai' in l.lower()),
                1
            )
            real_idx = 1 - fake_idx if len(id2label) == 2 else 0

            fake_prob = probs[fake_idx].item()
            real_prob = probs[real_idx].item()
            label = 'FAKE' if fake_prob > 0.5 else 'REAL'

            results['faces'].append({
                'bbox': (top, right, bottom, left),
                'label': label,
                'fake_probability': round(fake_prob, 4),
                'real_probability': round(real_prob, 4),
                'cam': None
            })

            if fake_prob > highest_fake_conf:
                highest_fake_conf = fake_prob

        results['frame_verdict'] = 'FAKE' if highest_fake_conf > 0.5 else 'REAL'
        results['confidence'] = round(
            highest_fake_conf if highest_fake_conf > 0.5 else (1 - highest_fake_conf), 4
        )
        return results

    def analyze_image(self, image_path: str) -> dict:
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': f'Cannot read image: {image_path}'}
        return self.analyze_frame(frame)

    def analyze_video(self, video_path: str, sample_rate: int = 10) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        frame_results = []
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
            'heatmap_frames': [],
            'risk_score': round(fake_ratio * 100, 1)
        }