"""
DeepFake Face-Swap Detector
Uses EfficientNet-B0 fine-tuned for deepfake detection with Grad-CAM explainability.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, Optional


class DeepFakeDetector(nn.Module):
    """
    EfficientNet-B0 based deepfake detector.
    Binary classifier: REAL (0) vs FAKE (1)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(DeepFakeDetector, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        # Hook for Grad-CAM
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Hook on last conv layer of EfficientNet
        target_layer = self.backbone.features[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GradCAM:
    """Generates Grad-CAM heatmaps for model explainability."""

    def __init__(self, model: DeepFakeDetector):
        self.model = model

    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            target_class: 0=REAL, 1=FAKE (default: highlight fake artifacts)
        Returns:
            heatmap as numpy array [H, W] normalized 0-1
        """
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.model.gradients  # [1, C, H, W]
        activations = self.model.activations  # [1, C, H, W]

        if gradients is None or activations is None:
            return np.zeros((224, 224))

        # Pool gradients over spatial dimensions
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


class FaceSwapAnalyzer:
    """
    Full pipeline: face detection → crop → classify → explain
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = DeepFakeDetector(num_classes=2, pretrained=True)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded weights from {model_path}")
        else:
            print("[WARN] No weights loaded. Using pretrained backbone (for demo/testing only).")

        self.model.to(self.device)
        self.gradcam = GradCAM(self.model)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

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
        """
        Analyze a single video frame.
        Returns detection results with confidence and heatmap.
        """
        results = {
            'faces': [],
            'frame_verdict': 'REAL',
            'confidence': 0.0,
            'heatmap_overlay': None
        }

        face_locations = self.detect_faces(frame)

        if not face_locations:
            results['frame_verdict'] = 'NO_FACE'
            return results

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        highest_fake_conf = 0.0
        composite_heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

        for (top, right, bottom, left) in face_locations:
            # Add padding around face
            h, w = frame.shape[:2]
            pad = 20
            top_p = max(0, top - pad)
            bottom_p = min(h, bottom + pad)
            left_p = max(0, left - pad)
            right_p = min(w, right + pad)

            face_crop = frame_rgb[top_p:bottom_p, left_p:right_p]
            face_pil = Image.fromarray(face_crop)
            input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

            # Forward pass
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
                real_prob = probs[0, 0].item()

            label = 'FAKE' if fake_prob > 0.5 else 'REAL'
            confidence = fake_prob if label == 'FAKE' else real_prob

            # Grad-CAM for FAKE detections
            cam = None
            if fake_prob > 0.3:
                input_tensor.requires_grad_(True)
                cam = self.gradcam.generate(input_tensor, target_class=1)
                cam_resized = cv2.resize(cam, (right_p - left_p, bottom_p - top_p))
                composite_heatmap[top_p:bottom_p, left_p:right_p] = np.maximum(
                    composite_heatmap[top_p:bottom_p, left_p:right_p],
                    cam_resized
                )

            results['faces'].append({
                'bbox': (top, right, bottom, left),
                'label': label,
                'fake_probability': round(fake_prob, 4),
                'real_probability': round(real_prob, 4),
                'cam': cam
            })

            if fake_prob > highest_fake_conf:
                highest_fake_conf = fake_prob

        results['frame_verdict'] = 'FAKE' if highest_fake_conf > 0.5 else 'REAL'
        results['confidence'] = round(highest_fake_conf if highest_fake_conf > 0.5 else (1 - highest_fake_conf), 4)
        results['heatmap_overlay'] = self._overlay_heatmap(frame, composite_heatmap)
        return results

    def _overlay_heatmap(self, frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Overlay Grad-CAM heatmap on frame."""
        if heatmap.max() == 0:
            return frame.copy()

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        return blended

    def analyze_video(self, video_path: str, sample_rate: int = 10) -> dict:
        """
        Analyze a video file for deepfake content.
        Args:
            video_path: Path to video file
            sample_rate: Analyze every N-th frame (default: 10)
        Returns:
            Aggregated results across all sampled frames
        """
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
        analyzed_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                result = self.analyze_frame(frame)
                frame_results.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps if fps > 0 else 0,
                    'verdict': result['frame_verdict'],
                    'confidence': result['confidence'],
                    'face_count': len(result['faces'])
                })

                if result['frame_verdict'] == 'FAKE':
                    fake_votes += 1
                elif result['frame_verdict'] == 'REAL':
                    real_votes += 1

                # Save first heatmap frame for output
                if result['frame_verdict'] == 'FAKE' and len(analyzed_frames) < 3:
                    analyzed_frames.append(result['heatmap_overlay'])

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
            'heatmap_frames': analyzed_frames,
            'risk_score': round(fake_ratio * 100, 1)
        }

    def analyze_image(self, image_path: str) -> dict:
        """Analyze a single image for deepfake face-swap."""
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': f'Cannot read image: {image_path}'}
        return self.analyze_frame(frame)