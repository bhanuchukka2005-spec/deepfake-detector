"""
Training Script for DeepFake Detector
Dataset: FaceForensics++ (c23 compression)
https://github.com/ondyari/FaceForensics

Usage:
  python train.py --data_dir ./data --epochs 20 --batch_size 32

NOTE: Face extraction uses OpenCV Haar Cascade (no dlib/cmake required).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
import os
import json
import random
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Trainable model wrapper (nn.Module)
# ---------------------------------------------------------------------------

class DeepFakeClassifier(nn.Module):
    """
    EfficientNet-B0 backbone fine-tuned for binary deepfake classification.
    Kept separate from FaceSwapAnalyzer (inference-only) in detector.py.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ Dataset Loader.

    Expected directory structure:
    data/
      real/
        video1/
          frame_001.png
          ...
      fake/
        DeepFakes/
          video1/
            frame_001.png
            ...
        FaceSwap/
          ...
        Face2Face/
          ...
    """

    def __init__(self, data_dir: str, split: str = 'train',
                 max_frames_per_video: int = 30, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == 'train')
        self.samples = []

        self._build_sample_list(max_frames_per_video)
        self._setup_transforms()

        print(f"[Dataset] {split}: {len(self.samples)} samples "
              f"({sum(1 for _, l in self.samples if l == 0)} real, "
              f"{sum(1 for _, l in self.samples if l == 1)} fake)")

    def _build_sample_list(self, max_frames: int):
        """Build list of (image_path, label) tuples."""
        real_dir = self.data_dir / 'real'
        fake_dir = self.data_dir / 'fake'

        # Real frames
        if real_dir.exists():
            for video_dir in real_dir.iterdir():
                if video_dir.is_dir():
                    frames = list(video_dir.glob('*.png')) + list(video_dir.glob('*.jpg'))
                    frames = random.sample(frames, min(max_frames, len(frames)))
                    self.samples.extend([(str(f), 0) for f in frames])

        # Fake frames (multiple manipulation methods)
        if fake_dir.exists():
            for method_dir in fake_dir.iterdir():
                if method_dir.is_dir():
                    for video_dir in method_dir.iterdir():
                        if video_dir.is_dir():
                            frames = list(video_dir.glob('*.png')) + list(video_dir.glob('*.jpg'))
                            frames = random.sample(frames, min(max_frames, len(frames)))
                            self.samples.extend([(str(f), 1) for f in frames])

        # Shuffle
        random.shuffle(self.samples)

        # Split
        n = len(self.samples)
        if self.split == 'train':
            self.samples = self.samples[:int(0.8 * n)]
        elif self.split == 'val':
            self.samples = self.samples[int(0.8 * n):int(0.9 * n)]
        else:  # test
            self.samples = self.samples[int(0.9 * n):]

    def _setup_transforms(self):
        """Setup data augmentation and normalization transforms."""
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
            return tensor, label
        except Exception:
            tensor = torch.zeros(3, 224, 224)
            return tensor, label

    def get_class_weights(self):
        """Compute class weights for handling class imbalance."""
        labels = [l for _, l in self.samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        total = n_real + n_fake
        return [total / (2 * n_real), total / (2 * n_fake)]


# ---------------------------------------------------------------------------
# Frame extraction utility (OpenCV Haar — no dlib/cmake required)
# ---------------------------------------------------------------------------

def extract_frames_from_video(video_path: str, output_dir: str,
                               max_frames: int = 50, face_only: bool = True):
    """
    Utility: Extract frames from video, optionally cropping to face region.
    Uses OpenCV Haar Cascade for face detection (no dlib or cmake required).
    Use this to preprocess FaceForensics++ videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total // max_frames)

    frame_idx = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            if face_only:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                # Relax if nothing detected
                if len(faces) == 0:
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
                    )
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    pad = 30
                    fh, fw = frame.shape[:2]
                    crop = frame[max(0, y - pad):min(fh, y + h + pad),
                                 max(0, x - pad):min(fw, x + w + pad)]
                    if crop.size > 0:
                        cv2.imwrite(f"{output_dir}/frame_{saved:04d}.png", crop)
                        saved += 1
            else:
                cv2.imwrite(f"{output_dir}/frame_{saved:04d}.png", frame)
                saved += 1

        frame_idx += 1

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model: DeepFakeClassifier, device: torch.device,
                 save_dir: str = './checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 20, lr: float = 1e-4):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0

        for epoch in range(epochs):
            # ---- TRAIN ----
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            # ---- VALIDATE ----
            val_loss, val_acc, val_auc = self.evaluate(val_loader, criterion)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)

            print(f"[Epoch {epoch+1}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                self.save_checkpoint(epoch, val_auc, 'best_model.pth')
                print(f"  ✓ New best model saved (AUC: {best_auc:.4f})")

        self.save_history()
        return self.history

    def evaluate(self, loader: DataLoader, criterion: nn.Module):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return avg_loss, acc, auc

    def save_checkpoint(self, epoch: int, metric: float, filename: str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metric': metric,
        }, self.save_dir / filename)

    def save_history(self):
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train DeepFake Detector')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    train_dataset = FaceForensicsDataset(args.data_dir, split='train', augment=True)
    val_dataset = FaceForensicsDataset(args.data_dir, split='val', augment=False)

    # Weighted sampler for class balance
    class_weights = train_dataset.get_class_weights()
    sample_weights = [class_weights[l] for _, l in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = DeepFakeClassifier(num_classes=2, pretrained=True)
    model.to(device)
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, device, save_dir=args.save_dir)
    history = trainer.train(train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    print("\n[Training Complete]")
    print(f"Best Val AUC: {max(history['val_auc']):.4f}")