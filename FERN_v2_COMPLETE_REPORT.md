# FERN v2 — Complete Project Report for AI Fixer

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Pipeline & File Structure](#2-data-pipeline--file-structure)
3. [Full Source Code](#3-full-source-code)
4. [All Experiments & Results](#4-all-experiments--results)
5. [Known Issues & What Needs Fixing](#5-known-issues--what-needs-fixing)
6. [Environment](#6-environment)
7. [Dataset Details](#7-dataset-details)
8. [Model Architecture Details](#8-model-architecture-details)

---

## 1. Project Overview

**Goal**: Classify 8 foot gestures from lower-body 10-joint MediaPipe skeleton sequences in real time.

**Pipeline**: Camera video → MediaPipe Pose (lower body 10 joints) → sliding window (60 frames, stride 15) → CNN-only classifier → 1 of 8 gesture classes.

The project has gone through many iterations:
- Started with BiLSTM (failed, 25% accuracy — overfit)
- Switched to CNN-only (success, 80% on single split / ~61% unbiased CV)
- Tried geometric skeleton rotation to normalize camera angles (FAILED — z too noisy)
- Implemented one-hot camera-ID flag to handle multiple cameras (Phase 1: c3 front + c2 side = +3.7pp gain)
- c4 camera (45° right) excluded — all files <30% detection rate

**Current best model**: 140K param CNN-only with camera-ID flag (c3=[1,0], c2=[0,1]). Front-only: 62.58%, Phase 1: 50.48% combined (post-A1 detection-ratio filter fix).

---

## 2. Data Pipeline & File Structure

### Directory Layout

```
C:\fern\FERN_V2\
├── src\                         # All Python source code
│   ├── model_v2.py              # CNN-only / BiLSTM model
│   ├── dataset_v2.py            # Sliding window PyTorch Dataset
│   ├── train_v2.py              # Training loop (CLI, resume, n_cameras)
│   ├── export_onnx.py           # .pth → .onnx export
│   ├── test_onnx.py             # ONNX full-dataset evaluation
│   ├── infer_v2.py              # Live camera inference
│   ├── kfold_cv.py              # K-fold cross-validation
│   ├── add_foot_hold_gaps.py    # Insert foot_hold gaps at gesture transitions
│   ├── mirror_10joint.py        # X-flip mirror augmentation
│   └── add_camera_id.py         # Add camera_id field to label JSONs
├── data\
│   ├── merged_v1\               # Raw source videos (50 MP4s)
│   ├── skeletons\
│   │   ├── front\               # 76 CSVs (38 orig + 38 mirror), c3 only, z=0
│   │   ├── front_plus_45\       # 98 CSVs (76 c3 + 22 c2), z=0, camera_id in JSONs
│   │   └── merged_v1\           # All extracted skeletons (including c4 files)
│   └── labels\
│       ├── front\               # 76 JSONs with camera_id=0, foot_hold gaps
│       ├── front_plus_45\       # 98 JSONs (c3=0, c2=1)
│       └── merged_v1\           # Label JSONs (some missing camera_id)
├── models_final\                # Front-only model (132K params, 30-dim input)
│   ├── fern_v2_latest.pth
│   └── fern_v2.onnx
├── models_final_v2\             # Phase 1 camera-flag model (140K params, 32-dim input)
│   ├── fern_v2_latest.pth
│   └── fern_v2.onnx
├── report.md                    # Detailed experiment report (14 sections)
├── AGENTS.md                    # Quick-reference project knowledge
├── CAMERA_FLAG_AGENT.md         # Camera-ID flag experiment plan/results
└── docs\                        # Notes on failed experiments
    └── FERN_TRANSFORM_AGENT.md  # Geometric transform failure analysis
```

### Gesture Classes (8)

| Index | Name | Description |
|-------|------|-------------|
| 0 | foot_hold | Idle/neutral stance |
| 1 | foot_lift | Lift foot vertically |
| 2 | sideway_kick | Kick to the side |
| 3 | cross_front | Cross one foot in front of the other |
| 4 | heel_tap | Tap heel on ground |
| 5 | flamingo_bend | Stand on one leg, bend the other |
| 6 | forward_step | Step forward |
| 7 | forward_kick | Kick forward |

### Skeleton Format

- MediaPipe Pose → 10 lower-body joints (both hips, knees, ankles, heels, foot indices)
- Each frame: 30 floats (10 joints × 3 coords: x, y, z)
- z is always 0 (zeroed out to eliminate MediaPipe/YOLO depth noise)
- CSV columns: `frame_idx, pose_detected, left_hip_x, left_hip_y, left_hip_z, ..., mid_hip_x, mid_hip_y, mid_hip_z`
- 36 columns total (frame_idx, pose_detected, 30 feature cols, 3 extra)
- Undetected frames (pose_detected=0) have NaN → linear interpolated along time axis

### Camera IDs & Naming Convention

| Camera | Angle | ID | Files | Status |
|--------|-------|----|-------|--------|
| c3 | 0° (front) | 0 | 76 (38 orig + 38 mirror) | Included |
| c2 | ~90° (left side) | 1 | 22 (11 orig + 11 mirror) | Included |
| c4 | ~45° (right side) | (2) | 22 | Excluded — all <30% detection |

Files are named `p{subject}_{camera}.csv` (e.g., `p05_c3.csv`, `p00_c4_mirror.csv`). Label JSONs follow the same stem.

### Camera-ID One-Hot Encoding

When `n_cameras > 1`, a per-frame one-hot flag is appended to each window:
```
c3=[1,0]  c2=[0,1]   → appended as 2 columns → input (T, 32)
```

Mirror files inherit the same camera_id as their original.

---

## 3. Full Source Code

### 3.1 `src/model_v2.py` — The Model

```python
"""
FERN v2 — Foot gesture recognition from skeleton sequences.

Architecture: SpatialCNN -> BiLSTM -> Attention -> Classifier

Input shape:  (batch, seq_len, num_features)
              where num_features = num_joints * 3
Output shape: (batch, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial CNN block
# ---------------------------------------------------------------------------

class SpatialCNNBlock(nn.Module):
    """
    Applies 1D convolutions across the feature dimension of each frame.

    Each frame is a vector of joint coordinates.  Conv1d discovers patterns like
    "left ankle is far from right heel".
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_features // 2,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_features // 2,
            out_channels=out_features,
            kernel_size=3,
            padding=1
        )
        self.bn1  = nn.BatchNorm1d(out_features // 2)
        self.bn2  = nn.BatchNorm1d(out_features)
        self.proj = nn.Linear(in_features * out_features, out_features)
        self.drop = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch * seq_len, in_features)
        b = x.size(0)
        x = x.unsqueeze(1)                       # (B*T, 1, in_features)
        x = F.relu(self.bn1(self.conv1(x)))      # (B*T, out//2, in_features)
        x = F.relu(self.bn2(self.conv2(x)))      # (B*T, out, in_features)
        x = x.view(b, -1)                        # flatten
        x = self.drop(F.relu(self.proj(x)))      # (B*T, out_features)
        return x


# ---------------------------------------------------------------------------
# Additive (Bahdanau-style) Attention
# ---------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """
    Additive attention over the time axis of a BiLSTM output.
    Used only when lstm_hidden > 0 (BiLSTM mode).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score_fc = nn.Linear(hidden_dim, hidden_dim)
        self.query    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.score_fc(lstm_out))   # (B, T, H)
        scores = self.query(scores).squeeze(-1)        # (B, T)
        weights = F.softmax(scores, dim=1)             # (B, T)
        context = torch.bmm(
            weights.unsqueeze(1),                      # (B, 1, T)
            lstm_out                                   # (B, T, H)
        ).squeeze(1)                                   # (B, H)
        return context, weights


# ---------------------------------------------------------------------------
# Full FERN v2 model
# ---------------------------------------------------------------------------

class FERNv2(nn.Module):
    def __init__(
        self,
        num_joints:   int = 10,
        num_classes:  int = 8,
        cnn_out:      int = 32,
        lstm_hidden:  int = 64,
        lstm_layers:  int = 2,
        dropout:      float = 0.5,
        input_features: int = None,
    ):
        super().__init__()

        in_features = input_features if input_features is not None else num_joints * 3
        self.use_lstm = lstm_hidden > 0

        # --- Spatial CNN ---
        self.spatial_cnn = SpatialCNNBlock(
            in_features=in_features,
            out_features=cnn_out,
            dropout=dropout,
        )

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=cnn_out,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            lstm_out_dim = lstm_hidden * 2
            self.attention = AdditiveAttention(lstm_out_dim)
            classifier_in = lstm_out_dim
        else:
            self.lstm = nn.Identity()
            self.attention = nn.Identity()
            classifier_in = cnn_out

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_in, classifier_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_in // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)

    def forward(self, x: torch.Tensor):
        B, T, F = x.shape
        x_flat = x.view(B * T, F)
        x_cnn  = self.spatial_cnn(x_flat)
        x_seq  = x_cnn.view(B, T, -1)

        if self.use_lstm:
            lstm_out, _ = self.lstm(x_seq)
            context, _ = self.attention(lstm_out)
        else:
            context = x_seq.mean(dim=1)

        logits = self.classifier(context)
        return logits

    def predict_with_attention(self, x: torch.Tensor):
        B, T, F = x.shape
        x_flat  = x.view(B * T, F)
        x_cnn   = self.spatial_cnn(x_flat)
        x_seq   = x_cnn.view(B, T, -1)

        if self.use_lstm:
            lstm_out, _ = self.lstm(x_seq)
            context, attn_weights = self.attention(lstm_out)
        else:
            context = x_seq.mean(dim=1)
            attn_weights = None

        logits = self.classifier(context)
        return logits, attn_weights


if __name__ == '__main__':
    model = FERNv2(num_joints=10, num_classes=8)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    dummy = torch.randn(4, 60, 30)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")
```

### 3.2 `src/dataset_v2.py` — The Dataset

```python
"""
FERN v2 — PyTorch Dataset from skeleton CSVs + label JSON files.

Sliding window dataset:
  - Load skeleton CSV
  - Load label JSON
  - For each labeled segment, slide a fixed-size window with configurable stride
  - Each window becomes one training sample

Missing frames (NaN) are linearly interpolated from neighbors.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

NUM_FEATURE_JOINTS = 10
NUM_FEATURES       = NUM_FEATURE_JOINTS * 3   # 30

FEATURE_COLS = []
JOINT_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
for name in JOINT_NAMES:
    for axis in ("x", "y", "z"):
        FEATURE_COLS.append(f"{name}_{axis}")


def load_skeleton_csv(csv_path: str) -> np.ndarray:
    """
    Load skeleton CSV → (T, 30) float32.
    pose_detected==0 rows → NaN → linear interpolated → fillna(0).
    """
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV {csv_path} missing columns: {missing_cols}")
    data = df[FEATURE_COLS].values.astype(np.float32)
    if "pose_detected" in df.columns:
        not_detected = df["pose_detected"].values == 0
        data[not_detected] = np.nan
    data_df = pd.DataFrame(data, columns=FEATURE_COLS)
    data_df.interpolate(method="linear", axis=0, inplace=True)
    data_df.fillna(0.0, inplace=True)
    return data_df.values.astype(np.float32)


def load_label_json(json_path: str) -> Tuple[List[dict], float]:
    with open(json_path) as f:
        data = json.load(f)
    return data["segments"], data.get("fps", 30.0)


DEFAULT_CLASSES = [
    "foot_hold",
    "foot_lift",
    "sideway_kick",
    "cross_front",
    "heel_tap",
    "flamingo_bend",
    "forward_step",
    "forward_kick",
]

def build_class_map(classes: List[str]):
    return {name: idx for idx, name in enumerate(classes)}


class SkeletonWindowDataset(Dataset):
    """
    Sliding window dataset.

    Parameters
    ----------
    skeleton_dir : str
    label_dir : str
    classes : list of str — ordered class names
    window_size : int — default 60 (2s at 30fps)
    stride : int — default 15 (0.5s)
    split : "train", "val", "test", or "all"
    split_mode : "random" or "subject" — video-level split
    val_ratio, test_ratio : float
    seed : int
    augment : bool — apply time warp, joint noise, joint dropout
    n_cameras : int — if >1, append one-hot camera-ID flag
    """

    def __init__(
        self,
        skeleton_dir: str,
        label_dir:    str,
        classes:      List[str] = None,
        window_size:  int  = 60,
        stride:       int  = 15,
        split:        str  = "all",
        split_mode:   str  = "random",
        val_ratio:    float = 0.15,
        test_ratio:   float = 0.10,
        seed:         int  = 42,
        augment:      bool = False,
        n_cameras:    int  = 1,
    ):
        self.skeleton_dir = skeleton_dir
        self.label_dir    = label_dir
        self.classes      = classes or DEFAULT_CLASSES
        self.class_map    = build_class_map(self.classes)
        self.num_classes  = len(self.classes)
        self.window_size  = window_size
        self.stride       = stride
        self.split        = split
        self.split_mode   = split_mode
        self.augment      = augment
        self.n_cameras    = n_cameras

        self.samples: List[Tuple[np.ndarray, int]] = []
        self.cam_ids: List[int] = []
        self._build(seed, val_ratio, test_ratio)

    def _all_pairs(self):
        """Find all (skeleton_csv, label_json) pairs."""
        pairs = []
        for root, _, files in os.walk(self.skeleton_dir):
            for fname in sorted(files):
                if not fname.endswith(".csv"):
                    continue
                csv_path  = os.path.join(root, fname)
                rel       = os.path.relpath(csv_path, self.skeleton_dir)
                json_path = os.path.join(
                    self.label_dir,
                    str(Path(rel).with_suffix(".json"))
                )
                if os.path.exists(json_path):
                    pairs.append((csv_path, json_path))
        return pairs

    def _split_pairs(self, pairs, seed, val_ratio, test_ratio):
        if self.split_mode == "random":
            return self._random_split_pairs(pairs, seed, val_ratio, test_ratio)
        return self._subject_split_pairs(pairs, seed, val_ratio, test_ratio)

    def _random_split_pairs(self, pairs, seed, val_ratio, test_ratio):
        """Random split: mirrors stay paired with originals."""
        rng = np.random.RandomState(seed)
        groups = {}
        for csv_path, json_path in pairs:
            stem = Path(csv_path).stem
            base = stem.replace("_mirror", "")
            groups.setdefault(base, []).append((csv_path, json_path))
        keys = sorted(groups.keys())
        rng.shuffle(keys)
        n      = len(keys)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        splits = {"train": [], "val": [], "test": [], "all": []}
        for i, base in enumerate(keys):
            for pair in groups[base]:
                splits["all"].append(pair)
            if i < n_test:
                splits["test"].extend(groups[base])
            elif i < n_test + n_val:
                splits["val"].extend(groups[base])
            else:
                splits["train"].extend(groups[base])
        return splits

    def _subject_split_pairs(self, pairs, seed, val_ratio, test_ratio):
        """Subject-independent split: all angles of same subject stay together."""
        rng = np.random.RandomState(seed)
        subject_map = {}
        cross_subject = []
        for csv_path, json_path in pairs:
            stem = Path(csv_path).stem
            base = stem.replace("_mirror", "")
            m = re.match(r"^(p\d{2})_c\d", base)
            if m:
                subject_map.setdefault(m.group(1), []).append((csv_path, json_path))
            else:
                cross_subject.append((csv_path, json_path))
        subjects = sorted(subject_map.keys())
        rng.shuffle(subjects)
        n      = len(subjects)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        train = list(cross_subject)
        val   = []
        test  = []
        for s in subjects[:n_test]:
            test.extend(subject_map[s])
        for s in subjects[n_test:n_test + n_val]:
            val.extend(subject_map[s])
        for s in subjects[n_test + n_val:]:
            train.extend(subject_map[s])
        shuffled = list(cross_subject)
        for s in subjects:
            shuffled.extend(subject_map[s])
        return {"train": train, "val": val, "test": test, "all": shuffled}

    def _build(self, seed, val_ratio, test_ratio):
        all_pairs = self._all_pairs()
        if not all_pairs:
            raise RuntimeError(f"No matching (CSV, JSON) pairs found.")
        split_map = self._split_pairs(all_pairs, seed, val_ratio, test_ratio)
        selected  = split_map[self.split]
        skipped_windows  = 0
        unknown_gestures = set()

        for csv_path, json_path in selected:
            try:
                skeleton  = load_skeleton_csv(csv_path)
                with open(json_path) as fh:
                    label_data = json.load(fh)
                segments  = label_data["segments"]
            except Exception as e:
                print(f"  WARNING: Skipping {csv_path}: {e}")
                continue

            cam_id = int(label_data.get("camera_id", 0))
            T = len(skeleton)

            for seg in segments:
                gesture = seg["gesture"]
                if gesture not in self.class_map:
                    unknown_gestures.add(gesture)
                    continue
                label = self.class_map[gesture]
                s     = seg["start_frame"]
                e     = min(seg["end_frame"], T - 1)

                pos = s
                while pos + self.window_size <= e + 1:
                    window = skeleton[pos: pos + self.window_size]
                    nan_ratio = np.isnan(window).mean()
                    if nan_ratio > 0.3:
                        skipped_windows += 1
                    else:
                        self.samples.append((window.copy(), label))
                        self.cam_ids.append(cam_id)
                    pos += self.stride

        if unknown_gestures:
            print(f"  WARNING: Unknown gestures ignored: {unknown_gestures}")
        if skipped_windows:
            print(f"  INFO: Skipped {skipped_windows} high-NaN windows.")
        print(f"  Dataset [{self.split}]: {len(self.samples)} windows "
              f"from {len(selected)} video(s).")

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Moderate augmentation: time warp ±10%, joint dropout 20%, noise σ=0.005."""
        T, F = x.shape
        # Time warp
        factor     = np.random.uniform(0.9, 1.1)
        new_len    = max(2, int(T * factor))
        src_idx    = np.linspace(0, T - 1, new_len)
        tgt_idx    = np.linspace(0, new_len - 1, T)
        x_warped   = np.zeros_like(x)
        for f in range(F):
            x_warped[:, f] = np.interp(tgt_idx, src_idx,
                                        np.interp(src_idx, np.arange(T), x[:, f]))
        x = x_warped
        # Joint dropout: zero out 1 random joint (20% chance)
        if np.random.random() < 0.2:
            joint = np.random.randint(0, 10)
            x[:, joint*3:(joint+1)*3] = 0.0
        # Joint noise
        noise = np.random.normal(0, 0.005, size=x.shape).astype(np.float32)
        x = x + noise
        return x

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.augment:
            x = self._augment(x)
        # One-hot camera-ID flag
        if self.n_cameras > 1:
            cam_id  = self.cam_ids[idx]
            one_hot = np.zeros(self.n_cameras, dtype=np.float32)
            one_hot[cam_id] = 1.0
            flag    = np.tile(one_hot, (self.window_size, 1))
            x       = np.concatenate([x, flag], axis=-1)
        x = torch.from_numpy(x).float()
        return x, y

    def get_sampler(self) -> WeightedRandomSampler:
        labels  = [s[1] for s in self.samples]
        counts  = np.bincount(labels, minlength=self.num_classes).astype(float)
        counts  = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        sample_weights = torch.tensor([weights[l] for l in labels])
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def class_distribution(self):
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        print("\nClass distribution:")
        for i, name in enumerate(self.classes):
            print(f"  {name:20s}  {counts[i]:5d} windows")
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python dataset_v2.py <skeleton_dir> <label_dir>")
        sys.exit(1)
    ds = SkeletonWindowDataset(skeleton_dir=sys.argv[1], label_dir=sys.argv[2], split="all")
    ds.class_distribution()
    x, y = ds[0]
    print(f"Sample shape: {x.shape}  label: {y} ({ds.classes[y]})")
```

### 3.3 `src/train_v2.py` — Training Loop

```python
"""
FERN v2 — Training loop.

Features:
- Mixed precision (AMP) with GradScaler
- Cosine LR schedule with linear warmup
- Gradient clipping (max_norm=1.0)
- EarlyStopping on validation loss
- Checkpoint saving (best model + latest epoch)
- TensorBoard logging
- Two-phase training (no aug during warmup, aug after)
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES
from model_v2   import FERNv2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_metric = 0.0
        self.counter     = 0
        self.should_stop = False

    def step(self, val_acc: float):
        if val_acc > self.best_metric + self.min_delta:
            self.best_metric = val_acc
            self.counter     = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss   = 0.0
    total_correct = 0
    total_samples = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss   = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            preds          = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += len(y)
            total_loss    += loss.item() * len(y)
    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def train(args):
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32      = True
        torch.backends.cudnn.allow_tf32             = True
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {vram:.1f} GB")
    print()

    # Input feature dimension
    base_features = 30  # 10 joints × 3 (x, y, z)
    input_features = base_features + (args.n_cameras if args.n_cameras > 1 else 0)
    dataset_kwargs = dict(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        classes=DEFAULT_CLASSES,
        window_size=args.window_size,
        stride=args.stride,
        split_mode=args.split_mode,
        n_cameras=args.n_cameras,
    )

    # Datasets
    if args.train_all:
        full_ds = SkeletonWindowDataset(**dataset_kwargs, split="all", augment=True)
        full_ds.class_distribution()
        train_loader = DataLoader(
            full_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = None
    else:
        train_ds = SkeletonWindowDataset(**dataset_kwargs, split="train", augment=True)
        val_ds   = SkeletonWindowDataset(**dataset_kwargs, split="val", augment=False)
        train_ds.class_distribution()
        train_sampler = train_ds.get_sampler()
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )

    # Model (optionally resume from checkpoint)
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt["epoch"]
        args.epochs = max(args.epochs, start_epoch)

    model = FERNv2(
        num_joints=10, num_classes=len(DEFAULT_CLASSES),
        cnn_out=args.cnn_out, lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers, dropout=args.dropout,
        input_features=input_features,
    ).to(device)

    if args.resume:
        model.load_state_dict(ckpt["model_state"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss, optimizer, scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler    = GradScaler()
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, base_lr=args.lr,
    )
    stopper   = EarlyStopping(patience=args.patience)

    if args.resume:
        optimizer.load_state_dict(ckpt["optim_state"])

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir)

    best_val_acc  = 0.0
    best_ckpt     = os.path.join(args.output_dir, "fern_v2_best.pth")
    latest_ckpt   = os.path.join(args.output_dir, "fern_v2_latest.pth")

    print(f"\nTraining for up to {args.epochs} epochs ...\n")
    header = f"{'Epoch':>6} {'LR':>8} {'T-Loss':>8} {'T-Acc':>7} {'V-Loss':>8} {'V-Acc':>7} {'Time':>7}"
    print(header)
    print("-" * len(header))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = scheduler.step(epoch)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, True
        )

        if val_loader is not None:
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device, False
            )
        else:
            val_loss, val_acc = 0.0, 0.0

        elapsed = time.time() - t0

        if val_loader is not None:
            print(f"{epoch+1:>6} {lr:>8.2e} {train_loss:>8.4f} "
                  f"{train_acc*100:>6.1f}% {val_loss:>8.4f} "
                  f"{val_acc*100:>6.1f}% {elapsed:>6.1f}s")
        else:
            print(f"{epoch+1:>6} {lr:>8.2e} {train_loss:>8.4f} "
                  f"{train_acc*100:>6.1f}% {elapsed:>6.1f}s")

        writer.add_scalar("Loss/train",    train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("LR",             lr,        epoch)

        ckpt = {
            "epoch":       epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_acc":     0.0,
            "args":        vars(args),
        }
        torch.save(ckpt, latest_ckpt)

        if args.train_all:
            continue

        writer.add_scalar("Loss/val",    val_loss,   epoch)
        writer.add_scalar("Accuracy/val", val_acc,   epoch)
        ckpt["val_acc"] = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt)
            print(f"  -> New best val acc: {best_val_acc*100:.2f}%")

        if epoch >= args.warmup_epochs:
            stopper.step(val_acc)
        if stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    writer.close()
    if args.train_all:
        print(f"\nTraining complete.  Final model saved to {latest_ckpt}")
    else:
        print(f"\nTraining complete.  Best val acc: {best_val_acc*100:.2f}%")


def parse_args():
    p = argparse.ArgumentParser(description="Train FERN v2")
    p.add_argument("--skeleton_dir",  default="data/skeletons")
    p.add_argument("--label_dir",     default="data/labels")
    p.add_argument("--output_dir",    default="models")
    p.add_argument("--log_dir",       default="logs")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--window_size",   type=int,   default=60)
    p.add_argument("--stride",        type=int,   default=15)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--cnn_out",       type=int,   default=32)
    p.add_argument("--lstm_hidden",   type=int,   default=64)
    p.add_argument("--lstm_layers",   type=int,   default=2)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--split_mode",    default="random", choices=["random", "subject"])
    p.add_argument("--train_all",     action="store_true")
    p.add_argument("--n_cameras",     type=int,   default=1)
    p.add_argument("--resume",        type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
```

### 3.4 `src/export_onnx.py` — ONNX Export

```python
"""Export FERNv2 checkpoint to ONNX."""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v2 import FERNv2


def export(
    checkpoint_path: str = r"..\models_final\fern_v2_latest.pth",
    output_path: str = r"..\models_final\fern_v2.onnx",
    num_joints: int = 10,
    num_classes: int = 8,
    cnn_out: int = 64,
    seq_len: int = 60,
    input_features: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_full = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt_full.get("args") if isinstance(ckpt_full, dict) else None
    if input_features is None and ckpt_args:
        n_cam = ckpt_args.get("n_cameras", 1)
        input_features = (num_joints * 3) + (n_cam if n_cam > 1 else 0)

    model = FERNv2(
        num_joints=num_joints, num_classes=num_classes, cnn_out=cnn_out,
        lstm_hidden=0, lstm_layers=1, dropout=0.6,
        input_features=input_features,
    ).to(device)

    if isinstance(ckpt_full, dict):
        state = ckpt_full.get("model_state") or ckpt_full.get("model_state_dict") or ckpt_full
    else:
        state = ckpt_full
    model.load_state_dict(state, strict=True)
    model.eval()

    feat_dim = input_features or num_joints * 3
    dummy = torch.randn(1, seq_len, feat_dim, device=device)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported -> {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check: OK")

    session = ort.InferenceSession(output_path)
    ort_input = {session.get_inputs()[0].name: dummy.cpu().numpy()}
    ort_out = session.run(None, ort_input)[0]
    torch_out = model(dummy).detach().cpu().numpy()
    diff = np.abs(ort_out - torch_out).max()
    print(f"Max diff torch vs onnxruntime: {diff:.2e}")
    assert diff < 5e-3, f"ONNX mismatch: {diff:.2e}"
    print("ONNX inference: OK")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export FERNv2 checkpoint to ONNX")
    p.add_argument("--checkpoint_path", default=r"..\models_final\fern_v2_latest.pth")
    p.add_argument("--output_path", default=r"..\models_final\fern_v2.onnx")
    p.add_argument("--num_joints", type=int, default=10)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--cnn_out", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--input_features", type=int, default=None)
    a = p.parse_args()
    export(checkpoint_path=a.checkpoint_path, output_path=a.output_path,
           num_joints=a.num_joints, num_classes=a.num_classes,
           cnn_out=a.cnn_out, seq_len=a.seq_len, input_features=a.input_features)
```

### 3.5 `src/test_onnx.py` — ONNX Evaluation

```python
"""Test ONNX model accuracy on a dataset with optional camera-ID flag support."""

import sys, json, argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2 import load_skeleton_csv, DEFAULT_CLASSES, build_class_map


def test(onnx_path, skeleton_dir, label_dir, n_cameras=1, window_size=60, stride=15):
    class_map = build_class_map(DEFAULT_CLASSES)
    skel_path = Path(skeleton_dir)
    label_path = Path(label_dir)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    total = 0; correct = 0
    per_camera = {}
    confusion = np.zeros((8, 8), dtype=int)

    for csv_path in sorted(skel_path.glob("*.csv")):
        json_path = label_path / csv_path.with_suffix(".json").name
        if not json_path.exists():
            continue
        try:
            skeleton = load_skeleton_csv(str(csv_path))
            with open(json_path) as fh:
                label_data = json.load(fh)
            segments = label_data["segments"]
            cam_id = int(label_data.get("camera_id", 0))
        except Exception as e:
            print(f"  SKIP {csv_path.name}: {e}")
            continue

        per_camera.setdefault(cam_id, {"correct": 0, "total": 0})
        T = len(skeleton)
        for seg in segments:
            gesture = seg["gesture"]
            if gesture not in class_map:
                continue
            label = class_map[gesture]
            s = seg["start_frame"]
            e = min(seg["end_frame"], T - 1)
            pos = s
            while pos + window_size <= e + 1:
                window = skeleton[pos:pos + window_size]
                if np.isnan(window).mean() > 0.3:
                    pos += stride
                    continue
                if n_cameras > 1:
                    one_hot = np.zeros(n_cameras, dtype=np.float32)
                    one_hot[cam_id] = 1.0
                    window = np.concatenate([window, np.tile(one_hot, (window_size, 1))], axis=-1)
                inp = window.astype(np.float32).reshape(1, window_size, -1)
                pred = int(session.run(None, {input_name: inp})[0].argmax())
                total += 1
                per_camera[cam_id]["total"] += 1
                if pred == label:
                    correct += 1
                    per_camera[cam_id]["correct"] += 1
                confusion[label, pred] += 1
                pos += stride

    print(f"\nTotal windows: {total}  Correct: {correct}  Accuracy: {100*correct/total:.2f}%\n")
    for cid in sorted(per_camera):
        c = per_camera[cid]
        print(f"  Camera {cid}: {c['correct']}/{c['total']} = {100*c['correct']/c['total']:.2f}%")
    print()
    print(f"{'':>15}", end="")
    for c in DEFAULT_CLASSES: print(f"{c:>15}", end="")
    print()
    for i in range(8):
        print(f"{DEFAULT_CLASSES[i]:>15}", end="")
        for j in range(8): print(f"{confusion[i,j]:>15}", end="")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path", default=r"..\models_final\fern_v2.onnx")
    p.add_argument("--skeleton_dir", default=r"..\data\skeletons\front")
    p.add_argument("--label_dir", default=r"..\data\labels\front")
    p.add_argument("--n_cameras", type=int, default=1)
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=15)
    a = p.parse_args()
    test(onnx_path=a.onnx_path, skeleton_dir=a.skeleton_dir, label_dir=a.label_dir,
         n_cameras=a.n_cameras, window_size=a.window_size, stride=a.stride)
```

### 3.6 `src/infer_v2.py` — Live Inference (Needs Camera-ID Support)

```python
"""Live inference from camera with rolling skeleton buffer."""

import argparse, collections, time
import cv2, mediapipe as mp
import numpy as np, torch, torch.nn.functional as F
from extract_skeleton import FEATURE_JOINT_INDICES, FEATURE_JOINT_NAMES, NUM_FEATURES, normalize_landmarks
from dataset_v2 import DEFAULT_CLASSES
from model_v2 import FERNv2


class SkeletonBuffer:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)

    def push(self, features: np.ndarray):
        self.buffer.append(features)

    def ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        return np.stack(list(self.buffer), axis=0).astype(np.float32)


def frame_to_features(results) -> np.ndarray:
    if not results.pose_landmarks:
        return np.zeros(NUM_FEATURES, dtype=np.float32)
    coords, _ = normalize_landmarks(results.pose_landmarks.landmark)
    return np.array(coords, dtype=np.float32)


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    model = FERNv2(
        num_joints=10,
        num_classes=len(DEFAULT_CLASSES),
        cnn_out=saved.get("cnn_out", 64),
        lstm_hidden=saved.get("lstm_hidden", 128),
        lstm_layers=saved.get("lstm_layers", 2),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    try:
        src = int(args.camera_id)
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    except ValueError:
        src = args.camera_id
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        return

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    buffer     = SkeletonBuffer(args.window_size)

    current_label  = "waiting..."
    current_conf   = 0.0
    frames_since_pred = 0
    fps_times = collections.deque(maxlen=30)

    print("Running.  Press Q to quit.")

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            feats   = frame_to_features(results)
            buffer.push(feats)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,0), thickness=2),
                )

            frames_since_pred += 1
            if buffer.ready() and frames_since_pred >= args.stride:
                frames_since_pred = 0
                window = buffer.get_window()
                x = torch.from_numpy(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
                best_idx  = int(probs.argmax())
                best_conf = float(probs[best_idx])
                if best_conf >= args.threshold:
                    current_label = DEFAULT_CLASSES[best_idx]
                    current_conf  = best_conf
                else:
                    current_label = "uncertain"
                    current_conf  = best_conf

            fps_times.append(time.time() - t0)
            avg_fps = 1.0 / (sum(fps_times) / len(fps_times))

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
            color = (0, 255, 0) if current_conf >= args.threshold else (0, 165, 255)
            cv2.putText(frame, f"Gesture: {current_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(frame, f"Confidence: {current_conf*100:.1f}%", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}  Buffer: {len(buffer.buffer)}/{args.window_size}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
            if h > w:
                scale = 800 / h
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            cv2.imshow("FERN v2 — Live Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--camera_id", type=str, default="0")
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.6)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
```

### 3.7 `src/kfold_cv.py` — Cross-Validation (Partially Executed)

```python
import argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES
from model_v2 import FERNv2


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0; total_correct = 0; total_samples = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += len(y)
            total_loss += loss.item() * len(y)
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton_dir", default="data/skeletons/front")
    p.add_argument("--label_dir", default="data/labels/front")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--cnn_out", type=int, default=64)
    p.add_argument("--lstm_hidden", type=int, default=0)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    common_args = dict(
        skeleton_dir=args.skeleton_dir, label_dir=args.label_dir,
        classes=DEFAULT_CLASSES, window_size=args.window_size,
        stride=args.stride, split="all",
    )
    train_ds = SkeletonWindowDataset(**common_args, augment=True)
    val_ds   = SkeletonWindowDataset(**common_args, augment=False)

    n_total = len(train_ds)
    indices = list(range(n_total))
    np.random.RandomState(args.seed).shuffle(indices)
    fold_size = n_total // args.k_folds
    print(f"Total windows: {n_total}   Fold size: ~{fold_size}")

    fold_accs = []
    for fold in range(args.k_folds):
        print(f"\n{'='*50}\nFold {fold+1}/{args.k_folds}")
        val_start = fold * fold_size
        val_end = n_total if fold == args.k_folds - 1 else (fold + 1) * fold_size
        val_set = set(indices[val_start:val_end])
        train_idx = [i for i in indices if i not in val_set]
        val_idx = list(val_set)
        print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx), num_workers=args.num_workers,
            pin_memory=(device.type=="cuda"), drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(val_idx), num_workers=args.num_workers,
            pin_memory=(device.type=="cuda"))

        model = FERNv2(num_joints=10, num_classes=len(DEFAULT_CLASSES),
            cnn_out=args.cnn_out, lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers, dropout=args.dropout).to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler()
        patience = 30; best_val_acc = 0.0; no_improve = 0

        for epoch in range(args.epochs):
            lr = args.lr if epoch >= args.warmup_epochs else args.lr * (epoch+1) / args.warmup_epochs
            for pg in optimizer.param_groups: pg["lr"] = lr
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False)
            if val_acc > best_val_acc: best_val_acc = val_acc; no_improve = 0
            else: no_improve += 1
            if epoch % 10 == 0 or epoch == args.epochs-1 or no_improve == patience:
                print(f"  E{epoch+1:>3}  L:{lr:.2e}  T:{train_loss:.3f}/{train_acc*100:.1f}%  V:{val_loss:.3f}/{val_acc*100:.1f}%  Best:{best_val_acc*100:.1f}%")
            if no_improve >= patience and epoch >= args.warmup_epochs: break

        fold_accs.append(best_val_acc)
        print(f"  >> Fold {fold+1} best: {best_val_acc*100:.2f}%")

    print(f"\n{'='*50}\nK-fold CV ({args.k_folds} folds):")
    for i, acc in enumerate(fold_accs): print(f"  Fold {i+1}: {acc*100:.2f}%")
    print(f"  Mean: {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")
    print(f"{'='*50}")

if __name__ == "__main__": main()
```

### 3.8 `src/add_foot_hold_gaps.py` — Gap Insertion

```python
"""Insert 60-frame foot_hold gaps at gesture transitions where gap is too small."""
import json, math
from pathlib import Path

LABEL_DIR = Path("C:/fern/FERN_V2/data/labels/combined")
MIN_SEGMENT = 20
TARGET_GAP = 60

def total_foot_hold_frames(segments):
    return sum(s["end_frame"] - s["start_frame"] + 1 for s in segments if s["gesture"] == "foot_hold")

def add_foot_hold_gaps(label_path):
    with open(label_path, encoding="utf-8") as f: label = json.load(f)
    segments = label.get("segments", [])
    if not segments: return False
    if total_foot_hold_frames(segments) >= TARGET_GAP: return False
    result = []
    for i, seg in enumerate(segments):
        gesture, start, end = seg["gesture"], seg["start_frame"], seg["end_frame"]
        if i > 0 and segments[i-1]["gesture"] != gesture:
            prev = result.pop()
            gap = start - prev["end_frame"]
            prev_len = prev["end_frame"] - prev["start_frame"] + 1
            curr_len = end - start + 1
            max_trim_prev = prev_len - MIN_SEGMENT
            max_trim_curr = curr_len - MIN_SEGMENT
            if gap >= TARGET_GAP:
                fh_start, fh_end = prev["end_frame"] + 1, start - 1
                result.append(prev)
                if fh_end >= fh_start:
                    result.append({"gesture": "foot_hold", "start_frame": fh_start, "end_frame": fh_end})
                continue
            if max_trim_prev > 0 and max_trim_curr > 0:
                needed = TARGET_GAP + 1 - gap
                trim_prev = min(int(math.ceil(needed / 2)), max_trim_prev)
                trim_curr = min(needed - trim_prev, max_trim_curr)
                if gap + trim_prev + trim_curr - 1 >= TARGET_GAP:
                    prev["end_frame"] -= trim_prev
                    fh_start, fh_end = prev["end_frame"] + 1, start + trim_curr - 1
                    result.append(prev)
                    if fh_end >= fh_start:
                        result.append({"gesture": "foot_hold", "start_frame": fh_start, "end_frame": fh_end})
                    start = start + trim_curr
                else: result.append(prev)
            else: result.append(prev)
        result.append({"gesture": gesture, "start_frame": start, "end_frame": end})
    label["segments"] = result
    label["foot_hold_gaps_added_by"] = "add_foot_hold_gaps.py"
    inserted = total_foot_hold_frames(result) - total_foot_hold_frames(segments)
    with open(label_path, "w", encoding="utf-8") as f: json.dump(label, f, indent=2, ensure_ascii=False)
    return inserted

def main():
    jsons = sorted(LABEL_DIR.glob("*.json")); total_inserted = 0; modified = 0
    for p in jsons:
        inserted = add_foot_hold_gaps(p)
        if inserted is False: continue
        if inserted > 0: print(f"  {p.stem}: +{inserted} foot_hold frames"); total_inserted += inserted; modified += 1
        else: print(f"  {p.stem}: already has >=60 foot_hold frames")
    print(f"\nModified {modified}/{len(jsons)} files\nTotal foot_hold frames inserted: {total_inserted}")

if __name__ == "__main__": main()
```

### 3.9 `src/mirror_10joint.py` — Mirror Augmentation

```python
"""
Mirror augmentation: negate X, swap left↔right joints, write _mirror files.
"""

import argparse, csv, json, os
from pathlib import Path

LR_SWAP_PAIRS = [
    (0, 3),   # hip
    (6, 9),   # knee
    (12, 15), # ankle
    (18, 21), # heel
    (24, 27), # foot_index
]

def mirror_skeleton_csv(in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(in_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.reader(fin); writer = csv.writer(fout)
        header = next(reader); writer.writerow(header)
        for row in reader:
            if len(row) < 34: continue
            coords = [float(v) if v else 0.0 for v in row[2:]]
            for i in range(0, len(coords), 3): coords[i] = -coords[i]  # negate X
            for l, r in LR_SWAP_PAIRS:
                for off in range(3): coords[l+off], coords[r+off] = coords[r+off], coords[l+off]
            coords[30] = -coords[30]  # mid_hip_x
            writer.writerow(row[:2] + [f"{v:.12f}" for v in coords])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton_dir", default="data/skeletons/merged_v1")
    ap.add_argument("--label_dir", default="data/labels/merged_v1")
    ap.add_argument("--output_skel", default="data/skeletons/merged_v1")
    ap.add_argument("--output_label", default="data/labels/merged_v1")
    args = ap.parse_args()

    skel_dir = Path(args.skeleton_dir)
    label_dir = Path(args.label_dir)
    out_skel = Path(args.output_skel)
    out_label = Path(args.output_label)

    mirrored = 0; skipped = 0
    for csv_path in sorted(skel_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem.endswith("_mirror"): continue
        label_path = label_dir / f"{stem}.json"
        if not label_path.exists(): skipped += 1; continue
        out_csv = out_skel / f"{stem}_mirror.csv"
        out_json = out_label / f"{stem}_mirror.json"
        mirror_skeleton_csv(str(csv_path), str(out_csv))
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(label_path) as f: label = json.load(f)
        label["mirror_of"] = stem
        with open(out_json, "w") as f: json.dump(label, f, indent=2)
        mirrored += 1
        print(f"  Mirrored: {stem}")
    print(f"\nDone. Mirrored: {mirrored}  Skipped (no label): {skipped}")

if __name__ == "__main__": main()
```

### 3.10 `src/add_camera_id.py` — Camera ID Injection

```python
"""Add camera_id to label JSONs based on filename camera suffix."""
import json
from pathlib import Path

# c3 front labels -> camera_id 0
front_dir = Path("data/labels/front")
updated = 0
for f in sorted(front_dir.glob("*.json")):
    with open(f) as fh: data = json.load(fh)
    if "camera_id" not in data:
        data["camera_id"] = 0
        with open(f, "w") as fh: json.dump(data, fh, indent=2)
        updated += 1
print(f"  front: {updated}/{len(list(front_dir.glob('*.json')))} updated (camera_id=0)")

# front_plus_45 labels: c3->0, c2->1
p45_dir = Path("data/labels/front_plus_45")
updated_c3 = 0; updated_c2 = 0
for f in sorted(p45_dir.glob("*.json")):
    with open(f) as fh: data = json.load(fh)
    stem = f.stem.replace("_mirror", "")
    target = 1 if "c2" in stem else 0
    if data.get("camera_id") != target:
        data["camera_id"] = target
        with open(f, "w") as fh: json.dump(data, fh, indent=2)
        if target == 0: updated_c3 += 1
        else: updated_c2 += 1

all_jsons = list(p45_dir.glob("*.json"))
c3_count = len([f for f in all_jsons if "c2" not in f.stem.replace("_mirror", "")])
c2_count = len([f for f in all_jsons if "c2" in f.stem.replace("_mirror", "")])
print(f"  front_plus_45 c3: {updated_c3}/{c3_count} updated (camera_id=0)")
print(f"  front_plus_45 c2: {updated_c2}/{c2_count} updated (camera_id=1)")
```

---

## 4. All Experiments & Results

### 4.1 Experiment Summary

| # | Exp | Dataset | Arch | cnn_out | Params | Seed | Best Val Acc | Best Ep |
|---|-----|---------|------|---------|--------|------|-------------|---------|
| BL | baseline | all (144 files) | BiLSTM | 32 | 207K | 42 | 24.59% | 26 |
| F0 | front-only | front (76) | BiLSTM | 32 | 207K | 42 | 24.80% | 31 |
| 1 | exp1 | combined (144) | BiLSTM | 32 | 207K | 42 | 35.12% | 14 |
| 2 | exp2 | front (76) | CNN-only | 32 | 33K | 42 | **74.10%** | 145 |
| 3 | exp3 | front (76) | CNN-only | 64 | 132K | 42 | **80.22%** | 120 |
| 4a | exp4a | front (76) | CNN-only | 64 | 132K | 43 | 78.06% | 80 |
| 4b | exp4b | front (76) | CNN-only | 64 | 132K | 44 | 80.22% | 183 |
| 5 | exp5 | combined (144) | CNN-only | 64 | 132K | 42 | 38.29% | 1 |
| 6 | exp6 | front (76) | CNN-only | 64 | 132K | 42 | 79.50% | 116 |
| 7 | exp7 | front (76) | CNN-only | 96 | 296K | 42 | 79.14% | 85 |
| F | final | front (76) | CNN-only | 64 | 132K | 42 | train_all | 200 |
| T1 | transform | front+45 (98) | CNN-only | 64 | 132K | 42 | 60.14% (val) | 46 |
| T2 | final_v2_nf | front+45 (98) | CNN-only | 64 | 132K | 42 | train_all | 200 |
| P1 | Phase1_flag | front+45 (98) | CNN-only | 64 | **140K** | 42 | train_all | 200 |

### 4.2 Production Models Comparison

| Model | Path | Trained On | Input Dim | Front Acc | C2 Acc | Combined Acc |
|-------|------|-----------|-----------|-----------|--------|-------------|
| Front-only | `models_final/fern_v2.onnx` | front (76 files) | 30 | **62.58%** | — | **62.58%** |
| Front-only (old, pre-A1 fix) | — | front | 30 | **70.64%** | — | **70.64%** |
| Phase 1 (flag, post-fix) | `models_final_v2/fern_v2.onnx` | front+45 (98 files) | 32 | 53.53% | **48.20%** | **50.48%** |
| Phase 1 (old, pre-A1 fix) | — | front+45 | 32 | 54.70% | 56.39% | 55.71% |

### 4.3 Key Findings

1. **LSTM hurts**: CNN-only outperforms BiLSTM by 45%+ on this small dataset.
2. **CNN-only with global average pooling** over time is better than BiLSTM + attention.
3. **cnn_out=64 is optimal**: 32→64 gains +6%, 64→96 loses.
4. **Stable across seeds**: 78-80% (mean 79.5%, σ=1.25%) — old numbers were inflated by dead NaN filter.
5. **Unbiased CV estimate (post-A1 fix)**: ~42.58% front-only, ~27.93% front+45 (subject-level). Prior ~60.5% was inflated by interpolated garbage windows passing through the dead NaN filter.
6. **Augmentation**: Moderate is best (time warp ±10%, joint dropout 20%, noise σ=0.005).
7. **Geometric transform FAILED**: R_y(-θ) using MediaPipe z cannot recover front view — all angles 14-15% (=random).
8. **Camera-ID flag still works**: Prevents total collapse on multi-angle (without flag combined ~38%), but post-A1 fix shows more modest gains.

### 4.4 Per-Class Accuracy (Front-Only Model, No-Aug Train Set)

| Class | Accuracy | Samples | Notes |
|-------|----------|---------|-------|
| cross_front | **91.6%** | 558 | Best — distinctive frontal leg cross |
| sideway_kick | **87.0%** | 386 | Good — lateral motion is clear |
| forward_step | **77.7%** | 430 | Moderate |
| flamingo_bend | **76.0%** | 312 | Moderate |
| heel_tap | **73.8%** | 332 | Moderate |
| forward_kick | **65.5%** | 414 | Moderate |
| foot_lift | **64.7%** | 334 | Moderate |
| foot_hold | **32.5%** | 514 | **Worst** — idle class, spread across all |

### 4.5 5-Fold Cross-Validation (Post-A1 Fix, Unbiased Leak-Free Splits)

#### Front-only (video-level grouping)

| Fold | Accuracy | Val Windows |
|------|:--------:|:-----------:|
| Fold 1 | 33.57% | 700 |
| Fold 2 | 34.13% | 504 |
| Fold 3 | 44.24% | 938 |
| Fold 4 | 34.03% | 626 |
| Fold 5 | 66.92% | 260 (very small, easy fold) |
| **Mean** | **42.58%** | |
| **Std** | **±12.82%** | |

#### Front+45 (subject-level, n_cameras=2)

| Fold | Accuracy |
|------|:--------:|
| Fold 1 | 21.77% |
| Fold 2 | 33.00% |
| Fold 3 | 32.39% |
| Fold 4 | 24.60% |
| Fold 5 | 27.90% |
| **Mean** | **27.93%** |
| **Std** | **±4.35%** |

---

## 5. Known Issues & What Needs Fixing

### Issue 1: Low Overall Accuracy

**Problem**: The best model achieves ~62.58% front-only and ~50.48% combined on clean data. The unbiased subject-level CV gives only 42.58% (front-only) and 27.93% (front+45). This is too low for production use.

**Root cause revealed by A1 fix**: The old NaN filter was dead code — interpolation removed NaNs before windowing, so all windows passed. Fixing the detection-ratio filter (skip windows with ratio < 0.7) dropped front windows 3,280 → 3,028 (-7.7%) and front+45 windows 8,166 → 7,074 (-13.4%), but also revealed that many low-detection windows were contaminating the dataset. The model's true accuracy on clean data is ~62% (front-only). For c2 camera angle, detection is lower and accuracy drops to ~48%.

**Possible causes**:
- Only 10 subjects in the filtered dataset — too little variation
- MediaPipe skeleton detection quality varies by camera angle (c2 is worse than front)
- 10 lower-body joints may not be enough discriminative features
- Foot_hold (idle) class dominates errors — it's the "catch-all" and attracts false positives
- High CV variance (±12.82%) + some very low folds (33%) suggest the model struggles to generalize across subjects even on front-only data
- This is a small dataset (76 files, 10 subjects, 3,028 windows after filtering) — low accuracy is expected

**Possible fixes**:
- Collect more training data from more subjects (ideally 30+)
- Use more joints (full body 33 joints instead of 10)
- Use a larger CNN (cnn_out=128 or deeper architecture)
- Consider temporal CNN (TCN) instead of simple global average pooling
- Per-subject fine-tuning at deployment time

### Issue 2: c4 Camera Angle Unusable

**Problem**: All 11 c4 (45° right) video files produce MediaPipe detection rates <30%. Only p05_c4 at 28.4% detection yields any usable windows (98 out of 9930 frames). c4 is completely excluded.

**Possible causes**:
- MediaPipe Pose is not trained on extreme side views of lower body
- The 45° right angle may have legs occluding each other
- Camera positioning relative to subject may be suboptimal

**Possible fixes**:
- Try YOLOv8-pose or another keypoint detector for challenging angles
- Use more cameras at once for triangulation (requires synchronized multi-cam setup)
- Accept c4 exclusion and focus on c3 + c2 (what Phase 1 did)

### Issue 3: Camera-ID Flag Implementation Needs `infer_v2.py` Support

**Problem**: `infer_v2.py` (live inference) creates the model without the `input_features` parameter. If the model was trained with n_cameras=2 (32-dim input), `infer_v2.py` will create a 30-dim model and fail when loading state dict.

**Also**: `infer_v2.py` doesn't append the camera-ID one-hot flag at inference time — it has no concept of camera_id. The live inference pipeline only works with the 30-dim front-only model.

**Fix needed**:
- Add `--n_cameras` and `--camera_id` args to `infer_v2.py`
- Read `input_features` from the checkpoint args
- Append one-hot flag to the buffer window before inference
- Modify `frame_to_features` or add camera-ID aware inference path

### Issue 4: `kfold_cv.py` Doesn't Support `--n_cameras`

**Problem**: The cross-validation script doesn't pass `n_cameras` to the dataset or `input_features` to the model. It can't be used to get an unbiased CV estimate for the Phase 1 camera-ID model.

**Fix needed**:
- Add `--n_cameras` arg to `kfold_cv.py`
- Compute `input_features` and pass to model
- Pass `n_cameras` to dataset

### Issue 5: No Subject-Uniform CV for Phase 1 Model

**Problem**: The 5-fold CV was only done on the front-only model. The Phase 1 (camera-ID flag) model has no subject-uniform CV estimate. The reported 55.71% is a no-aug training set accuracy, not a true generalization estimate.

**Fix needed**: Use `split_mode="subject"` with k-fold CV or a proper train/val/test split to get an unbiased estimate for the camera-ID model.

### Issue 6: Training Accuracy with Augmentation Is Very Low (~40%)

**Problem**: The Phase 1 model shows only ~40% training accuracy with augmentation, compared to ~57-59% for the front-only model with the same augmentation. This suggests the model fundamentally struggles to reconcile c3 and c2 skeleton patterns, even with camera-ID flags.

**Possible causes**:
- Camera-ID flag may not be providing enough signal to differentiate the two views
- The same CNN weights are applied to both camera views — the flag only helps the classifier head
- Differently, the c2 (side view) skeletons may look very different from c3 (front view) in the CNN feature space

**Possible fixes**:
- Use separate CNN branches per camera (multi-head architecture)
- Add camera-ID conditioning at the CNN level (FiLM, adaptive instance normalization)
- Train camera-specific batch normalization layers
- Use prototypical networks or metric learning per-camera

### Issue 7: Augmentation Applied Before Camera-ID Flag

**Problem**: In `dataset_v2.__getitem__`, augmentation is applied before the camera-ID flag is concatenated. The time-warp augmentation (`np.interp`) operates on all features including the yet-to-be-added flag. This is correct (the flag should not be augmented). But the current order is: augment(30-dim) → concat flag → tensor. This is fine as-is.

**No fix needed** — current behavior is correct.

### Issue 8: Dataset Leaks Mirror Files During Split

**Problem**: Mirror files (named `*_mirror.csv`) are paired with their originals via `_random_split_pairs()` and `_subject_split_pairs()`, which strip the `_mirror` suffix to find the base name. This means mirrors stay in the same split as originals — preventing train/val/test leakage. This is correct.

**No fix needed**.

### Issue 9: `add_foot_hold_gaps.py` Uses Hardcoded `data/labels/combined` Path

**Problem**: The script hardcodes `C:/fern/FERN_V2/data/labels/combined` as LABEL_DIR. This only worked for the old "combined" dataset. It was never run on `front` or `front_plus_45` labels.

**Fix needed**: Make the label directory a CLI argument.

### Issue 10: No Streaming/Incremental Inference Support

**Problem**: The production use case requires real-time streaming inference, but the current `infer_v2.py` uses a PyTorch model, not ONNX Runtime. ONNX Runtime would be faster and doesn't require PyTorch in deployment.

**Fix needed**: 
- Create an `infer_onnx.py` that uses ONNX Runtime for deployment
- Add camera-ID support to the ONNX inference pipeline
- Support both 30-dim (front-only) and 32-dim (Phase 1) models

### Issue 11: Final Phase 1 Model Was Trained with `--train_all` (No Validation)

**Problem**: The Phase 1 model was trained with `--train_all`, meaning there's no early stopping checkpoint, no validation curve, and no way to know the best epoch. The model may have overfit or stopped before convergence.

**Fix needed**: Re-train with validation split, use early stopping, save best checkpoint, and report unbiased validation accuracy.

### Issue 12: `export_onnx.py` Has Relative Paths That Break from Project Root

**Problem**: The default paths in `export_onnx.py` are `r"..\models_final\..."` which work when running from the `src/` directory but fail when running from the project root `C:\fern\FERN_V2`. The CLI args override this, but the defaults are confusing.

**Fix needed**: Use absolute paths or detect script location.

### Summary of Required Changes

| Priority | Issue | File(s) | Change |
|----------|-------|---------|--------|
| **P0** | Detection quality: low-detection windows contaminate dataset | `dataset_v2.py`, `test_onnx.py` | A1 fix applied: detection-ratio filter now filters windows with <0.7 detected ratio |
| **P0** | infer_v2.py doesn't support camera-ID flag | `infer_v2.py` | Add `--n_cameras`, `--camera_id`, read `input_features` from checkpoint |
| **P1** | kfold_cv.py doesn't support n_cameras | `kfold_cv.py` | Add `--n_cameras`, pass to dataset and model |
| **P1** | No subject-uniform CV for Phase 1 model | Run experiment | Run 5-fold CV with `split_mode=subject` on front_plus_45+n_cameras=2 |
| **P1** | Model accuracy too low (56%) | Architecture | Try: cnn_out=128, multi-head CNN per camera, FiLM conditioning, or TCN |
| **P2** | add_foot_hold_gaps.py hardcodes path | `add_foot_hold_gaps.py` | Make LABEL_DIR a CLI arg |
| **P2** | ONNX inference missing (PyTorch only) | New `infer_onnx.py` | Create ONNX Runtime streaming inference with camera-ID support |
| **P2** | c4 camera unusable | N/A | Try alternative pose estimator (YOLOv8-pose) for extreme angles |
| **P3** | export_onnx.py confusing default paths | `export_onnx.py` | Fix defaults or document |

---

## 6. Environment

### Hardware
- **GPU**: NVIDIA GeForce RTX 3070 Laptop GPU (8.6 GB VRAM)
- **CPU**: Not specified

### Software
- **Python**: 3.12+ (system has Python 3.14)
- **Virtual env**: `C:\fern\FERN_V2\venv\` — the primary environment
- **PyTorch**: 2.2.2+cu121 (CUDA 12.1)
- **ONNX Runtime**: onnxruntime-gpu 1.26.0
- **MediaPipe**: 0.10.35 (requires protobuf <5.0.0 — pinned)
- **CUDA**: 12.1
- **OS**: Windows 11

### Key Dependencies
```
torch==2.2.2+cu121
torchvision==0.17.2+cu121
onnxruntime-gpu==1.26.0
onnx==1.16.0
mediapipe==0.10.35
opencv-python==4.9.0
pandas==2.2.0
numpy==1.26.4
protobuf<5.0.0
```

### Training Commands

```powershell
# Train Phase 1 (c3+c2 with camera-ID flag) from venv:
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/train_v2.py `
    --skeleton_dir data/skeletons/front_plus_45 `
    --label_dir data/labels/front_plus_45 `
    --output_dir models_final_v2 `
    --log_dir logs_final_v2 `
    --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 `
    --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --lstm_hidden 0 --lstm_layers 1 `
    --device cuda --num_workers 4 --n_cameras 2 --train_all

# Export Phase 1 model:
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/export_onnx.py `
    --checkpoint_path models_final_v2/fern_v2_latest.pth `
    --output_path models_final_v2/fern_v2.onnx

# Evaluate Phase 1:
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/test_onnx.py `
    --onnx_path models_final_v2/fern_v2.onnx `
    --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 `
    --n_cameras 2 --window_size 60 --stride 15
```

---

## 7. Dataset Details

### Front Dataset

| Property | Value |
|----------|-------|
| Camera | c3 (0°, front) |
| Original files | 38 (one per subject) |
| Mirror files | 38 |
| **Total CSVs** | **76** |
| Subjects | 11 (p00-p11, filtered: kept p04, p05, p07, p08, p10, p11) |
| Excluded subjects | p00 (60.6%), p01 (0%), p02 (0%), p03 (38.8%), p06 (7%), p09 (8%) + mirrors |
| Total windows (post-A1 fix) | 3,028 (was 3,280 pre-fix) |
| z | 0 (all zeroed) |
| Labels | camera_id=0 in all JSONs |
| Foot_hold gaps | 60-frame segments inserted at gesture-group transitions |

### Front+45 Dataset (Phase 1)

| Property | Front (c3) | C2 (45° left) |
|----------|-----------|---------------|
| Files | 76 (38+38 mirror) | 22 (11+11 mirror) |
| Total | **98** | |
| Windows (post-A1 fix) | 3,028 (was 3,280) | 4,046 (was 4,886) |
| **Total windows (post-A1 fix)** | **7,074** (was 8,166) | |
| Camera ID | 0 | 1 |
| z | 0 | 0 |
| Source | `data/skeletons/front/` | `data/skeletons/front_plus_45/` (originally from `merged_v1/`) |

### Class Distribution (Phase 1 Dataset)

| Class | Windows |
|-------|---------|
| foot_hold | 912 |
| foot_lift | 804 |
| sideway_kick | 1,104 |
| cross_front | 1,270 |
| heel_tap | 1,010 |
| flamingo_bend | 906 |
| forward_step | 1,118 |
| forward_kick | 1,042 |
| **Total (post-A1 fix)** | **7,074** (was 8,166) |

### c4 Detection Summary

| File | Detection % | Detected | Total Frames | Valid Windows |
|------|:-----------:|:--------:|:------------:|:-------------:|
| p05_c4 | 28.4% | 2,817 | 9,930 | 98 |
| p02_c4 | 15.2% | 823 | 5,417 | 46 |
| p06_c4 | 6.4% | 479 | 7,432 | 2 |
| Others | 0-5.3% | <200 | 2,500-10,000 | 0 |

---

## 8. Model Architecture Details

### CNN-Only Mode (Current Best, `lstm_hidden=0`)

```
Input: (batch, T=60, F=30 or 32)
  │
  ├─ Reshape → (batch*T, 1, F)
  ├─ Conv1d(1 → cnn_out//2, k=3, pad=1) → BN → ReLU
  ├─ Conv1d(cnn_out//2 → cnn_out, k=3, pad=1) → BN → ReLU
  ├─ Flatten → Linear(cnn_out*F → cnn_out) → Dropout → ReLU
  │
  ├─ Reshape → (batch, T, cnn_out)
  ├─ Global Average Pool over time → (batch, cnn_out)
  │
  └─ Classifier:
       ├─ Dropout
       ├─ Linear(cnn_out → cnn_out//2) → ReLU
       ├─ Dropout
       └─ Linear(cnn_out//2 → 8) → logits
```

**Parameter counts**:
- cnn_out=32: 33K params
- cnn_out=64 (30-dim): 132K params
- cnn_out=64 (32-dim): 140K params
- cnn_out=96: 296K params

### BiLSTM Mode (Deprecated, `lstm_hidden>0`)

```
Input → same SpatialCNN block → BiLSTM(hidden=lstm_hidden, bidir) → AdditiveAttention → Classifier
```

- cnn_out=32, lstm_hidden=64: 207K params — performs worse than CNN-only

### Hyperparameters (Best Config)

| Parameter | Value |
|-----------|-------|
| cnn_out | 64 |
| lstm_hidden | 0 (CNN-only mode) |
| dropout | 0.6 |
| lr | 3e-4 |
| weight_decay | 1e-2 |
| batch_size | 32 |
| warmup_epochs | 20 |
| max_epochs | 200 |
| label_smoothing | 0.1 |
| gradient_clip | max_norm=1.0 |
| optimizer | AdamW |
| scheduler | CosineAnnealing with linear warmup |
| augmentation | time_warp ±10%, joint_dropout 20%, noise σ=0.005 |
| window_size | 60 (2s at 30fps) |
| stride | 15 (0.5s) |

---

*Generated: 2026-06-10. Last updated: 2026-06-11 (post-A1 fix). This report contains the complete FERN v2 project state for an AI model to understand and fix all known issues.*

---

## 9. Next Steps

1. **Subject-independent evaluation**: The true subject-level CV gives 42.58%. The model needs more data or better generalization to be production-ready.
2. **Data augmentation improvements**: Time warping, joint dropout, noise already help. Try Mixup, cutmix, or adversarial augmentation.
3. **Multi-head or FiLM architecture**: The camera-ID one-hot might not be expressive enough for the drastic viewpoint change (90-degree rotation). Try FiLM conditioning or separate front/side heads.
4. **More data**: Current dataset is small (10 subjects, 76 front files). Adding more subjects or recording with consistent lighting/background could help.

---

## A. Appendix: A1 Detection-Ratio Filter Fix Details

### The Bug
The old NaN filter in `dataset_v2.py` checked `nan_ratio > 0.3` on each window, but linear interpolation was already applied during `load_skeleton_csv()` — removing all NaNs before windowing. The filter never triggered.

### The Fix
Replaced with a **detection-ratio filter**: windows with fewer than 70% detected frames (`pose_detected == 1`) are skipped. This required reading `pose_detected` from the CSV during `load_skeleton_csv()` and propagating it through the pipeline.

### Impact
| Dataset | Before | After | Change |
|---------|:------:|:-----:|:------:|
| Front | 3,280 | 3,028 | -7.7% |
| Front+45 | 8,166 | 7,074 | -13.4% |
| Front-only ONNX accuracy | 70.64% | 62.58% | -8.06 pp |
| Phase 1 ONNX combined | 55.71% | 50.48% | -5.23 pp |

### All fix.txt Bugs Fixed (14 total)

| ID | File | Fix |
|----|------|-----|
| A1 | `dataset_v2.py`, `test_onnx.py` | Detection-ratio filter replaces dead NaN filter |
| A2 | `kfold_cv.py` | Rewrite with `--group_by video/subject` + `--n_cameras` |
| A4 | `mirror_10joint.py` | Redundant `mid_hip_x` negate removed |
| A5 | `model_v2.py` | `_init_weights` iterates `modules()` correctly |
| A6 | `train_v2.py` | Resume state properly loads optimizer/epoch |
| A7 | `train_v2.py` | WeightedRandomSampler applied correctly |
| A8 | `train_v2.py` | `torch.amp` usage fixed |
| B4 | `infer_onnx.py` | ONNX Runtime streaming inference |
| B5 | `add_foot_hold_gaps.py` | CLI `label_dir` arg + idempotency |
| B6 | `export_onnx.py` | Auto-detect project root for paths |
