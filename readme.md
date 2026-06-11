![FERN Banner](assets/banner.svg)

# FERN — Foot gEsture Recognition Network

Real-time foot gesture recognition from a standard RGB camera using MediaPipe skeleton extraction and a CNN-BiLSTM-Attention classifier.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00897B?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Alpha-orange?style=flat-square)]()

---

## Overview

FERN works on any standard RGB camera — no depth sensor, no wearable, no special rig. It extracts 33-joint pose skeletons via MediaPipe, normalizes them to body-relative coordinates, and feeds 60-frame sliding windows into a compact CNN-BiLSTM-Attention network that classifies 7 foot gestures at ~20 ms/window on an RTX 3070. CPU inference is also supported.

The pipeline is fully modular: swap the camera, retrain on new subjects, or extend the gesture set without touching the model architecture.

---

## Architecture

```
RGB Video Frames
       │
       ▼
  MediaPipe PoseLandmarker
  33 joints × (x, y, z)
       │
       ▼  normalize to mid-hip + torso length
  30 features / frame
       │
       ▼  sliding window  T=60, stride=15
  ┌──────────────────────┐
  │  CNN-1D              │  local joint-pair patterns
  │  BiLSTM              │  bidirectional temporal encoding
  │  Attention           │  frame-level relevance weighting
  └──────────────────────┘
       │
       ▼
  Softmax → 7-class label + confidence
```

**823,239 parameters.** Trains in under 30 min on a laptop GPU.

---

## Gesture Classes

| ID | Label | Description |
|:--:|:------|:------------|
| 0 | `foot_lift` | Lift foot straight up |
| 1 | `sideway_kick` | Kick foot laterally |
| 2 | `cross_front` | Cross foot in front of body |
| 3 | `heel_tap` | Tap heel to ground |
| 4 | `flamingo_bend` | Single-leg balance with bend |
| 5 | `forward_step` | Step forward |
| 6 | `forward_kick` | Kick forward |

---

## Results — Alpha Checkpoint

| Metric | Value |
|:-------|------:|
| Test Accuracy (Top-1) | **93.67%** |
| Test Accuracy (Top-3) | **98.73%** |
| Best Validation Accuracy | 98.78% |
| Skeleton Detection Rate | 100.0% |
| Subjects | 35 |
| Model Parameters | 823,239 |
| Training Device | RTX 3070 Laptop (8 GB) |

**Per-class F1:**

| Gesture | Precision | Recall | F1 |
|:--------|----------:|-------:|---:|
| `foot_lift` | 100% | 100% | **100%** |
| `cross_front` | 100% | 100% | **100%** |
| `flamingo_bend` | 100% | 91.7% | **95.7%** |
| `forward_step` | 88.2% | 100% | **93.8%** |
| `sideway_kick` | 100% | 77.8% | **87.5%** |
| `heel_tap` | 70.0% | 100% | **82.4%** |
| `forward_kick` | 100% | 71.4% | **83.3%** |

---

## Requirements

- Python 3.11
- PyTorch 2.2.2 + CUDA 12.1 *(CPU-only supported)*
- MediaPipe 0.10.14
- OpenCV 4.9+
- NumPy 1.26.4
- ffmpeg *(dataset tooling only)*

Full list: [`requirements_v2.txt`](requirements_v2.txt)

**MediaPipe pose model** (~30 MB, download once):

```bash
# Place at: ~/.cache/mediapipe/models/pose_landmarker_heavy.task
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

---

## Quick Start

```bash
git clone https://github.com/Vision-Orchestration/FERN
cd FERN
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# macOS / Linux
# source venv/bin/activate

pip install -r requirements_v2.txt

# Webcam inference
python src/infer_v2.py --model models/fern_v2_best.pth --camera_id 0

# Video file inference
python src/infer_v2.py --model models/fern_v2_best.pth --camera_id path/to/video.mp4
```

---

## Training Pipeline

### 1 — Label videos

```bash
python src/label_videos.py \
    --video_dir  data/raw \
    --label_dir  data/labels \
    --gestures   foot_lift sideway_kick cross_front heel_tap flamingo_bend forward_step forward_kick \
    --reps       3
```

| Key | Action |
|:---:|:-------|
| `S` | Mark start |
| `E` | Mark end |
| `W` | Save label |
| `R` | Undo |
| `A` / `D` | ±5 frames |
| `B` / `F` | ±50 frames |
| `Q` | Quit without saving |

### 2 — Extract skeletons

```bash
python src/extract_skeleton.py \
    --video_dir  data/raw \
    --output_dir data/skeletons
```

### 3 — Train

```bash
python src/train_v2.py \
    --skeleton_dir data/skeletons \
    --label_dir    data/labels \
    --output_dir   models \
    --epochs       100 \
    --batch_size   32 \
    --device       cuda
```

### 4 — Evaluate

```bash
python src/evaluate_v2.py \
    --model        models/fern_v2_best.pth \
    --skeleton_dir data/skeletons \
    --label_dir    data/labels
```

---

## Dataset Augmentation

**Left-right mirror** — doubles the dataset without re-recording:

```bash
python src/mirror_dataset.py \
    --skeleton_dir data/skeletons \
    --label_dir    data/labels \
    --output_skel  data/skeletons \
    --output_label data/labels
```

**Import FERN v1 clips:**

```bash
python src/merge_v1_database.py \
    --v1_dir     data/v1_clips \
    --output_dir data/merged_v1 \
    --gap_frames 5
```

Expects `v1_clips/<gesture_name>/<clip_files>`. Outputs merged videos and auto-generated label JSONs.

---

## Roadmap

### Alpha *(current)*
- [x] MediaPipe skeleton extraction pipeline
- [x] CNN-BiLSTM-Attention architecture
- [x] Interactive labeling tool with S/E single-pass marking
- [x] 93.67% test accuracy — 35 subjects
- [ ] LR mirror augmentation
- [ ] FERN v1 database merge
- [ ] Idle / null class
- [ ] Window onset offset fix
- [ ] Subject-independent evaluation split

### Beta
- [ ] 45° second camera (feature-level fusion, 60 → 120 input features)
- [ ] 3D reconstruction via stereo triangulation *(paper ablation)*
- [ ] DroidGrid phone camera integration
- [ ] Confidence smoothing via temporal majority vote

### Gold
- [ ] Full augmentation (rotation ±5°, brightness, mirror)
- [ ] Ablation study: single-cam vs fusion vs 3D
- [ ] Paper draft: methodology, results, related work

### Release
- [ ] Submission to IEEE Sensors Journal / MDPI Sensors
- [ ] Open-source weights + live demo
- [ ] Dataset release (skeleton CSVs, de-identified)

---

## Repository Structure

```
FERN/
├── assets/
│   └── banner.svg
├── src/
│   ├── model_v2.py            # CNN-BiLSTM-Attention architecture
│   ├── dataset_v2.py          # Sliding-window dataset loader
│   ├── train_v2.py            # Training loop (cosine LR + early stopping)
│   ├── evaluate_v2.py         # Per-class metrics and confusion matrix
│   ├── extract_skeleton.py    # MediaPipe PoseLandmarker extraction
│   ├── infer_v2.py            # Live inference (webcam or video file)
│   ├── label_videos.py        # Interactive frame-level labeling tool
│   ├── mirror_dataset.py      # LR skeleton augmentation
│   └── merge_v1_database.py   # FERN v1 clip importer
├── models_final/              # Final checkpoints
├── models_final_v2/           # v2 checkpoints
├── models_sweep/              # Hyperparameter sweep outputs
├── data/
│   ├── raw/                   # Raw videos (not tracked)
│   ├── skeletons/             # Extracted CSVs (not tracked)
│   └── labels/                # Label JSONs (not tracked)
├── requirements_v2.txt
├── run_nightly.ps1
└── README.md
```

---

## Citation

```bibtex
@misc{fern2026,
  title  = {FERN: Real-Time Foot Gesture Recognition via
            MediaPipe Skeleton and CNN-BiLSTM-Attention},
  author = {Mohasebati, Morteza},
  year   = {2026},
  note   = {Alpha release},
  url    = {https://github.com/Vision-Orchestration/FERN}
}
```

---

## License

Released under the [MIT License](LICENSE).
