<div align="center">

<img src="https://raw.githubusercontent.com/Vision-Orchestration/FERN/main/assets/banner.png" alt="FERN Banner" width="100%">

# FERN — Foot and Exercise Recognition Network

**Real-time foot gesture recognition from a standard RGB camera.**  
MediaPipe skeleton extraction → CNN-BiLSTM-Attention → 7-class inference

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00897B?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Alpha-orange?style=flat-square)]()

<br>

| Test Accuracy | Top-3 Accuracy | Detection Rate | Subjects | Parameters |
|:---:|:---:|:---:|:---:|:---:|
| **93.67%** | **98.73%** | **100%** | **35** | **823K** |

</div>

---

## Overview

FERN is a real-time foot gesture recognition system that works on any standard RGB camera — no depth sensor, no wearable, no special hardware. It extracts 33-joint MediaPipe pose skeletons, normalizes them to body-relative coordinates, and feeds 60-frame sliding windows into a compact CNN-BiLSTM-Attention network that classifies 7 distinct foot gestures at ~20 ms/window on an RTX 3070 (CPU inference also supported).

The system is designed as a modular pipeline: you can swap in a new camera, retrain on new subjects, or extend the gesture vocabulary without touching the core architecture.

---

## Architecture

```
RGB Video Frames
       │
       ▼
┌─────────────────────────────┐
│   MediaPipe PoseLandmarker  │  33 body joints × (x, y, z)
└─────────────────────────────┘
       │
       ▼  Normalize to Mid-Hip + Torso Length
┌─────────────────────────────┐
│      30 features / frame    │
└─────────────────────────────┘
       │
       ▼  Sliding window  [T=60, stride=15]
┌─────────────────────────────┐
│          CNN-1D             │  Local spatial patterns between joints
│          BiLSTM             │  Bidirectional temporal motion encoding
│         Attention           │  Frame-level relevance weighting
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│    Softmax  →  7 classes    │  Gesture label + confidence score
└─────────────────────────────┘
```

**823,239 parameters.** Trains in under 30 minutes on a laptop GPU. Runs in real-time on CPU.

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

## Results

### Summary (Alpha Checkpoint)

| Metric | Value |
|:-------|------:|
| Test Accuracy (Top-1) | **93.67%** |
| Test Accuracy (Top-3) | **98.73%** |
| Best Validation Accuracy | 98.78% |
| Skeleton Detection Rate | 100.0% |
| Subjects | 35 |
| Model Parameters | 823,239 |
| Training Device | RTX 3070 Laptop (8 GB) |

### Per-Class F1 Scores

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
- PyTorch 2.2.2 + CUDA 12.1 *(CPU-only also supported)*
- MediaPipe 0.10.14
- OpenCV 4.9+
- NumPy 1.26.4
- ffmpeg *(for dataset tooling)*

Full dependency list: [`requirements_v2.txt`](requirements_v2.txt)

> **MediaPipe pose model** (~30 MB, download separately):
> ```
> https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
> ```
> Place at: `~/.cache/mediapipe/models/pose_landmarker_heavy.task`

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/Vision-Orchestration/FERN
cd FERN
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # macOS / Linux
pip install -r requirements_v2.txt

# 2a. Run live inference (webcam)
python src/infer_v2.py --model models/fern_v2_best.pth --camera_id 0

# 2b. Run inference on a video file
python src/infer_v2.py --model models/fern_v2_best.pth --camera_id path/to/video.mp4
```

---

## Training Pipeline

### Step 1 — Label Videos

```bash
python src/label_videos.py \
    --video_dir  data/raw \
    --label_dir  data/labels \
    --gestures   foot_lift sideway_kick cross_front heel_tap flamingo_bend forward_step forward_kick \
    --reps       3
```

**Labeling controls:**

| Key | Action |
|:---:|:-------|
| `S` | Mark gesture start |
| `E` | Mark gesture end |
| `W` | Save label |
| `R` | Undo |
| `A` / `D` | ±5 frames |
| `B` / `F` | ±50 frames |
| `Q` | Quit without saving |

### Step 2 — Extract Skeletons

```bash
python src/extract_skeleton.py \
    --video_dir  data/raw \
    --output_dir data/skeletons
```

### Step 3 — Train

```bash
python src/train_v2.py \
    --skeleton_dir data/skeletons \
    --label_dir    data/labels \
    --output_dir   models \
    --epochs       100 \
    --batch_size   32 \
    --device       cuda
```

### Step 4 — Evaluate

```bash
python src/evaluate_v2.py \
    --model        models/fern_v2_best.pth \
    --skeleton_dir data/skeletons \
    --label_dir    data/labels
```

---

## Dataset Augmentation

### Left-Right Mirror *(doubles dataset without re-recording)*

```bash
python src/mirror_dataset.py \
    --skeleton_dir  data/skeletons \
    --label_dir     data/labels \
    --output_skel   data/skeletons \
    --output_label  data/labels
```

### Import FERN v1 Clips

```bash
python src/merge_v1_database.py \
    --v1_dir     data/v1_clips \
    --output_dir data/merged_v1 \
    --gap_frames 5
```

Expects `v1_clips/<gesture_name>/<clip_files>`. Outputs merged long videos and auto-generated label JSONs.

---

## Roadmap

### Alpha *(current)*
- [x] MediaPipe skeleton extraction pipeline
- [x] CNN-BiLSTM-Attention architecture
- [x] Interactive labeling tool with S/E single-pass marking
- [x] 93.67% test accuracy across 35 subjects
- [ ] LR mirror augmentation
- [ ] FERN v1 database merge
- [ ] Idle / null class
- [ ] Window onset offset fix
- [ ] Subject-independent evaluation split

### Beta
- [ ] 45° second camera (feature-level fusion, 60→120 input features)
- [ ] 3D reconstruction via stereo triangulation *(paper ablation)*
- [ ] DroidGrid phone camera integration
- [ ] Confidence smoothing via temporal majority vote

### Gold
- [ ] Full augmentation pipeline (rotation ±5°, brightness, mirror)
- [ ] Ablation study: single-cam vs fusion vs 3D reconstruction
- [ ] Paper draft: methodology, results, related work

### Release
- [ ] Paper submission (IEEE Sensors Journal / MDPI Sensors)
- [ ] Open-source release with pretrained weights and live demo
- [ ] Dataset release (skeleton CSVs, de-identified)

---

## Repository Structure

```
FERN/
├── src/
│   ├── model_v2.py           # CNN-BiLSTM-Attention architecture
│   ├── dataset_v2.py         # Sliding-window dataset loader
│   ├── train_v2.py           # Training loop (cosine LR + early stopping)
│   ├── evaluate_v2.py        # Per-class metrics and confusion matrix
│   ├── extract_skeleton.py   # MediaPipe PoseLandmarker extraction
│   ├── infer_v2.py           # Live inference (webcam or video file)
│   ├── label_videos.py       # Interactive frame-level labeling tool
│   ├── mirror_dataset.py     # LR skeleton augmentation
│   └── merge_v1_database.py  # FERN v1 clip importer
├── models_final/             # Final checkpoints
├── models_final_v2/          # v2 checkpoints
├── models_sweep/             # Hyperparameter sweep outputs
├── data/
│   ├── raw/                  # Raw videos (not tracked)
│   ├── skeletons/            # Extracted CSVs (not tracked)
│   └── labels/               # Label JSONs (not tracked)
├── requirements_v2.txt
├── run_nightly.ps1           # Nightly training scheduler
├── AGENTS.md                 # Agent prompt documentation
├── CAMERA_FLAG_AGENT.md      # Camera flag agent spec
└── FERN_v2_COMPLETE_REPORT.md
```

---

## Citation

If you use FERN in your research, please cite:

```bibtex
@misc{fern2026,
  title   = {FERN: Real-Time Foot Gesture Recognition via MediaPipe Skeleton and CNN-BiLSTM-Attention},
  author  = {Mohasebati, Morteza},
  year    = {2026},
  note    = {Alpha release},
  url     = {https://github.com/Vision-Orchestration/FERN}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built by <a href="https://github.com/Vision-Orchestration">Vision-Orchestration</a></sub>
</div>
