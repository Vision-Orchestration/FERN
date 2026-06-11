<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=28&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&width=600&lines=FERN+v2;Foot+Gesture+Recognition+Network" alt="FERN" />

<br/>

**Extract MediaPipe skeletons from standard RGB cameras and classify 8 foot gestures in real time — no depth sensor, no wearables, no special rig.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00897B?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005BBB?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## Overview

FERN turns any webcam into a foot-gesture controller. It extracts 33-joint pose skeletons via MediaPipe, normalises them to body-relative coordinates, and runs a compact CNN (~132K params) on 60-frame sliding windows — producing an 8-class gesture prediction every ~20 ms on an RTX 3070.

The pipeline is fully modular: swap the camera, retrain on new subjects, or extend the gesture set without touching the model architecture.

| | Feature | Detail |
|---|---|---|
| 🎯 | **Real-time inference** | ~20 ms/window on GPU, CPU-supported |
| 📷 | **Any RGB camera** | Webcam, phone (via DroidCam), or video file |
| 🧠 | **CNN-only** | BiLSTM was evaluated and dropped — CNN outperforms at this dataset scale |
| 🔀 | **Multi-angle** | Per-frame camera-ID flag lets one model handle multiple angles |
| 🏷️ | **Auto-labeling** | Recording assistant generates label JSONs during capture — no manual annotation |
| 🔄 | **Data augmentation** | Mirror, time warp, joint dropout, noise injection |

---

## Gestures (8 classes)

All gestures are performed with the **right foot**.

| ID | Class | Description |
|:--:|-------|------------|
| 0 | `foot_hold` | Standing still / idle — no gesture |
| 1 | `foot_lift` | Lift foot straight up |
| 2 | `sideway_kick` | Kick foot laterally |
| 3 | `cross_front` | Cross foot in front of body |
| 4 | `heel_tap` | Tap heel to ground |
| 5 | `flamingo_bend` | Single-leg balance with knee bend |
| 6 | `forward_step` | Step forward |
| 7 | `forward_kick` | Kick forward |

> `foot_hold` serves as the idle/null class — without it the model forces predictions on non-gesture frames. It needs dedicated diverse footage, not just transition padding.

---

## Architecture

```
RGB video frames
      │
      ▼
MediaPipe PoseLandmarker  ──→  33 body joints × (x, y, z, visibility)
      │
      ▼
Normalise to Mid-Hip + Torso Length  ──→  30 features per frame
      │
      ▼
[optional] Camera-ID one-hot flag  ──→  30 + N features
      │
      ▼
Sliding window (60 frames, stride 15)
      │
      ▼
CNN1D  ──→  local spatial-temporal patterns
      │
      ▼
Softmax  ──→  8-class label + confidence
```

**~132K parameters.** Runs real-time on CPU; ~20 ms/window on RTX 3070 Laptop.

---

## Current Results

| Metric | Value |
|--------|-------|
| Single-split test accuracy | ~80% |
| 5-fold cross-validation (CV) | ~60% (subject-independent) |
| Model parameters | ~132K |
| Architecture | CNN-only |
| Training device | RTX 3070 Laptop (8 GB) / Ryzen 7 5800H |
| Primary camera | c3 (front, 0°) |

**Known weak classes:**
- `heel_tap` — inherently weak from front view; side camera recommended
- `foot_hold` — weak when only recorded as transition padding

---

## Camera Setup

FERN supports multi-angle capture with a single camera-conditioned model.

| Camera | Angle | Position | Status |
|--------|-------|----------|--------|
| c3 | 0° (front) | Ground level | ✓ Training baseline |
| c4 | ~45° (right) | Ground level | ✓ Active |
| c2 | ~90° (left) | Ground level | ✓ Active |
| c1 | Elevated | — | ✗ Excluded — breaks normalisation |
| c5 | — | — | ✗ Excluded — insufficient subjects |

**Camera-ID flag:** A per-frame one-hot vector is appended to skeleton features so one model learns angle-conditioned recognition across all cameras. Geometric rotation via MediaPipe z-depth was evaluated and failed (~15% accuracy). Stereo triangulation is designed but requires physical calibration.

---

## Quick Start

### 1. Clone & install

```powershell
git clone https://github.com/Vision-Orchestration/FERN
cd FERN
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements_v2.txt
$env:PYTHONPATH = "$(Get-Location)\src"
```

### 2. Get the pose model

Download [`pose_landmarker_heavy.task`](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task) (~30 MB) and place it at:

```
C:\Users\<user>\.cache\mediapipe\models\pose_landmarker_heavy.task
```

### 3. Run live inference

```powershell
# Webcam
python src\infer_v2.py --model models_sweep\fern_v2.onnx --camera_id 0

# Video file
python src\infer_v2.py --model models_sweep\fern_v2.onnx --camera_id "path\to\video.mp4"
```

---

## Training Pipeline

### 1. Record dataset

```powershell
python src\recording_assistant.py
```

The recording assistant provides a fullscreen tkinter UI with countdown/GO/REST cues, stick-figure gesture illustrations, DroidGrid REST API integration for multi-camera sync, and auto-generated label JSONs. Target: **20 subjects**, varied height and footwear.

### 2. Extract skeletons

```powershell
python src\extract_skeleton.py --video_dir data\raw --output_dir data\skeletons
```

### 3. Train

```powershell
python src\train_v2.py --skeleton_dir data\skeletons --label_dir data\labels --output_dir models --epochs 100 --batch_size 32 --device cuda
```

### 4. Evaluate

```powershell
python src\evaluate_v2.py --model models\fern_v2_best.pth --skeleton_dir data\skeletons --label_dir data\labels
```

---

## Augmentation Tools

| Tool | Command | Effect |
|------|---------|--------|
| **LR Mirror** | `python src\mirror_dataset.py ...` | Doubles dataset via X-flip |
| **V1 Merge** | `python src\merge_v1_database.py ...` | Imports FERN v1 clip database |

> Subject-aware splits required — mirrored pairs must stay in the same train/val/test fold.

---

## DroidGrid Integration

FERN uses [DroidGrid](https://github.com/Vision-Orchestration/DroidGrid) for multi-camera capture. Phones stream video via RTSP → MediaMTX broker → laptop, with pass-through recording per camera and live inference overlay on both laptop and phone displays.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `R` | Start recording (recording assistant) |
| `S` | Stop recording |
| `Q` | Quit |
| `H` | Toggle HUD overlay (inference) |

---

## Roadmap

### Alpha (current)
- [x] MediaPipe skeleton extraction
- [x] CNN-only model (~132K params)
- [x] Auto-labeling recording assistant
- [x] Multi-camera setup (c3, c4, c2)
- [x] Camera-ID one-hot flag design
- [ ] 20-subject dataset recording
- [ ] LR mirror augmentation
- [ ] Idle class with dedicated footage
- [ ] Subject-independent 5-fold CV as primary metric

### Beta
- [ ] Camera-flag model on multi-angle data
- [ ] Stereo triangulation (requires calibration)
- [ ] Confidence smoothing (temporal majority vote)

### Gold
- [ ] Full augmentation suite
- [ ] Ablation study
- [ ] Paper draft

### Release
- [ ] Paper submission
- [ ] Open-source weights + demo
- [ ] Dataset release

---

## File Structure

```
FERN/
├── src/
│   ├── model_v2.py              # CNN-only architecture
│   ├── dataset_v2.py            # Sliding-window dataset with camera-flag
│   ├── train_v2.py              # Training loop (cosine LR + early stopping)
│   ├── evaluate_v2.py           # Per-class metrics + confusion matrix
│   ├── extract_skeleton.py      # MediaPipe skeleton extraction
│   ├── infer_v2.py              # Live inference (webcam / video file)
│   ├── recording_assistant.py   # Recording UI with DroidGrid integration
│   ├── mirror_dataset.py        # LR skeleton augmentation
│   └── merge_v1_database.py     # FERN v1 clip merger
├── models_final/                # v1 checkpoints
├── models_final_v2/             # v2 checkpoints
├── models_sweep/                # Hyperparameter sweep outputs
├── AGENTS.md                    # OpenCode agent instructions
├── CAMERA_FLAG_AGENT.md         # Camera-flag implementation
├── FERN_v2_COMPLETE_REPORT.md   # Full technical report
├── run_nightly.ps1              # Nightly training script
├── requirements_v2.txt
└── README.md
```

---

## Key Design Decisions

| Decision | Outcome |
|----------|---------|
| BiLSTM evaluated and dropped | CNN outperforms at current dataset scale |
| MediaPipe z-depth rotation | Failed (~15% accuracy) — z too noisy for single-camera transforms |
| Auto-generated labels | Eliminates manual annotation errors |
| Camera-ID one-hot flag | Single model handles multiple angles |
| Stereo triangulation | Designed; needs physical calibration |
| Early stopping warmup guard | Required — val_loss vs val_acc mismatch caused false stops |

---

## Troubleshooting

**ONNX inference is slow**
- Make sure you're using the GPU build: `pip install onnxruntime-gpu`
- Check that ONNX Runtime sees your CUDA device

**Training hangs on Windows**
- Use `--num_workers 0` — DataLoader multiprocessing hangs with `SubsetRandomSampler`

**Low accuracy**
- Start with front-camera (c3) data only before adding multi-angle
- Use hyperparameter sweep results: `cnn_out=128`, `dropout=0.3`, `lr=3e-4`

**Camera shows no detection**
- Check `data/skeletons/` CSVs exist and have valid joint coordinates
- Verify the pose model `.task` file is in the correct cache path

---

## FAQ

**Q: What hardware do I need?**  
Any laptop with a webcam. CUDA GPU recommended for training (RTX 3070 or better), but CPU training and inference work.

**Q: How many subjects are in the dataset?**  
Currently ~10 subjects for front camera. Target is 20+ for generalisation.

**Q: Can I use this with DroidCam phones?**  
Yes — the recording assistant integrates with DroidGrid for multi-phone capture.

**Q: Why CNN and not BiLSTM?**  
Extensive eval showed CNN-only outperforming BiLSTM by 45%+ on this dataset size. BiLSTM will be revisited with more data.

**Q: How do I contribute data?**  
Record with `recording_assistant.py`, run the skeleton extraction and training pipeline, then submit a PR with your skeletons and labels.

---

## Contributing

Issues and pull requests are welcome.

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit with a clear message
4. Open a pull request

---

## Citation

```bibtex
@misc{fern2026,
  title   = {FERN: Real-Time Foot Gesture Recognition via MediaPipe Skeleton and CNN},
  author  = {Vision-Orchestration},
  year    = {2026},
  url     = {https://github.com/Vision-Orchestration/FERN}
}
```

---

## License

MIT

---

<div align="center">

Part of the [Vision-Orchestration](https://github.com/Vision-Orchestration) toolkit.

*MediaPipe skeletons + CNN. No depth sensor. No wearables. Just a camera.*

</div>
