# FERN v2 — Foot Gesture Recognition Network

Real-time foot gesture recognition from standard RGB cameras.
MediaPipe skeleton extraction → CNN → 8-class gesture classification.

**Current status:** Active development — CNN-only model, ~80% accuracy (single-split), ~60% 5-fold CV on front-camera data. Multi-camera angle fusion in progress.

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
[optional] Camera-ID one-hot flag appended  ──→  30 + N features
      │
      ▼
Sliding window (60 frames, stride 15)
      │
      ▼
CNN1D  ──→  local spatial-temporal patterns between joints
      │
      ▼
Softmax  ──→  8-class gesture label + confidence
```

**~132K parameters.** Runs real-time on CPU; ~20 ms/window on RTX 3070 Laptop.

> **Why CNN-only?**
> BiLSTM was evaluated extensively and consistently underperformed at current dataset scale.
> CNN-only is the proven baseline for FERN v2.

---

## Gestures (8 classes)

All gestures are performed with the **right foot**.

| ID | Class            | Description                              |
|----|------------------|------------------------------------------|
| 0  | `foot_hold`      | Standing still / idle — no gesture       |
| 1  | `foot_lift`      | Lift foot straight up                    |
| 2  | `sideway_kick`   | Kick foot laterally                      |
| 3  | `cross_front`    | Cross foot in front of body              |
| 4  | `heel_tap`       | Tap heel to ground                       |
| 5  | `flamingo_bend`  | Single-leg balance with knee bend        |
| 6  | `forward_step`   | Step forward                             |
| 7  | `forward_kick`   | Kick forward                             |

> **Note:** `foot_hold` serves as the idle/null class to prevent forced predictions on non-gesture frames.
> It requires dedicated diverse footage — not just transition padding.

---

## Camera Setup

FERN v2 is designed for multi-angle capture with a single-model, camera-conditioned approach.

| Camera | Angle         | Position       | Status               |
|--------|---------------|----------------|----------------------|
| c3     | 0° (front)    | Ground level   | ✓ Training baseline  |
| c4     | ~45° (right)  | Ground level   | ✓ Active             |
| c2     | ~90° (left)   | Ground level   | ✓ Active             |
| c1     | Elevated      | —              | ✗ Excluded — breaks normalization |
| c5     | —             | —              | ✗ Excluded — insufficient subjects |

**Angle conditioning:** A per-frame camera-ID one-hot flag is appended to skeleton features so a single model learns angle-conditioned recognition across all cameras.

> **Note:** Geometric rotation via MediaPipe z-depth was evaluated (Phase 1) and failed (~15% accuracy).
> Stereo triangulation (Phase 2) is designed but requires physical camera calibration.

---

## Current Results

| Metric                        | Value                                   |
|-------------------------------|-----------------------------------------|
| Single-split test accuracy    | ~80%                                    |
| 5-fold cross-validation (CV)  | ~60% (unbiased, subject-independent)    |
| Model parameters              | ~132K                                   |
| Architecture                  | CNN-only                                |
| Training device               | RTX 3070 Laptop GPU (8 GB) / Ryzen 7 5800H |
| Primary camera used           | c3 (front, 0°)                          |

**Known weak classes:**
- `heel_tap` — inherently weak from front view; side camera recommended
- `foot_hold` — historically weak when only recorded as transition padding

---

## Requirements

- Python 3.11
- PyTorch 2.2.2 + CUDA 12.1 (CPU-only also supported)
- MediaPipe 0.10.14
- OpenCV 4.9+
- NumPy 1.26.4
- ffmpeg (for dataset and recording tools)

Full dependency list: see `requirements_v2.txt`.

**MediaPipe pose model** (download separately, ~30 MB):

```powershell
# Place the model file at:
C:\Users\<user>\.cache\mediapipe\models\pose_landmarker_heavy.task

# Download URL:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

> FERN uses the local `.task` file — **no internet connection required at inference time**.

---

## Quick Start

```powershell
# Clone and set up
git clone https://github.com/Vision-Orchestration/FERN
cd FERN
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements_v2.txt

# Set PYTHONPATH
$env:PYTHONPATH = "$(Get-Location)\src"

# Live inference (webcam)
python src\infer_v2.py --model models_final_v2\fern_v2_best.pth --camera_id 0

# Live inference (video file)
python src\infer_v2.py --model models_final_v2\fern_v2_best.pth --camera_id "path\to\video.mp4"
```

---

## Training Pipeline

### 1. Record Dataset (with Recording Assistant)

```powershell
python src\recording_assistant.py
```

The recording assistant provides:
- Fullscreen tkinter UI with countdown / GO / REST cues
- Stick-figure gesture illustrations per class
- DroidGrid REST API integration for multi-camera capture
- Auto-generated label JSONs (eliminates manual annotation)
- Crash recovery via per-gesture checkpoint saves
- Wall-clock anchors (start_sec / end_sec) for multi-camera sync

**Protocol:** 7 reps × 2 rounds (reverse order), 3 s countdown → 1.5 s GO → 1 s REST.
Target dataset: **20 subjects**, varied height and footwear.

### 2. Extract Skeletons

```powershell
python src\extract_skeleton.py `
    --video_dir  data\raw `
    --output_dir data\skeletons
```

### 3. Train

```powershell
python src\train_v2.py `
    --skeleton_dir data\skeletons `
    --label_dir    data\labels `
    --output_dir   models_final_v2 `
    --epochs       100 `
    --batch_size   32 `
    --device       cuda
```

### 4. Evaluate

```powershell
python src\evaluate_v2.py `
    --model        models_final_v2\fern_v2_best.pth `
    --skeleton_dir data\skeletons `
    --label_dir    data\labels
```

---

## Dataset Augmentation Tools

### LR Mirror (doubles dataset without re-recording)

```powershell
python src\mirror_dataset.py `
    --skeleton_dir  data\skeletons `
    --label_dir     data\labels `
    --output_skel   data\skeletons `
    --output_label  data\labels
```

> **Important:** Subject-aware splits are required — subjects recorded from both sides must not be split across train/val to prevent data leakage.

### Import FERN v1 Clips

```powershell
python src\merge_v1_database.py `
    --v1_dir     data\v1_clips `
    --output_dir data\merged_v1 `
    --gap_frames 5
```

Expects `v1_clips/<gesture_name>/<clip_files>`.
Outputs merged long videos + auto-generated label JSONs.

---

## DroidGrid Integration

FERN v2 uses **DroidGrid** as its data capture and live inference delivery layer.

- Phones stream video via RTSP → MediaMTX broker → laptop server
- FFMPEG pass-through recording per camera
- Recording assistant communicates with DroidGrid REST API to synchronise multi-camera capture
- Live inference output can be overlaid on both laptop and phone displays

See: [Vision-Orchestration/DroidGrid](https://github.com/Vision-Orchestration/DroidGrid)

---

## Roadmap

### Alpha (current)

- [x] MediaPipe skeleton extraction pipeline
- [x] CNN-only model (~132K params, proven baseline)
- [x] Auto-labeling recording assistant (v1.2) with DroidGrid integration
- [x] Multi-camera setup: c3 (front), c4 (45°), c2 (90°)
- [x] Camera-ID one-hot flag design for angle-conditioned single model
- [x] AGENTS.md instruction format for OpenCode agent delegation
- [ ] Fresh v2 dataset recording — 20 subjects, c3 + c4
- [ ] LR mirror augmentation
- [ ] FERN v1 database merge
- [ ] Idle class (`foot_hold`) with dedicated diverse footage
- [ ] Window onset offset fix (label alignment delay)
- [ ] Subject-independent 5-fold CV as primary evaluation metric

### Beta

- [ ] Camera-flag model training on multi-angle dataset
- [ ] 3D reconstruction via stereo triangulation (requires physical calibration)
- [ ] DroidGrid phone camera integration for live inference
- [ ] Confidence smoothing (temporal majority vote)
- [ ] `heel_tap` improvement via side-camera data

### Gold

- [ ] Full augmentation (brightness, mirror, minor rotation)
- [ ] Ablation study: single-cam vs camera-flag fusion vs stereo 3D
- [ ] Paper: methodology, results, related work
- [ ] Deployment package (CPU + GPU, cross-machine tested)

### Release

- [ ] Paper submission (IEEE Sensors Journal / MDPI Sensors)
- [ ] GitHub open-source release with weights and demo
- [ ] Dataset release (skeleton CSVs, label JSONs)

---

## File Structure

```
FERN/
├── src/
│   ├── model_v2.py              # CNN-only architecture
│   ├── dataset_v2.py            # Sliding window dataset with camera-flag support
│   ├── train_v2.py              # Training loop with cosine LR + early stopping
│   ├── evaluate_v2.py           # Per-class metrics + confusion matrix
│   ├── extract_skeleton.py      # MediaPipe PoseLandmarker extraction
│   ├── infer_v2.py              # Live inference (webcam or video file)
│   ├── recording_assistant.py   # Recording UI with DroidGrid integration
│   ├── mirror_dataset.py        # LR skeleton augmentation
│   └── merge_v1_database.py     # FERN v1 clip merger
├── data/
│   ├── raw/                     # Raw videos (not tracked)
│   ├── skeletons/               # Extracted CSVs (not tracked)
│   └── labels/                  # Label JSONs (not tracked)
├── models_final/                # v1 checkpoints
├── models_final_v2/             # v2 checkpoints (not tracked)
├── models_sweep/                # Hyperparameter sweep outputs
├── AGENTS.md                    # OpenCode agent instruction document
├── CAMERA_FLAG_AGENT.md         # Camera-flag implementation instructions
├── FERN_v2_COMPLETE_REPORT.md   # Full technical report
├── run_nightly.ps1              # Nightly training run script (PowerShell)
├── requirements_v2.txt
└── README.md
```

---

## Key Design Decisions

| Decision | Outcome |
|---|---|
| BiLSTM evaluated and dropped | Consistently underperformed at current dataset scale; CNN-only is the baseline |
| MediaPipe z-depth rotation (Phase 1) | Failed — ~15% accuracy; z is too noisy for single-camera geometric transforms |
| Auto-generated label JSONs | Strictly better than post-hoc annotation — eliminates labeling error |
| Camera-ID one-hot flag | Single model learns angle-conditioned recognition; avoids training separate per-camera models |
| Stereo triangulation (Phase 2) | Designed; requires physical calibration before implementation |
| Early stopping warmup guard | Required — val_loss vs val_acc mismatch caused false early stops without it |

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
