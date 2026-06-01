# FERN v2 — Foot and Exercise Recognition Network

Real-time foot gesture recognition from a standard RGB camera.  
MediaPipe skeleton extraction → CNN-BiLSTM-Attention → 7-class classification.

**Current status:** Alpha phase — 93.67% test accuracy, 35 subjects, 100% skeleton detection rate across all recordings.

---

## Architecture

```
RGB video frames
      │
      ▼
MediaPipe PoseLandmarker  ──→  33 body joints × (x, y, z)
      │
      ▼
Normalise to Mid-Hip + Torso Length  ──→  30 features per frame
      │
      ▼
Sliding window (60 frames, stride 15)
      │
      ▼
CNN1D  ──→  local spatial patterns between joints
      │
      ▼
BiLSTM  ──→  temporal motion patterns
      │
      ▼
Attention  ──→  focus on key motion frames
      │
      ▼
Softmax  ──→  7-class gesture label + confidence
```

**823K parameters.** Runs real-time on CPU; ~20ms/window on RTX 3070.

---

## Gestures (7 classes)

| ID | Gesture | Description |
|----|---------|-------------|
| 0 | `foot_lift` | Lift foot straight up |
| 1 | `sideway_kick` | Kick foot laterally |
| 2 | `cross_front` | Cross foot in front of body |
| 3 | `heel_tap` | Tap heel to ground |
| 4 | `flamingo_bend` | Single-leg balance with bend |
| 5 | `forward_step` | Step forward |
| 6 | `forward_kick` | Kick forward |

---

## Results (Alpha checkpoint)

| Metric | Value |
|--------|-------|
| Test accuracy (Top-1) | 93.67% |
| Test accuracy (Top-3) | 98.73% |
| Validation accuracy (best epoch) | 98.78% |
| Skeleton detection rate | 100.0% |
| Subjects | 35 |
| Model parameters | 823,239 |
| Training device | RTX 3070 Laptop (8 GB) |

Per-class F1:

| Gesture | Precision | Recall | F1 |
|---------|-----------|--------|----|
| foot_lift | 100% | 100% | 100% |
| sideway_kick | 100% | 77.8% | 87.5% |
| cross_front | 100% | 100% | 100% |
| heel_tap | 70% | 100% | 82.4% |
| flamingo_bend | 100% | 91.7% | 95.7% |
| forward_step | 88.2% | 100% | 93.8% |
| forward_kick | 100% | 71.4% | 83.3% |

---

## Requirements

- Python 3.11
- PyTorch 2.2.2 + CUDA 12.1 (CPU-only also supported)
- MediaPipe 0.10.14
- OpenCV 4.9+
- NumPy 1.26.4
- ffmpeg (for dataset tools)

Full dependency list: see `requirements.txt`.

MediaPipe pose model (download separately, ~30 MB):  
Place at `~/.cache/mediapipe/models/pose_landmarker_heavy.task`  
Download: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`

---

## Quick start

```powershell
# Clone and set up
git clone https://github.com/YOUR_USERNAME/FERN_v2.git
cd FERN_v2
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Live inference (webcam)
python src\infer_v2.py --model models\fern_v2_best.pth --camera_id 0

# Live inference (video file)
python src\infer_v2.py --model models\fern_v2_best.pth --camera_id "path\to\video.mp4"
```

---

## Training pipeline

### 1. Label videos

```powershell
python src\label_videos.py `
    --video_dir   data\raw `
    --label_dir   data\labels `
    --gestures    foot_lift sideway_kick cross_front heel_tap flamingo_bend forward_step forward_kick `
    --reps        3
```

Controls: `S` = mark gesture start, `E` = mark end, `W` = save, `R` = undo,  
`A/D` = ±5 frames, `B/F` = ±50 frames, `Q` = quit without saving.

### 2. Extract skeletons

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
    --output_dir   models `
    --epochs       100 `
    --batch_size   32 `
    --device       cuda
```

### 4. Evaluate

```powershell
python src\evaluate_v2.py `
    --model        models\fern_v2_best.pth `
    --skeleton_dir data\skeletons `
    --label_dir    data\labels
```

---

## Dataset augmentation tools

### LR mirror (doubles dataset without re-recording)

```powershell
python src\mirror_dataset.py `
    --skeleton_dir  data\skeletons `
    --label_dir     data\labels `
    --output_skel   data\skeletons `
    --output_label  data\labels
```

### Import FERN v1 clips

```powershell
python src\merge_v1_database.py `
    --v1_dir     data\v1_clips `
    --output_dir data\merged_v1 `
    --gap_frames 5
```

Expects `v1_clips/<gesture_name>/<clip_files>`.  
Outputs merged long videos + auto-generated label JSONs.

---

## Roadmap

### Alpha (current)
- [x] MediaPipe skeleton extraction pipeline
- [x] CNN-BiLSTM-Attention model
- [x] Interactive labeling tool with S/E single-pass marking
- [x] 93.67% test accuracy, 35 subjects
- [ ] LR mirror augmentation
- [ ] FERN v1 database merge
- [ ] Idle/null class
- [ ] Window onset offset fix
- [ ] Subject-independent evaluation

### Beta
- [ ] 45° second camera (feature-level fusion, 60-feature input)
- [ ] 3D reconstruction via stereo triangulation (paper ablation)
- [ ] DroidGrid phone camera integration
- [ ] Confidence smoothing (temporal majority vote)

### Gold
- [ ] Full augmentation (rotation ±5°, brightness, mirror)
- [ ] Ablation study: single-cam vs fusion vs 3D
- [ ] Paper: methodology, results, related work
- [ ] Deployment package (CPU + GPU, cross-machine tested)

### Release
- [ ] Paper submission (IEEE Sensors Journal / MDPI Sensors)
- [ ] GitHub open-source with weights and demo
- [ ] Dataset release (skeleton CSVs)

---

## File structure

```
FERN_v2/
├── src/
│   ├── model_v2.py          # CNN-BiLSTM-Attention architecture
│   ├── dataset_v2.py        # Sliding window dataset
│   ├── train_v2.py          # Training loop with cosine LR + early stopping
│   ├── evaluate_v2.py       # Per-class metrics + confusion matrix
│   ├── extract_skeleton.py  # MediaPipe PoseLandmarker extraction
│   ├── infer_v2.py          # Live inference (webcam or video file)
│   ├── label_videos.py      # Interactive labeling tool
│   ├── mirror_dataset.py    # LR skeleton augmentation
│   └── merge_v1_database.py # FERN v1 clip merger
├── configs/
│   └── train_config.yaml
├── models/                  # Checkpoints (not tracked in git)
├── data/
│   ├── raw/                 # Raw videos (not tracked)
│   ├── skeletons/           # Extracted CSVs (not tracked)
│   └── labels/              # Label JSONs (not tracked)
├── logs/                    # TensorBoard logs (not tracked)
├── requirements.txt
└── README.md
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{fern2026,
  title   = {FERN: Real-Time Foot Gesture Recognition via MediaPipe Skeleton and CNN-BiLSTM-Attention},
  author  = {YOUR NAME},
  year    = {2026},
  note    = {Alpha release}
}
```

---

## License

MIT
