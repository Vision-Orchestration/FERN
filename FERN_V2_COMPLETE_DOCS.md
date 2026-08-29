# FERN v2 — Complete Project Documentation

## Overview

FERN v2 is a foot gesture recognition system using MediaPipe skeleton extraction (10 joints) + CNN1D-BiLSTM-Attention. Runs on Windows with CUDA, exports to ONNX for production inference.

---

## 1. Project Structure

```
FERN_V2/
├── src/
│   ├── model_v2.py              — CNN1D-BiLSTM-Attention model
│   ├── dataset_v2.py            — Sliding window dataset + 7 on-the-fly augmentations
│   ├── train_v2.py              — Training loop (AdamW, cosine LR, warmup, early stopping)
│   ├── evaluate_v2.py           — Full-dataset evaluation + confusion matrix
│   ├── infer_v2.py              — Live camera inference (PyTorch)
│   ├── infer_onnx.py            — Live camera inference (ONNX, no PyTorch)
│   ├── export_onnx.py           — Export .pth → .onnx + numerical validation
│   ├── test_onnx.py             — Full-dataset ONNX accuracy + per-camera breakdown
│   ├── kfold_cv.py              — 5-fold cross-validation (subject-level grouping)
│   ├── extract_skeleton.py      — MediaPipe skeleton extraction from video
│   ├── mirror_10joint.py        — X-flip mirror augmentation
│   ├── mirror_skeletons.py      — DEPRECATED (use mirror_10joint.py)
│   ├── add_foot_hold_gaps.py    — Insert 60-frame foot_hold gaps at transitions
│   ├── fix_labels.py            — Fix label files (add camera_id, keep-neutral)
│   ├── label_videos_v3.py       — Interactive video labeling tool
│   ├── refine_labels.py         — Refine gesture boundaries using motion
│   ├── merge_v1_db.py           — Merge old v1 database
│   ├── augment_rotate.py        — Rotation augmentation (±5°, ±10°, ±15°)
│   ├── augment_spatial.py       — Spatial augmentation (scale, shift, noise)
│   ├── augment_interpolate.py   — Temporal interpolation for slow gestures
│   ├── transform_skeleton.py    — Geometric rotation (R_y) for angled cameras
│   ├── config_loader.py         — YAML config loader
│   └── __init__.py              — Package init
├── configs/
│   └── train_config.yaml        — Training hyperparameters (YAML)
├── data/
│   ├── skeletons/
│   │   ├── front/               — 88 CSVs (44 subjects × 2 mirror)
│   │   ├── front_aug/           — 484 augmented CSVs (rotation + spatial + interp)
│   │   ├── front_plus_45/       — 98 CSVs (front + c2, camera-ID flag)
│   │   └── merged_v1/           — All raw extracted skeletons
│   ├── labels/
│   │   ├── front/               — 88 JSONs
│   │   ├── front_aug/           — 484 augmented JSONs
│   │   ├── front_plus_45/       — 98 JSONs (c3=0, c2=1)
│   │   └── merged_v1/           — All raw label JSONs
│   └── raw videos/
│       ├── front/               — Original recording videos
│       ├── front/labeled/       — Completed labeled videos
│       └── 45 from right/       — 45° angle recordings
├── models/                      — Current model checkpoint + ONNX
├── sweep/                       — Sweep optimal model + logs
├── final/                       — Old front-only model
├── final_v2/                    — Phase 1 (camera flag) model
├── configs/train_config.yaml    — YAML config file
├── requirements_v2.txt          — Python dependencies
├── run.bat                      — End-user launcher
├── run_nightly.ps1              — Nightly training pipeline
├── FERN_v2.spec                 — PyInstaller build spec
├── AGENTS.md                    — Claude agent knowledge base
├── LICENSE                      — MIT license
├── CHAMBER_SUMMARY.md           — Chamber audit summary
└── .gitignore                   — Git ignore rules
```

---

## 2. Gesture Classes (8 classes)

| ID | Gesture | Description |
|:--:|---------|-------------|
| 0 | `foot_hold` | Idle/neutral stance (person standing naturally) |
| 1 | `foot_lift` | Lifting foot up |
| 2 | `sideway_kick` | Kicking to the side |
| 3 | `cross_front` | Crossing leg in front |
| 4 | `heel_tap` | Tapping heel |
| 5 | `flamingo_bend` | Bending like flamingo |
| 6 | `forward_step` | Stepping forward |
| 7 | `forward_kick` | Kicking forward |

---

## 3. Data Pipeline

### 3.1 Recording → Skeletons → Labels → Training

```
Video (.mp4)
    │
    ▼ extract_skeleton.py
Skeleton CSV (frame_idx, pose_detected, 10 joints × 3 coords)
    │
    ▼ mirror_10joint.py (X-flip augmentation)
Skeleton CSV (original + mirror pair)
    │
    ▼ label_videos_v3.py (interactive labeling)
Label JSON (segments with gesture, start_frame, end_frame)
    │
    ▼ add_foot_hold_gaps.py (insert foot_hold transitions)
Label JSON (with foot_hold segments between gestures)
    │
    ▼ augment_rotate.py + augment_spatial.py + augment_interpolate.py
Augmented CSVs + JSONs (484 new files from 88 originals)
    │
    ▼ train_v2.py
Model checkpoint (.pth)
    │
    ▼ export_onnx.py
ONNX model (.onnx) for production inference
```

### 3.2 Skeleton CSV Format

Each CSV has 36 columns:
- `frame_idx` — frame number
- `pose_detected` — 1 if MediaPipe detected pose, 0 otherwise
- 10 joints × 3 coords (x, y, z) = 30 feature columns:
  - `left_hip_x/y/z`, `right_hip_x/y/z`
  - `left_knee_x/y/z`, `right_knee_x/y/z`
  - `left_ankle_x/y/z`, `right_ankle_x/y/z`
  - `left_heel_x/y/z`, `right_heel_x/y/z`
  - `left_foot_index_x/y/z`, `right_foot_index_x/y/z`
- `mid_hip_x/y/z` — hip center
- `torso_length` — torso measurement

All z-coordinates are 0 (front camera only, MediaPipe depth eliminated).

### 3.3 Label JSON Format

```json
{
  "video_path": "data/raw videos/front/video.mp4",
  "fps": 29.9,
  "total_frames": 5454,
  "segments": [
    {"gesture": "foot_lift", "start_frame": 5, "end_frame": 50},
    {"gesture": "sideway_kick", "start_frame": 80, "end_frame": 130},
    ...
  ]
}
```

### 3.4 Dataset File Counts

| Directory | Files | Description |
|-----------|:-----:|-------------|
| `data/skeletons/front/` | 88 | Original + mirror (44 subjects × 2) |
| `data/labels/front/` | 88 | Matching labels |
| `data/skeletons/front_aug/` | 484 | Augmented (rotation + spatial + interpolation) |
| `data/labels/front_aug/` | 484 | Matching augmented labels |
| `data/skeletons/front_plus_45/` | 98 | Multi-camera (front + 45° side) |
| `data/labels/front_plus_45/` | 98 | With camera_id (0=front, 1=side) |

---

## 4. Data Augmentation

### 4.1 On-the-Fly Augmentations (training time)

Applied in `dataset_v2.py` `_augment()` method. Only active during training (`augment=True`), disabled during validation/testing.

| Augmentation | Probability | Parameters | Purpose |
|-------------|:-----------:|-----------|---------|
| Time warp | 100% | ±10% scale | Temporal speed variation |
| Speed variation | 40% | 80-120% resample | Different movement speeds |
| Frame dropout | 30% | 5-10% of frames zeroed | Simulate missed detections |
| Temporal shift | 30% | ±5-10 frames crop/pad | Window alignment robustness |
| Joint dropout | 20% | Zero out 1 joint (3 coords) | Missing joint robustness |
| Joint noise | 100% | σ=0.005 Gaussian | Sensor noise simulation |
| Random scaling | 30% | 0.9-1.1 factor | Distance variation |

### 4.2 Offline Augmentations (pre-generated)

Applied once to generate new CSV files in `data/skeletons/front_aug/`.

#### Rotation Augmentation (`augment_rotate.py`)

Applies small Y-axis rotations to simulate slight camera angle variation.

```powershell
python src/augment_rotate.py \
  --input_dir  data/skeletons/front \
  --label_dir  data/labels/front \
  --output_skel data/skeletons/front_aug \
  --output_label data/labels/front_aug \
  --angles 5 10 15 \
  --mirror
```

- Input: 88 front CSVs
- Output: 88 × 3 angles × 2 (mirror) = 528 new files
- Uses `transform_skeleton.py` `rotation_y()` for the geometric transform
- Each rotation is a realistic camera angle simulation (±5°, ±10°, ±15°)

#### Spatial Augmentation (`augment_spatial.py`)

Applies scaling, vertical shift, and joint perturbation.

```powershell
python src/augment_spatial.py \
  --input_dir  data/skeletons/front \
  --label_dir  data/labels/front \
  --output_skel data/skeletons/front_aug \
  --output_label data/labels/front_aug \
  --variants 2 \
  --mirror
```

- Input: 88 front CSVs
- Output: 88 × 2 variants × 2 (mirror) = 352 new files
- Transforms:
  - Random scaling: 0.9-1.1 (simulates distance variation)
  - Random vertical shift: ±5% (simulates camera height)
  - Random joint perturbation: σ=0.01 (more aggressive than training-time σ=0.005)

#### Interpolation Augmentation (`augment_interpolate.py`)

Doubles frame count by linear interpolation between consecutive frames.

```powershell
python src/augment_interpolate.py \
  --input_dir  data/skeletons/front \
  --label_dir  data/labels/front \
  --output_skel data/skeletons/front_aug \
  --output_label data/labels/front_aug \
  --factor 2 \
  --gestures foot_hold flamingo_bend
```

- Input: 88 front CSVs
- Output: 88 × 1 = 88 new files (with adjusted frame indices)
- Adjusts label segment frame indices to match new frame count
- Best for slow, smooth gestures (foot_hold, flamingo_bend)

### 4.3 Augmentation Summary

| Method | Type | New Files | Total After |
|--------|------|:---------:|:-----------:|
| Original + mirror | — | — | 88 |
| Rotation (3 angles + mirrors) | offline CSV | 264 | 352 |
| Spatial (2 variants + mirrors) | offline CSV | 176 | 528 |
| Interpolation (2x frame interp) | offline CSV | 44 | 572 |
| **Effective training data** | | | **~6.5×** |

Plus 7 on-the-fly augmentations at training time.

---

## 5. Model Architecture

### FERNv2 (CNN1D-BiLSTM-Attention)

```python
FERNv2(
    num_joints=10,
    num_classes=8,
    cnn_out=128,        # CNN output channels
    lstm_hidden=0,      # 0 = CNN-only (best for small datasets)
    lstm_layers=1,
    dropout=0.3,
    input_features=30,  # 10 joints × 3 (x, y, z)
)
```

Architecture layers:
1. **Conv1D blocks** (3 layers): 1×1 → 3×1 → 5×1 convolutions with batch norm + ReLU + dropout
2. **BiLSTM** (optional, `lstm_hidden=0` disables it): captures temporal dependencies
3. **Attention**: weighted pooling over time dimension
4. **Classifier**: Linear layers → 8-class output

**Key finding**: CNN-only (`lstm_hidden=0`) outperforms BiLSTM by 45%+ on this small dataset.

### Best Config (from hyperparameter sweep)

```
cnn_out=128, lstm_hidden=0, dropout=0.3
lr=3e-4, weight_decay=1e-2, batch_size=32
window_size=60, stride=15
warmup_epochs=20, label_smoothing=0.1
early_stopping_metric: loss
```

---

## 6. Training

### 6.1 Train on Original Data (88 files)

```powershell
python src/train_v2.py \
  --skeleton_dir data/skeletons/front \
  --label_dir data/labels/front \
  --output_dir sweep \
  --epochs 200 --warmup_epochs 20 \
  --batch_size 32 --window_size 60 --stride 15 \
  --lr 3e-4 --weight_decay 1e-2 \
  --dropout 0.3 --cnn_out 128 \
  --lstm_hidden 0 --lstm_layers 1 \
  --device cuda --num_workers 0 \
  --train_all
```

### 6.2 Train on Augmented Data (484 files)

```powershell
python src/train_v2.py \
  --skeleton_dir data/skeletons/front_aug \
  --label_dir data/labels/front_aug \
  --output_dir sweep_aug \
  --epochs 200 --warmup_epochs 20 \
  --batch_size 32 --window_size 60 --stride 15 \
  --lr 3e-4 --weight_decay 1e-2 \
  --dropout 0.3 --cnn_out 128 \
  --lstm_hidden 0 --lstm_layers 1 \
  --device cuda --num_workers 0 \
  --train_all
```

### 6.3 5-Fold Cross-Validation (Unbiased Estimate)

```powershell
python src/kfold_cv.py \
  --skeleton_dir data/skeletons/front_aug \
  --label_dir data/labels/front_aug \
  --epochs 50 --warmup_epochs 15 \
  --batch_size 32 --window_size 60 --stride 15 \
  --lr 3e-4 --weight_decay 1e-2 \
  --dropout 0.3 --cnn_out 128 \
  --lstm_hidden 0 --lstm_layers 1 \
  --k_folds 5 --device cuda --num_workers 0 \
  --group_by subject
```

**Important**: `--num_workers 0` is REQUIRED on Windows. `num_workers > 0` causes DataLoader to hang.

### 6.4 Export to ONNX

```powershell
python src/export_onnx.py \
  --checkpoint_path sweep_aug/fern_v2_latest.pth \
  --output_path sweep_aug/fern_v2.onnx
```

Auto-reads `num_classes`, `cnn_out`, `dropout`, `input_features` from checkpoint args.

### 6.5 Test ONNX Model

```powershell
python src/test_onnx.py \
  --onnx_path sweep_aug/fern_v2.onnx \
  --skeleton_dir data/skeletons/front_aug \
  --label_dir data/labels/front_aug \
  --window_size 60 --stride 15
```

---

## 7. Inference

### 7.1 Live Camera (PyTorch)

```powershell
python src/infer_v2.py \
  --model sweep_aug/fern_v2_latest.pth \
  --camera_id 0 \
  --window_size 60 --stride 10 \
  --threshold 0.5
```

### 7.2 Live Camera (ONNX, no PyTorch)

```powershell
python src/infer_onnx.py \
  --onnx_path sweep_aug/fern_v2.onnx \
  --camera_id 0 --camera_view 0 \
  --window_size 60 --stride 10 \
  --threshold 0.5
```

### 7.3 End-User Launcher (run.bat)

```batch
run.bat                              # webcam 0, threshold 0.5
run.bat --camera_id 1                # second camera
run.bat --camera_id video.mp4        # run on video file
run.bat --threshold 0.80             # stricter confidence
run.bat --camera_view 1              # camera flag index (0=front, 1=side)
```

---

## 8. Labeling New Videos

### 8.1 Interactive Labeling Tool

```powershell
python src/label_videos_v3.py \
  --video_dir "data/raw videos/front" \
  --label_dir data/labels/raw_front
```

**Controls:**
| Key | Action |
|-----|--------|
| `1`-`7` | Select gesture (1=foot_lift .. 7=forward_kick) |
| `H` | Select foot_hold as current gesture |
| `S` | Mark START of selected gesture |
| `E` | Mark END of current gesture |
| `R` | Undo last mark |
| `Z` | Delete a completed segment (cycle with arrows, Enter to confirm) |
| `A/D` | Back/forward 5 frames |
| `B/F` | Back/forward 50 frames |
| `W` | Save and move to next video |
| `Q` | Quit without saving |

No segment count limits — tag what you see, any number of segments.

### 8.2 Video Types

| Type | What's in it | How to label |
|------|-------------|-------------|
| Pure foot_hold | Person stands still ~3 min | Press `H` at start, `E` at end → one long segment |
| Regular set | 7 gestures × 3 reps | Use `1`-`7` to pick gesture, `S`/`E` to mark |
| Set + foot_hold at end | Gestures + ~3 min idle at end | Same as regular, then `H` for the tail |

### 8.3 Post-Labeling Steps

After labeling, run these in order:

```powershell
# 1. Extract skeletons from labeled videos
python src/extract_skeleton.py --video_dir "data/raw videos/front" --output data/skeletons/raw_front

# 2. Mirror augmentation
python src/mirror_10joint.py --input data/skeletons/raw_front --labels data/labels/raw_front

# 3. Insert foot_hold gaps
python src/add_foot_hold_gaps.py --skeleton_dir data/skeletons/raw_front --label_dir data/labels/raw_front

# 4. Rename files to pXX_cY convention (for subject-level CV grouping)
# Example: "set and foot hold at the end.csv" → "p18_c1.csv"

# 5. Copy to training directory
copy data\skeletons\raw_front\*.csv data\skeletons\front\
copy data\labels\raw_front\*.json data\labels\front\
```

---

## 9. Camera-ID Flag System

The multi-camera flag system is fully implemented for when you have footage from multiple angles.

### How it works

- Each label JSON can have a `camera_id` field: `0` = front (c3), `1` = side (c2)
- `dataset_v2.py` one-hot encodes this: `[1,0]` for front, `[0,1]` for side
- Model input becomes `(T, 32)` instead of `(T, 30)` — 2 extra dims for the flag
- `--n_cameras 2` enables the flag in all scripts

### Training with camera flag

```powershell
python src/train_v2.py \
  --skeleton_dir data/skeletons/front_plus_45 \
  --label_dir data/labels/front_plus_45 \
  --n_cameras 2 \
  --output_dir final_v2 \
  ... (other params)
```

### Inference with camera flag

```powershell
python src/infer_v2.py --model final_v2/models/fern_v2_latest.pth --camera_view 0
python src/infer_onnx.py --onnx_path final_v2/models/fern_v2.onnx --camera_view 1
```

---

## 10. Results History

### Best Models

| Model | Config | Train Data | CV Accuracy | ONNX Accuracy |
|-------|--------|-----------|:-----------:|:-------------:|
| Sweep optimal | cnn_out=128, dropout=0.3 | front (76 orig) | 44.36% | 86.29% |
| Phase 1 (flag) | cnn_out=64, dropout=0.6 | front+45 (98) | — | 50.48% |
| After retrain + bug fix | sweep config | front (88) | **74.23%** | 91.0% |

### Key Findings

1. **LSTM hurts**: CNN-only outperforms BiLSTM by 45%+ on small datasets
2. **Front-only is better**: Combined camera angles drop accuracy from 80% → 38%
3. **cnn_out=128 is best**: +3.25 pp over baseline (44.36% vs 41.11%)
4. **Data scarcity is primary bottleneck**: More subjects > bigger model
5. **Mirror augmentation doubles data** with zero quality loss
6. **Rotation augmentation triples data** with realistic camera angle variation

---

## 11. Dependencies

```
torch==2.2.2
torchvision==0.17.2
mediapipe==0.10.14
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
opencv-python==4.9.0.80
tensorboard==2.16.2
tqdm==4.66.2
pyyaml==6.0.1
onnxruntime==1.17.1
pytest==8.1.1
pytest-cov==5.0.0
```

**Note**: `torchaudio` was removed (130MB dead weight). `opencv-contrib-python` must NOT be installed (conflicts with `opencv-python`).

---

## 12. Known Issues

1. **Windows num_workers**: DataLoader with `num_workers > 0` hangs on Windows. Always use `--num_workers 0`.
2. **Fold variance**: 5-fold CV has ±16.96% variance. Fold 3 dropped to 44.66% due to "hard" subjects.
3. **z=0**: All skeleton CSVs have z=0 (MediaPipe depth eliminated). Geometric transforms that use z are no-ops.
4. **protobuf lock**: mediapipe 0.10.35 requires protobuf <5.0.0.

---

## 13. Chamber Audit (Completed)

48 of 52 suggestions implemented across 7 phases:
- **Phase 1**: Emergency fixes (crash bugs, security, license, build)
- **Phase 2**: Label quality (validation, bounds checks, overlap detection)
- **Phase 3**: Training pipeline (resume args, label_smoothing, early stopping)
- **Phase 4**: Infrastructure (YAML config, __init__.py, nightly script)
- **Phase 5**: Deployment (model filename contract, camera_view, threshold)
- **Phase 6**: Documentation (AGENTS.md, gitignore, CHAMBER_SUMMARY.md)
- **Phase 7**: Augmentation (rotation, spatial, interpolation, on-the-fly)

Full summary: `CHAMBER_SUMMARY.md`

---

## 14. Quick Reference — Common Commands

```powershell
# Activate venv
C:\fern\FERN_V2\venv\Scripts\Activate.ps1

# Train (original data)
python src/train_v2.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --output_dir sweep --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.3 --cnn_out 128 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 0 --train_all

# Train (augmented data)
python src/train_v2.py --skeleton_dir data/skeletons/front_aug --label_dir data/labels/front_aug --output_dir sweep_aug --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.3 --cnn_out 128 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 0 --train_all

# 5-fold CV
python src/kfold_cv.py --skeleton_dir data/skeletons/front_aug --label_dir data/labels/front_aug --epochs 50 --warmup_epochs 15 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.3 --cnn_out 128 --lstm_hidden 0 --lstm_layers 1 --k_folds 5 --device cuda --num_workers 0 --group_by subject

# Export ONNX
python src/export_onnx.py --checkpoint_path sweep_aug/fern_v2_latest.pth --output_path sweep_aug/fern_v2.onnx

# Test ONNX
python src/test_onnx.py --onnx_path sweep_aug/fern_v2.onnx --skeleton_dir data/skeletons/front_aug --label_dir data/labels/front_aug --window_size 60 --stride 15

# Label videos
python src/label_videos_v3.py --video_dir "data/raw videos/front" --label_dir data/labels/raw_front

# Live inference
python src/infer_v2.py --model sweep_aug/fern_v2_latest.pth --camera_id 0 --threshold 0.5
```
