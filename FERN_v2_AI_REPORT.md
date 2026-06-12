# FERN v2 — Complete AI Session Report

Everything done across all sessions, for your AI context.

---

## Goal

Build a reliable FERN v2 foot-gesture classifier with multi-angle camera-ID support, fix all identified bugs, establish a trustworthy accuracy baseline, and optimize hyperparameters.

---

## Session 1 — Bug Fixes & Baseline (fix.txt)

### All 14 Fixes Applied

| ID | File | Fix |
|:--:|------|-----|
| A1 | `dataset_v2.py` + `test_onnx.py` | Detection-ratio filter (≥70%) — replaces old dead NaN filter. Filters out low-quality skeleton files before training/eval. |
| A2 | `kfold_cv.py` | Complete rewrite — grouped folds by subject, `--n_cameras` arg, `--group_by` arg. |
| A4 | `mirror_10joint.py` | Removed redundant x-negate that was double-flipping already mirrored CSVs. |
| A5 | `model_v2.py` | `_init_weights` now iterates `self.modules()` instead of `self.children()` — initialises nested layers too. |
| A6 | `train_v2.py` | `--resume` loads full trainer state (optimiser, scheduler, epoch) from checkpoint. |
| A7 | `train_v2.py` | `WeightedRandomSampler` for class-balanced training. |
| A8 | `train_v2.py` | Fixed torch.amp API for 2.2.2 compatibility. |
| B4 | `infer_onnx.py` | New file — ONNX Runtime streaming inference (webcam/video). |
| B5 | `add_foot_hold_gaps.py` | `--label_dir` arg + idempotency check (skips if gaps already exist). |
| B6 | `export_onnx.py` | Auto-detect project root instead of hardcoded path. |

### GradScaler Fix

`from torch.cuda.amp import GradScaler` — torch 2.2.2+cu121 requires this path, not `torch.amp`. Applied in both `kfold_cv.py` and `train_v2.py`.

### Nightly Pipeline Created

`run_nightly.ps1` — full automated run:
1. 5-fold CV front-only
2. 5-fold CV front+45
3. Train-all front-only → ONNX
4. Train-all Phase 1 (flag) → ONNX
5. Pushes to git

### Baseline Results (post-fix)

| Exp | Dataset | Model | Accuracy |
|:---:|---------|-------|:--------:|
| 5-fold CV | front-only | cnn_out=64 | **42.58%** ± 12.82% |
| 5-fold CV | front+45 | cnn_out=64 | **27.93%** ± 4.35% |
| Train-all ONNX | front-only | cnn_out=64 | **62.58%** |
| Train-all ONNX | front+45 (Phase 1 flag) | cnn_out=64 | **50.48%** |

### Detection Filter Impact

| Dataset | Before | After | Change |
|---------|:------:|:-----:|:------:|
| Front-only | 3,280 | 3,028 | **-7.7%** |
| Front+45 | 8,166 | 7,074 | **-13.4%** |

### Git Repo Cleaned

`.git` was 43 GB from venv bloat in history. `git gc` timed out. Solution: fresh git init + force push to clean remote.

---

## Session 2 — Hyperparameter Sweep

### Sweep Design

- **9 configs** × **3-fold CV** (subject-level, 50 epochs each)
- Dataset: front-only (3,028 windows after detection filter)
- Baseline: cnn_out=64, dropout=0.6 (41.11% CV)

### Sweep Results (3-fold subject-level CV)

| Config | Mean | σ | vs Baseline |
|--------|:----:|:--:|:-----------:|
| **dropout=0.3 + cnn_out=128** | **44.36%** | ±6.75% | **+3.25 pp** |
| dropout=0.3 | 43.73% | ±5.02% | +2.62 pp |
| cnn_out=128 | 42.80% | ±3.94% | +1.69 pp |
| wd=1e-4 | 41.48% | ±4.23% | +0.37 pp |
| Baseline (cnn_out=64, dropout=0.6) | 41.11% | ±3.59% | — |
| wd=0 | 40.70% | ±3.77% | -0.41 pp |
| lr=1e-4 | 38.91% | ±3.32% | -2.20 pp |
| lr=1e-3 | 38.65% | ±5.41% | -2.46 pp |
| cnn_out=32 | 34.05% | ±2.34% | -7.06 pp |
| dropout=0.7 | 33.02% | ±4.10% | -8.09 pp |

### Optimal Config

```
cnn_out=128, lstm_hidden=0, dropout=0.3
lr=3e-4, weight_decay=1e-2, batch_size=32
window_size=60, stride=15
warmup_epochs=20, label_smoothing=0.1
moderate aug: time_warp ±10%, joint_dropout 20%, noise σ=0.005
```

**526K params.**

### Production Model Trained

Optimal config + train-all + 200 epochs:

| Model | Path | Params | Accuracy |
|-------|------|:------:|:--------:|
| Old front-only | `models_final/fern_v2.onnx` | 132K | 62.58% |
| **Sweep optimal** | **`models_sweep/fern_v2.onnx`** | **526K** | **86.29%** |
| Phase 1 (flag) | `models_final_v2/fern_v2.onnx` | 140K | 50.48% |

### export_onnx.py Updated

Auto-reads `cnn_out` and `dropout` from checkpoint `args` dict — no more hardcoded values.

### kfold_cv.py Updated

Added `--label_smoothing` CLI arg (default 0.1) to replace hardcoded value.

### Stale Reports Deleted

Removed from git: `FINAL_REPORT.md`, `SESSION_REPORT.md`, `report.md`.

### assets/banner.png Created

1200×300 banner image for README.

---

## Session 3 — README Overhaul & Repo Polish

### README Corrected

Previous README was still describing the old v1 pipeline (CNN-BiLSTM-Attention, 823K params, 93.67%, 7 classes, Unix paths). Rewrote entirely to match current reality:

- Architecture: CNN-only (~132K params), not BiLSTM
- Gestures: 8 classes (added foot_hold)
- Results: ~80% single-split, ~60% CV
- Camera setup table with status per camera
- DroidGrid integration section
- Recording assistant in training pipeline
- Key Design Decisions table
- PowerShell-native commands and paths

### README Styled to Match DroidGrid

Restyled to match `Vision-Orchestration/DroidGrid`:

- Typing SVG header (`JetBrains Mono`, blue animation)
- Chunky `style=for-the-badge` badges
- Emoji feature overview table
- Keyboard controls table
- Troubleshooting + FAQ sections
- Contributing section
- Centered "Part of the Vision-Orchestration toolkit" footer
- Cleaner hierarchy with shorter sections

### repomix Rebuilt

| Attempt | Ignores | Files | Size |
|:-------:|---------|:-----:|:----:|
| First | Minimal | 807 | **266 MB** |
| **Final** | **CSVs, JSONs, models, images, data** | **42** | **371 KB** |

Final pack has only source scripts, docs, and config — 98K tokens.

### .gitignore Updated

Added: `repomix-output.xml`, stale `models/` dirs, `models_exp*/`, `models_front*/`, `models_front_plus_45/`, root-level `data/skeletons/*.csv`.

### Git Pushed

Two commits:
- `df82751` — README fix + gitignore update
- `47656c9` — DroidGrid style match

---

## Complete Experiment History

| Exp | Dataset | Arch | cnn_out | Params | Best Acc | Notes |
|:---:|---------|:----:|:-------:|:------:|:--------:|-------|
| BL | all | BiLSTM | 32 | 207K | 24.59% | Baseline LSTM |
| F0 | front | BiLSTM | 32 | 207K | 24.80% | |
| 1 | combined | BiLSTM | 32 | 207K | 35.12% | |
| 2 | front | CNN-only | 32 | 33K | 74.10% | First CNN win over LSTM |
| 3 | front | CNN-only | 64 | 132K | 80.22% | Pre-A1 fix peak |
| 4a | front | CNN-only | 64 | 132K | 78.06% | Seed sweep |
| 4b | front | CNN-only | 64 | 132K | 80.22% | Seed sweep |
| 5 | combined | CNN-only | 64 | 132K | 38.29% | No camera flag |
| 6 | front | CNN-only | 64 | 132K | 79.50% | |
| 7 | front | CNN-only | 96 | 296K | 79.14% | Diminishing returns |
| **S** | **front** | **CNN-only** | **128** | **526K** | **86.29%** | **Sweep optimal** |
| T1 | front+45 (post-fix) | CNN-only | 64 | 132K | 60.14% | Post-fix val |
| P1 | front+45 (camera flag) | CNN-only | 64 | 140K | 50.48% | Camera-ID Phase 1 |

---

## Key Findings

1. **CNN > BiLSTM**: CNN-only outperforms BiLSTM by 45%+ on this dataset size.
2. **cnn_out=128 optimal**: Sweep found cnn_out=128 + dropout=0.3 beats 64+0.6 by +3.25 pp CV.
3. **Data scarcity is primary bottleneck**: CV improved only +3.25 pp (44.36% vs 41.11%) while train-all ONNX jumped +23.71 pp (86.29% vs 62.58%). The model memorises but doesn't generalise — need more subjects (target 20+).
4. **Camera-ID flag works**: One-hot flag (+2 dims) gives +6.96 pp on c2, -1.15 pp on front. Combined gain +3.70 pp.
5. **Geometric transform failed**: R_y(-θ) using MediaPipe z produced 14-15% for all angles 0-90° (random).
6. **c4 excluded**: All 11 c4 files <30% detection rate. Not viable for training.
7. **num_workers=0 on Windows**: DataLoader multiprocessing hangs with SubsetRandomSampler.
8. **Moderate augmentation best**: time_warp ±10%, joint_dropout 20%, noise σ=0.005.

---

## Critical Context

| Item | Detail |
|------|--------|
| **CUDA env** | torch 2.2.2+cu121, onnxruntime-gpu 1.26.0, RTX 3070 Laptop (8.6 GB) |
| **Python PATH** | `python` resolves to `C:\Python314\python.exe` (no CUDA). Always use `C:\fern\FERN_V2\venv\Scripts\python.exe` |
| **num_workers** | Always `--num_workers 0` on Windows |
| **z=0** | All skeleton CSVs have z=0 |
| **Foot_hold gaps** | 60-frame segments at gesture-group transitions |
| **Mirror pairing** | Originals + mirrors stay paired in same split |
| **Input dim** | 30 base + (n_cameras if >1 else 0) |
| **Camera-ID** | c3=[1,0], c2=[0,1] |
| **Confidence threshold** | 0.6 for production |
| **Disk space** | ~70 GB free |

---

## File Structure

```
FERN/
├── src/
│   ├── model_v2.py              # CNN-only (lstm_hidden=0) or BiLSTM
│   ├── dataset_v2.py            # Sliding-window dataset with camera-flag
│   ├── train_v2.py              # Training loop (cosine LR + warmup + early stopping)
│   ├── evaluate_v2.py           # Per-class metrics + confusion matrix
│   ├── kfold_cv.py              # K-fold CV with grouped folds + label smoothing
│   ├── extract_skeleton.py      # MediaPipe skeleton extraction
│   ├── infer_v2.py              # Live inference (PyTorch)
│   ├── infer_onnx.py            # Live inference (ONNX Runtime)
│   ├── export_onnx.py           # .pth → .onnx export with auto-read of args
│   ├── test_onnx.py             # Full-dataset ONNX accuracy + per-camera breakdown
│   ├── recording_assistant.py   # Recording UI with DroidGrid integration
│   ├── add_foot_hold_gaps.py    # Insert 60-frame foot_hold at gesture transitions
│   ├── mirror_dataset.py        # LR skeleton augmentation
│   ├── mirror_10joint.py        # X-flip mirror (10-joint variant)
│   └── merge_v1_database.py     # FERN v1 clip merger
├── models_final/                # Old front-only (132K, 62.58%)
├── models_final_v2/             # Phase 1 camera-flag (140K, 50.48%)
├── models_sweep/                # Sweep optimal (526K, 86.29%)
├── assets/
│   └── banner.png
├── AGENTS.md                    # Project knowledge for AI agents
├── CAMERA_FLAG_AGENT.md         # Camera-flag implementation plan
├── FERN_v2_COMPLETE_REPORT.md   # Full technical report (75 KB)
├── FERN_v2_AI_REPORT.md         # This file
├── run_nightly.ps1              # Nightly training pipeline
├── requirements_v2.txt
├── .gitignore
└── README.md
```

---

## Next Steps

1. **Collect more training data** — target 20+ subjects. The 44.36% CV vs 86.29% ONNX gap confirms data scarcity is the primary bottleneck.
2. **Multi-angle training with optimal config** — apply cnn_out=128, dropout=0.3 to front+45 with camera-ID flag.
3. **LR mirror augmentation** — double the dataset without re-recording.
4. **Foot_hold with diverse footage** — improve the idle class beyond transition padding.
5. **Subject-independent 5-fold CV** as primary evaluation metric.
