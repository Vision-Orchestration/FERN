# FERN v2 — Agent Instruction File
# Rev: 2026-06-12

---

## 0. Read This First

This file is the single source of truth for any agent working on FERN v2.
Before touching any file, read every section. Sections marked [LOCKED] describe
decisions that have been made empirically and must NOT be revisited without new
experimental evidence.

---

## 1. Environment

```
Root:        C:\fern\FERN_V2\
Python:      C:\fern\FERN_V2\venv\Scripts\python.exe
PYTHONPATH:  C:\fern\FERN_V2\src
Shell:       PowerShell only
GPU:         RTX 3070 Laptop (8 GB VRAM) / Ryzen 7 5800H
```

Always activate with:
```powershell
cd C:\fern\FERN_V2
$env:PYTHONPATH = "C:\fern\FERN_V2\src"
.\venv\Scripts\Activate.ps1
```

---

## 2. Current Best Model [LOCKED]

| Property | Value |
|---|---|
| Path | `sweep/models/fern_v2.onnx` |
| Checkpoint | `sweep/models/fern_v2_latest.pth` |
| Architecture | CNN-only (`lstm_hidden=0`) |
| Parameters | 526K |
| `cnn_out` | 128 |
| `dropout` | 0.3 |
| `lr` | 3e-4 |
| `weight_decay` | 1e-2 |
| `batch_size` | 32 |
| `window_size` | 60 |
| `stride` | 15 |
| `warmup_epochs` | 20 |
| `label_smoothing` | 0.1 |
| Augmentation | time_warp ±10%, joint_dropout 20%, noise σ=0.005 |
| Train-all ONNX accuracy | 86.29% (front-only) |
| 5-fold CV (subject-level) | ~44.36% ± 6.75% |

The gap between 86% and 44% is a **data scarcity problem**, not a model problem.
Do NOT attempt architecture changes until subject count reaches 20+.

---

## 3. Gesture Classes [LOCKED — ORDER MATTERS]

```python
DEFAULT_CLASSES = [
    "foot_hold",      # 0 — idle/null class
    "foot_lift",      # 1
    "sideway_kick",   # 2
    "cross_front",    # 3
    "heel_tap",       # 4
    "flamingo_bend",  # 5
    "forward_step",   # 6
    "forward_kick",   # 7
]
```

All gestures are performed with the **right foot only**.
`foot_hold` is the idle/null class — it requires dedicated diverse footage,
not just transition padding.

---

## 4. Skeleton CSV Format [LOCKED]

**36 columns total:**
```
frame_idx, pose_detected,
left_hip_x/y/z,       right_hip_x/y/z,
left_knee_x/y/z,      right_knee_x/y/z,
left_ankle_x/y/z,     right_ankle_x/y/z,
left_heel_x/y/z,      right_heel_x/y/z,
left_foot_index_x/y/z, right_foot_index_x/y/z,
mid_hip_x/y/z,        torso_length
```

Feature columns = the 30 joint x/y/z columns only.
`mid_hip_*` and `torso_length` are normalization metadata — not model input.

**z = 0 everywhere.** [LOCKED]
MediaPipe single-camera z-depth is too noisy for geometric transforms.
This was tested (Phase 1, 2026-06-06) and produced ~15% accuracy (random).
Do NOT attempt rotation transforms using MediaPipe z. Avenue is closed.

---

## 5. Camera Setup [LOCKED]

| Camera | Angle | One-Hot ID | Status |
|---|---|---|---|
| c3 | 0° front | `[1, 0]` | Active — primary training camera |
| c2 | ~90° left | `[0, 1]` | Active — Phase 1 camera flag |
| c4 | ~45° right | — | **EXCLUDED** — all 11 files <30% detection |
| c1 | elevated | — | **EXCLUDED** — elevation breaks normalization |
| c5 | unknown | — | **EXCLUDED** — too few subjects (p01, p02, p06 only) |

Camera-ID encoding when `n_cameras=2`:
- `c3 → camera_id=0 → one_hot=[1,0]`
- `c2 → camera_id=1 → one_hot=[0,1]`
- Appended as 2 columns → input shape becomes `(T, 32)`

Camera-ID flag experiment result (Phase 1):
- c2 accuracy: +6.96 pp (49.43% → 56.39%)
- c3 accuracy: -1.15 pp (55.85% → 54.70%)
- Combined: +3.70 pp (52.01% → 55.71%)

---

## 6. Data Directories

```
C:\fern\FERN_V2\data\
├── skeletons\
│   ├── front\               76 CSVs (38 orig + 38 mirror), z=0, camera_id=0
│   ├── front_plus_45\       98 CSVs (76 c3 + 22 c2), z=0
│   └── merged_v1\           raw extracted skeletons (includes excluded cameras)
└── labels\
    ├── front\               76 JSONs, camera_id=0, foot_hold gaps inserted
    ├── front_plus_45\       98 JSONs (c3→0, c2→1)
    └── merged_v1\           label JSONs (some missing camera_id)
```

**Mirror files:** `*_mirror.csv` / `*_mirror.json` — X negated, left/right joints swapped.
Mirror files carry the **same camera_id** as their original.
`mirror_10joint.py` performs the augmentation. Run on `merged_v1` data only.

**foot_hold gaps:** 60-frame segments inserted at gesture-group transitions.
`add_foot_hold_gaps.py` handles this. Already applied to `data/labels/front/`.
Must be re-applied after any new labeling session.

---

## 7. Training Commands

### Standard (front-only, optimal config, train_all)
```powershell
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/train_v2.py `
    --skeleton_dir data/skeletons/front `
    --label_dir    data/labels/front `
    --output_dir   sweep `
    --log_dir      sweep/logs `
    --epochs       200 `
    --warmup_epochs 20 `
    --batch_size   32 `
    --window_size  60 `
    --stride       15 `
    --lr           3e-4 `
    --weight_decay 1e-2 `
    --dropout      0.3 `
    --cnn_out      128 `
    --lstm_hidden  0 `
    --lstm_layers  1 `
    --device       cuda `
    --num_workers  0 `
    --train_all
```

### With camera-ID flag (front + c2)
```powershell
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/train_v2.py `
    --skeleton_dir data/skeletons/front_plus_45 `
    --label_dir    data/labels/front_plus_45 `
    --output_dir   final_v2 `
    --log_dir      final_v2/logs `
    --epochs       200 `
    --warmup_epochs 20 `
    --batch_size   32 `
    --window_size  60 `
    --stride       15 `
    --lr           3e-4 `
    --weight_decay 1e-2 `
    --dropout      0.3 `
    --cnn_out      128 `
    --lstm_hidden  0 `
    --lstm_layers  1 `
    --device       cuda `
    --num_workers  0 `
    --n_cameras    2 `
    --train_all
```

### 5-fold CV (subject-level, unbiased estimate)
```powershell
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/kfold_cv.py `
    --skeleton_dir data/skeletons/front `
    --label_dir    data/labels/front `
    --epochs       50 `
    --warmup_epochs 15 `
    --batch_size   32 `
    --window_size  60 `
    --stride       15 `
    --lr           3e-4 `
    --weight_decay 1e-2 `
    --dropout      0.3 `
    --cnn_out      128 `
    --lstm_hidden  0 `
    --k_folds      5 `
    --device       cuda `
    --num_workers  0 `
    --group_by     subject
```

**`--num_workers 0` is mandatory on Windows.** DataLoader multiprocessing hangs
with SubsetRandomSampler on Windows. This is not optional.

---

## 8. ONNX Export and Evaluation

### Export
```powershell
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/export_onnx.py `
    --checkpoint_path sweep/models/fern_v2_latest.pth `
    --output_path     sweep/models/fern_v2.onnx
```

`export_onnx.py` reads `n_cameras` and `cnn_out` from the checkpoint's saved
`args` dict. Do NOT pass `--cnn_out` manually unless the checkpoint is broken.

### Evaluate ONNX
```powershell
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/test_onnx.py `
    --onnx_path sweep/models/fern_v2.onnx `
    --skeleton_dir data/skeletons/front `
    --label_dir data/labels/front `
    --n_cameras 1
# Expected: ~86% accuracy
```

For camera-flag model, use `--n_cameras 2` and point to `front_plus_45`.

---

## 9. Model Architecture Rules [LOCKED]

- **BiLSTM is abandoned.** Consistently underperformed CNN-only at current dataset scale.
  Do NOT re-enable BiLSTM (`lstm_hidden > 0`) without a dataset of 20+ subjects.

- **`input_features` param:** When `n_cameras > 1`, `input_features = 30 + n_cameras`.
  The `FERNv2` model accepts this via `input_features` constructor arg.
  `export_onnx.py` reads it automatically from checkpoint args.

- **Do NOT resume from an existing checkpoint after changing `n_cameras`.**
  The input dimension changes (30 → 32 → 33), weights are incompatible.
  Always train from scratch for each phase.

- **Early stopping guard:** Must be disabled until after `warmup_epochs`.
  Track `val_acc` (not `val_loss`) for patience counter.

---

## 10. Known Weak Classes

| Class | Issue |
|---|---|
| `heel_tap` | Inherently weak from front view — needs c2 side camera |
| `foot_hold` | Weak when only recorded as transition padding — needs dedicated footage |

Both require more diverse training data, not architecture changes.

---

## 11. Next Priority Task

**Record 20+ new subjects** using `recording_assistant.py`, then retrain
with the optimal config from §7. Target: ~60–65% unbiased CV accuracy
(current: ~44%) for the academic paper.

Secondary: re-run camera-ID flag experiment after data expansion to assess
whether the +3.70 pp gain holds at scale.

---

## 12. Files the Agent May Edit

| File | Purpose |
|---|---|
| `src/train_v2.py` | Training loop |
| `src/dataset_v2.py` | Sliding window dataset |
| `src/model_v2.py` | CNN-only model |
| `src/export_onnx.py` | ONNX export |
| `src/test_onnx.py` | ONNX evaluation |
| `src/kfold_cv.py` | Cross-validation |
| `src/mirror_10joint.py` | Mirror augmentation |
| `src/add_foot_hold_gaps.py` | foot_hold gap insertion |
| `src/add_camera_id.py` | Stamp camera_id into label JSONs |
| `src/recording_assistant.py` | Guided recording UI (v1.2) |
| `run_nightly.ps1` | Nightly pipeline |

## 13. Files the Agent Must NOT Edit

| File | Reason |
|---|---|
| `data/**` | Runtime data — modify via scripts only |
| `sweep/models/fern_v2.onnx` | Production artifact — regenerate via export |
| `docs/FERN_TRANSFORM_AGENT.md` | Historical record — do not delete or alter |
| `CAMERA_FLAG_AGENT.md` | Experiment record |

---

## 14. Verification Checklist After Any Change

Run these in order. Stop on first failure.

```powershell
# 1. Smoke test model
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/model_v2.py

# 2. Smoke test dataset
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/dataset_v2.py `
    data/skeletons/front data/labels/front

# 3. Quick training sanity (2 epochs, CPU)
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/train_v2.py `
    --skeleton_dir data/skeletons/front --label_dir data/labels/front `
    --output_dir models_test --log_dir logs_test `
    --epochs 2 --batch_size 16 --device cpu --num_workers 0

# 4. ONNX eval on existing model
& "C:\fern\FERN_V2\venv\Scripts\python.exe" src/test_onnx.py `
    --onnx_path sweep/models/fern_v2.onnx `
    --skeleton_dir data/skeletons/front `
    --label_dir data/labels/front `
    --n_cameras 1
# Expected: ~86% accuracy
```

If step 4 returns significantly below 86%, a file was broken. Do not commit.

---

## 15. Experiment Log (Do Not Repeat These)

| Experiment | Result | Decision |
|---|---|---|
| BiLSTM (cnn_out=32, 207K) | 24.59–35.12% | Abandoned — CNN-only wins |
| CNN-only cnn_out=64 | 80.22% single-split | Current production baseline |
| CNN-only cnn_out=128 | 86.29% train-all | Current best (sweep optimal) |
| MediaPipe z rotation (Phase 1) | ~15% (random) | Closed — z too noisy |
| Stereo triangulation (Phase 2) | Not executed | Requires physical calibration session |
| Camera-ID flag c3+c2 (output: `final_v2/models/`) | +3.70 pp combined | Works — use for multi-camera |
| c4 camera inclusion | <30% detection all files | Excluded permanently |
