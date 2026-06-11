# FERN v2 — Camera-ID Flag Agent

## Status: PHASE 1 COMPLETE — c3 + c2

**Phase 1 executed** with c3 (front) + c2 (~90° left) using one-hot camera-ID flag. c4 excluded due to <30% detection on all 11 subjects.

## Results

| Model | Front (c3) | C2 | Combined |
|-------|-----------|-----|----------|
| No flag (T2) | 55.85% | 49.43% | 52.01% |
| **With flag (P1)** | **54.70%** | **56.39%** | **55.71%** |
| Change | -1.15 pp | **+6.96 pp** | **+3.70 pp** |

**Verdict**: Camera-ID flag works. Combined accuracy improved +3.70 pp. c2 gains +6.96 pp at the cost of -1.15 pp on front.

## Camera Setup (fixed — do not change)

| ID | Name | Angle | Height | View | Status |
|----|------|-------|--------|------|--------|
| 0  | c3   | 0° front | ground | foot only | ✅ Phase 1 (76 files, 3280 windows) |
| 1  | c2   | ~90° left | ground | foot only | ✅ Phase 1 (22 files, 4886 windows) |
| 2  | c4   | ~45° right | ground | foot only | ❌ EXCLUDED (all 11 files <30% detection) |
| —  | c1   | ~75° left  | HIGH   | full body | ❌ EXCLUDE (elevation breaks normalization) |
| —  | c5   | unknown    | —      | —         | ❌ EXCLUDE (only p01, p02, p06 — too few) |

**Encoding:**
```
Phase 1 (2 cameras): c3=[1,0]  c2=[0,1]   → appends 2 columns → input (T, 32)
```

## Implementation Details

### Modified Files

| File | Change |
|------|--------|
| `src/dataset_v2.py` | Added `n_cameras` param; reads `camera_id` from JSON; appends one-hot flag to windows in `__getitem__` |
| `src/model_v2.py` | Added `input_features` param (default `None` → `num_joints * 3` for backward compat) |
| `src/train_v2.py` | Added `--n_cameras` arg; computes `input_features = 30 + n_cameras`; passes to dataset and model; added `--resume` |

### New Features

1. **`--n_cameras` arg**: `1` = no flag (default, backward compat), `2` = c3+c2, `3` = c3+c2+c4
2. **`--resume` arg**: Resume training from a `.pth` checkpoint, restores optimizer state
3. **`test_onnx.py`**: Now supports `--n_cameras` with per-camera accuracy breakdown

### c4 Assessment (45° Right Camera)

All 11 c4 files have <30% MediaPipe detection. Best file (p05_c4) has 28.4% detection giving only 98 valid windows at 30% NaN threshold. Total viable c4 windows: ~144 — negligible compared to 8166 in Phase 1. c4 excluded from all training.

## Model Artifacts

| Artifact | Path | Notes |
|----------|------|-------|
| Phase 1 checkpoint | `models_final_v2/fern_v2_latest.pth` | 140K params, n_cameras=2, 200 epochs |
| Phase 1 ONNX | `models_final_v2/fern_v2.onnx` | opset 17, input_dim=32, verified |

## Commands

```powershell
# Train Phase 1
python src/train_v2.py --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --output_dir models_final_v2 --log_dir logs_final_v2 --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 4 --n_cameras 2 --train_all

# Export to ONNX
python src/export_onnx.py --checkpoint_path models_final_v2/fern_v2_latest.pth --output_path models_final_v2/fern_v2.onnx

# Evaluate per-camera accuracy
python src/test_onnx.py --onnx_path models_final_v2/fern_v2.onnx --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --n_cameras 2 --window_size 60 --stride 15
```

## What to Do Next

- **Do NOT pursue c4**: Detection is too poor (all <30%, only 144 usable windows)
- **Deploy Phase 1 model** for multi-camera setups using `models_final_v2/fern_v2.onnx`
- **Keep front-only model** (`models_final/fern_v2.onnx`) for single-camera front-only deployments
- **If c2 accuracy needs improvement**: Consider collecting more c2 labeled data (only 22 files currently)
