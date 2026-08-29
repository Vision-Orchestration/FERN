# FERN v2 — Project Knowledge

## Project Structure
- `src/model_v2.py` — CNN-only (lstm_hidden=0) or BiLSTM model; supports `input_features` param
- `src/dataset_v2.py` — Sliding window dataset; supports `n_cameras` for one-hot camera-ID flag; 7 on-the-fly augmentations
- `src/train_v2.py` — Training loop, AdamW, cosine LR, warmup, early stopping, `--n_cameras` and `--resume` args
- `src/infer_v2.py` — Live camera inference with MediaPipe
- `src/export_onnx.py` — Export `.pth` → `.onnx` + numerical validation; auto-reads `input_features` from checkpoint
- `src/test_onnx.py` — Full-dataset accuracy + confusion matrix via ONNX Runtime; supports `--n_cameras`
- `src/kfold_cv.py` — 5-fold CV (partially executed)
- `src/augment_rotate.py` — Rotation augmentation (±5°, ±10°, ±15°) with optional mirrors
- `src/augment_spatial.py` — Spatial augmentation (scaling, vertical shift, joint perturbation)
- `src/augment_interpolate.py` — Temporal interpolation for smooth gestures
- `src/add_foot_hold_gaps.py` — Insert 60-frame foot_hold gaps at gesture transitions
- `src/mirror_10joint.py` — X-flip mirror augmentation
- `data/skeletons/front/` — 88 CSVs (44 orig + 44 mirror), z=0, front camera only
- `data/labels/front/` — 88 JSONs, camera_id=0
- `data/skeletons/front_aug/` — 484 augmented CSVs (rotation + spatial + interpolation)
- `data/labels/front_aug/` — 484 augmented JSONs
- `data/skeletons/front_plus_45/` — 98 CSVs (76 front + 22 c2), z=0, camera_id in JSONs
- `data/labels/front_plus_45/` — 98 JSONs (c3=0, c2=1)
- `data/skeletons/merged_v1/` — All raw extracted skeletons including c4 files
- `data/labels/merged_v1/` — Label JSONs (camera_ids NOT set for c4)
- `final/models/` — Old front-only model `.pth` + `.onnx` (cnn_out=64, 62.58%)
- `final_v2/models/` — Phase 1 (c3+c2 with n_cameras=2) model `.pth` + `.onnx` (50.48%)
- `sweep/models/` — Optimal config model `.pth` + `.onnx` (cnn_out=128, dropout=0.3, 86.29%)

## Best Config (sweep optimal)
```
cnn_out=128, lstm_hidden=0, dropout=0.3
lr=3e-4, weight_decay=1e-2, batch_size=32
window_size=60, stride=15
warmup_epochs=20, label_smoothing=0.1
moderate aug: time_warp ±10%, joint_dropout 20%, noise σ=0.005
```

## Hyperparameter Sweep (3-fold CV, front-only, subject-level)
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

## Key Commands
```powershell
# Train optimal config (front-only, cnn_out=128, dropout=0.3)
python src/train_v2.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --output_dir sweep --log_dir sweep/logs --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.3 --cnn_out 128 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 0 --train_all

# Train Phase 1 (c3+c2 with camera-ID flag)
python src/train_v2.py --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --output_dir final_v2 --log_dir final_v2/logs --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 0 --n_cameras 2 --train_all

# Resume training from checkpoint (loads saved args for model construction)
python src/train_v2.py ... --resume final_v2/models/fern_v2_latest.pth

# Export to ONNX (auto-reads num_classes, cnn_out, dropout from checkpoint)
python src/export_onnx.py --checkpoint_path final_v2/models/fern_v2_latest.pth --output_path final_v2/models/fern_v2.onnx

# Test ONNX accuracy + per-camera breakdown
python src/test_onnx.py --onnx_path final_v2/models/fern_v2.onnx --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --n_cameras 2 --window_size 60 --stride 15

# 3-fold CV (front-only)
python src/kfold_cv.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --epochs 50 --warmup_epochs 15 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --k_folds 3 --device cuda --num_workers 0 --group_by subject

# Live inference (PyTorch)
python src/infer_v2.py --model sweep/models/fern_v2_latest.pth --camera_id 0 --window_size 60 --stride 10 --threshold 0.5

# Live inference (ONNX, no PyTorch)
python src/infer_onnx.py --onnx_path sweep/models/fern_v2.onnx --camera_id 0 --camera_view 0 --threshold 0.5

# Use YAML config instead of CLI args
python src/train_v2.py --config configs/train_config.yaml
```

## Experiment Results
| Exp | Dataset | Arch | cnn_out | Params | Best Val Acc | Notes |
|-----|---------|------|---------|--------|-------------|-------|
| BL | all | BiLSTM | 32 | 207K | 24.59% | Baseline LSTM |
| F0 | front | BiLSTM | 32 | 207K | 24.80% | |
| 1 | combined | BiLSTM | 32 | 207K | 35.12% | |
| 2 | front | CNN-only | 32 | 33K | 74.10% | First CNN win |
| 3 | front | CNN-only | 64 | 132K | 80.22% | Pre-A1 fix peak |
| 4a | front | CNN-only | 64 | 132K | 78.06% | Seed sweep |
| 4b | front | CNN-only | 64 | 132K | 80.22% | Seed sweep |
| 5 | combined | CNN-only | 64 | 132K | 38.29% | No camera flag |
| 6 | front | CNN-only | 64 | 132K | 79.50% | |
| 7 | front | CNN-only | 96 | 296K | 79.14% | Diminishing returns |
| **S** | **front** | **CNN-only** | **128** | **526K** | **86.29%** | **Sweep optimal (train_all ONNX)** |
| T1 | front+45 | CNN-only | 64 | 132K | 60.14% (val) | Post-fix val |
| P1 | front+45 | CNN-only | 64 | 140K | 55.71% | Camera-ID flag |

## Production Models
| Model | Path | Trained On | Input Dim | Params | Front Acc |
|-------|------|-----------|:---------:|:------:|:---------:|
| Old front-only | `final/models/fern_v2.onnx` | front (76 files) | 30 | 132K | 62.58% |
| **Sweep optimal** | **`sweep/models/fern_v2.onnx`** | **front (76 files)** | **30** | **526K** | **86.29%** |
| Phase 1 (flag) | `final_v2/models/fern_v2.onnx` | front+45 (98 files) | 32 | 140K | 50.48% |

## Key Findings
1. **LSTM hurts**: CNN-only outperforms BiLSTM by 45%+ on this small dataset
2. **Front-only**: Combined camera angles drop accuracy from 80% → 38%
3. **cnn_out=128 is best**: Sweep found cnn_out=128 + dropout=0.3 beats cnn_out=64 + dropout=0.6 by +3.25 pp CV (44.36% vs 41.11%). ONNX train_all: **86.29%** vs 62.58%.
4. **Data scarcity is primary bottleneck**: CV improved only +3.25 pp (44.36% vs 41.11%) while train_all ONNX jumped +23.71 pp (86.29% vs 62.58%). Bigger model memorizes better but doesn't generalize — need more subjects.
5. **Unbiased CV estimate**: ~44.4% (3-fold subject-level) with optimal config. True generalization for front-only.
6. **Augmentation**: Moderate is best (time warp ±10%, joint dropout 20%, noise σ=0.005)
7. **Geometric transform FAILED**: R_y(-θ) using MediaPipe z cannot recover front-view — angle sweep 0-90° all 14-15% (random). See `FERN_TRANSFORM_AGENT.md`.
8. **Camera-ID flag WORKS**: One-hot flag (+2 dims) gives +6.96 pp on c2 (49.43% → 56.39%) with only -1.15 pp on front (55.85% → 54.70%). Combined gain: +3.70 pp (52.01% → 55.71%).
9. **c4 excluded**: All 11 c4 files <30% detection. Best (p05_c4) has 28.4% detection → only 98 usable windows. Not viable for training.
10. **num_workers=0 on Windows**: DataLoader with `num_workers > 0` hangs on Windows. Always use `--num_workers 0`.

## Critical Context
- **CUDA env**: `torch 2.2.2+cu121`, `onnxruntime-gpu 1.26.0`, RTX 3070 Laptop (8.6 GB)
- **protobuf lock**: mediapipe 0.10.35 requires protobuf <5.0.0
- **z=0**: All skeleton CSVs have z=0 (eliminates MediaPipe/YOLO depth conflict)
- **Foot_hold gaps**: 60-frame segments inserted at every gesture-group transition
- **Mirror pairing**: Originals and mirrors stay paired in same split (train/val/test)
- **Confidence threshold**: 0.5 recommended for production (lowered from 0.6 to reduce uncertain frames)
- **Camera-ID encoding**: c3=[1,0], c2=[0,1] — appends 2 columns → input (T, 32)
- **Input feature dim**: 30 base + (n_cameras if >1 else 0) = 32 for Phase 1
- **num_workers=0 on Windows**: DataLoader multiprocessing hangs with SubsetRandomSampler. Always use `--num_workers 0`.
- **train_all also saves best**: `--train_all` now saves the final checkpoint as both `_latest.pth` and `_best.pth` for consistent model file contract.
- **Resume loads saved args**: Model construction during `--resume` reads cnn_out, lstm_hidden, lstm_layers, dropout, input_features from the checkpoint's args dict instead of CLI defaults.
- **ONNX auto-reads more params**: Export now also reads `num_classes` and `dropout` from checkpoint args (not just `input_features` and `cnn_out`).
- **Config loader available**: `config_loader.py` loads YAML configs and merges with CLI args. See `configs/train_config.yaml`.

## Resources
- Full report: `FERN_v2_COMPLETE_REPORT.md`
- Sweep optimal checkpoint: `sweep/models/fern_v2_latest.pth` (526K params, 86.29% ONNX)
- Sweep optimal ONNX: `sweep/models/fern_v2.onnx` (86.29% front-only)
- Phase 1 (flag) checkpoint: `final_v2/models/fern_v2_latest.pth`
- Phase 1 ONNX: `final_v2/models/fern_v2.onnx` (50.48% combined)
- Old front-only ONNX: `final/models/fern_v2.onnx` (62.58%)
- Sweep logs: `sweep_*.log`
- Camera flag plan: `CAMERA_FLAG_AGENT.md`
- Transform attempt: `docs/FERN_TRANSFORM_AGENT.md`
