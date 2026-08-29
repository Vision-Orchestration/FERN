# FERN v2 — Project Knowledge

## Project Structure
- `src/model_v2.py` — CNN-only (lstm_hidden=0) or BiLSTM model; supports `input_features` param
- `src/dataset_v2.py` — Sliding window dataset; supports `n_cameras` for one-hot camera-ID flag; 7 on-the-fly augmentations (time warp, speed variation, frame dropout, temporal shift, joint dropout, joint noise, random scaling)
- `src/train_v2.py` — Training loop, AdamW, cosine LR, warmup, early stopping, `--n_cameras` and `--resume` args
- `src/infer_v2.py` — Live camera inference with MediaPipe
- `src/export_onnx.py` — Export `.pth` → `.onnx` + numerical validation; auto-reads `input_features` from checkpoint
- `src/test_onnx.py` — Full-dataset accuracy + confusion matrix via ONNX Runtime; supports `--n_cameras`
- `src/kfold_cv.py` — 5-fold CV with subject-level grouping
- `src/eval_full.py` — Full-dataset evaluation with confusion matrix + per-class metrics
- `src/learning_curve.py` — Accuracy vs # training subjects (held-out test)
- `src/tsne_viz.py` — t-SNE visualization of CNN embeddings colored by class
- `src/augment_rotate.py` — Rotation augmentation (±5°, ±10°, ±15°) with optional mirrors
- `src/augment_spatial.py` — Spatial augmentation (scaling, vertical shift, joint perturbation)
- `src/augment_interpolate.py` — Temporal interpolation for smooth gestures
- `src/add_foot_hold_gaps.py` — Insert 60-frame foot_hold gaps at gesture transitions
- `src/mirror_10joint.py` — X-flip mirror augmentation (active)
- `src/mirror_skeletons.py` — Deprecated, use mirror_10joint.py
- `src/generate_report.py` — Generate PDF paper report with all charts and figures
- `data/skeletons/front/` — 88 CSVs (44 orig + 44 mirror), z=0, front camera only
- `data/labels/front/` — 88 JSONs, camera_id=0
- `data/skeletons/front_aug/` — 418 augmented CSVs (rotation + spatial + interpolation) [held-out removed]
- `data/labels/front_aug/` — 418 augmented JSONs [held-out removed]
- `data/skeletons/front_test/` — 12 CSVs (6 held-out subjects × 2 mirror)
- `data/labels/front_test/` — 12 JSONs
- `data/skeletons/front_aug_test/` — 66 augmented CSVs (6 held-out subjects)
- `data/labels/front_aug_test/` — 66 augmented JSONs
- `results_orig/` — Original model results (train_all, confusion matrix, learning curve, t-SNE)
- `results_aug/` — Augmented model results (train_all, confusion matrix, t-SNE)
- `final/models/` — Old front-only model `.pth` + `.onnx` (cnn_out=64, 62.58%)
- `final_v2/models/` — Phase 1 (c3+c2 with n_cameras=2) model `.pth` + `.onnx` (50.48%)
- `sweep/models/` — Optimal config model `.pth` + `.onnx` (cnn_out=128, dropout=0.3, 86.29%)

## Best Config
```
cnn_out=128, lstm_hidden=0, dropout=0.3
lr=3e-4, weight_decay=1e-2, batch_size=32
window_size=60, stride=15
warmup_epochs=20, label_smoothing=0.1
```

## Latest Results (Aug 29, 2026)

### Primary Metrics
| Metric | Original | Augmented |
|--------|:--------:|:---------:|
| Train-all accuracy | 94.44% | 94.53% |
| 5-fold CV (subject-level) | 74.23% ±16.96% | 68.96% ±22.01% |
| **Held-out test (6 unseen subjects)** | **68.14%** | **66.37%** |

### Learning Curve (held-out test accuracy vs # training subjects)
| Subjects | Original | Augmented |
|:--------:|:--------:|:---------:|
| 5 | 80.97% | 80.65% |
| 10 | 84.51% | 83.10% |
| 20 | 81.86% | 83.27% |
| 30 | 87.17% | 85.06% |
| 38 | 89.38% | — |

### Per-Class Breakdown (Original Model, held-out test, 68.14%)
| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| foot_hold | 25.0% | 1.9% | 3.4% |
| foot_lift | 57.1% | 100.0% | 72.7% |
| sideway_kick | 72.7% | 100.0% | 84.2% |
| cross_front | 81.5% | 100.0% | 89.8% |
| heel_tap | 52.0% | 81.2% | 63.4% |
| flamingo_bend | 66.7% | 66.7% | 66.7% |
| forward_step | 75.7% | 82.4% | 78.9% |
| forward_kick | 60.9% | 93.3% | 73.7% |

### Key Findings
1. **CNN-only dominates BiLSTM**: +16 pp over BiLSTM baseline
2. **Width matters more than depth**: cnn_out=128 + dropout=0.3 → +3.25 pp CV
3. **Augmentation does NOT improve generalization**: CV drops 5.27 pp, held-out drops 1.77 pp
4. **Data scarcity is the primary bottleneck**: train-all ~94.5% vs held-out ~68% = 26 pp gap
5. **foot_hold collapses on unseen subjects**: 25%/0% precision — clearest overfitting signal
6. **Learning curve still climbing**: 80.97% (5 subj) → 89.38% (38 subj)
7. **Geometric transform FAILED**: R_y(-θ) using MediaPipe z cannot recover front-view
8. **Camera-ID flag WORKS**: +14.60 pp on multi-angle data
9. **num_workers=0 on Windows**: Always use `--num_workers 0`

## Held-Out Test Set
- **Subjects**: p25, p30, p35, p40, p45, p50 (frozen before any tuning)
- **Original files**: 12 (6 subjects × 2 mirror)
- **Augmented files**: 66 (6 subjects × 11 augmented variants)
- **Location**: `data/skeletons/front_test/` + `data/labels/front_test/`

## Key Commands
```powershell
# Train optimal config (front-only)
python src/train_v2.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --output_dir results_orig --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.3 --cnn_out 128 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 0 --train_all

# Evaluate on full dataset (confusion matrix + per-class)
python src/eval_full.py --model results_orig/fern_v2_latest.pth --skeleton_dir data/skeletons/front --label_dir data/labels/front --title "ORIGINAL"

# Evaluate on held-out test set
python src/eval_full.py --model results_orig/fern_v2_latest.pth --skeleton_dir data/skeletons/front_test --label_dir data/labels/front_test --title "HELD-OUT"

# 5-fold CV with subject-level grouping
python src/kfold_cv.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.3 --cnn_out 128 --k_folds 5 --device cuda --num_workers 0 --group_by subject --label_smoothing 0.1

# Learning curve (accuracy vs # subjects)
python src/learning_curve.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --test_skeleton_dir data/skeletons/front_test --test_label_dir data/labels/front_test --output results_orig/learning_curve

# t-SNE visualization
python src/tsne_viz.py --model results_orig/fern_v2_latest.pth --skeleton_dir data/skeletons/front --label_dir data/labels/front --output results_orig/tsne.png --title "t-SNE"

# Generate PDF report
python src/generate_report.py

# Live inference (PyTorch)
python src/infer_v2.py --model results_orig/fern_v2_latest.pth --camera_id 0 --window_size 60 --stride 10 --threshold 0.5

# Live inference (ONNX)
python src/infer_onnx.py --onnx_path sweep/models/fern_v2.onnx --camera_id 0 --camera_view 0 --threshold 0.5
```

## Critical Context
- **CUDA env**: `torch 2.2.2+cu121`, `onnxruntime-gpu 1.26.0`, RTX 3070 Laptop (8.6 GB)
- **Python**: 3.11.9 (venv), system has 3.14.6 but FERN requires 3.11
- **protobuf lock**: mediapipe 0.10.35 requires protobuf <5.0.0
- **z=0**: All skeleton CSVs have z=0 (eliminates MediaPipe/YOLO depth conflict)
- **num_workers=0 on Windows**: Always use `--num_workers 0`
- **opencv-contrib-python**: Must NOT be installed (conflicts with opencv-python)
- **train_all saves both**: `_latest.pth` and `_best.pth`
- **Resume loads saved args**: Model construction reads from checkpoint args dict
- **ONNX auto-reads params**: Export reads `num_classes`, `dropout`, `cnn_out`, `input_features` from checkpoint

## Resources
- Paper report: `FERN_V2_Paper_Report.pdf` (11 pages, all charts/tables)
- Paper materials: `PAPER_MATERIALS.md` (ablation table, dataset paragraph, all metrics)
- Augmentation log: `AUGMENTATION_LOG.md` (scripts, commands, file counts)
- Original model: `results_orig/fern_v2_latest.pth` (526K params, 94.44% train-all)
- Augmented model: `results_aug/fern_v2_latest.pth` (526K params, 94.53% train-all)
- Learning curve: `results_orig/learning_curve/learning_curve.json`
- t-SNE plots: `results_orig/tsne.png`, `results_aug/tsne.png`
