# FERN v2 — Project Knowledge

## Project Structure
- `src/model_v2.py` — CNN-only (lstm_hidden=0) or BiLSTM model; supports `input_features` param
- `src/dataset_v2.py` — Sliding window dataset; supports `n_cameras` for one-hot camera-ID flag
- `src/train_v2.py` — Training loop, AdamW, cosine LR, warmup, early stopping, `--n_cameras` and `--resume` args
- `src/infer_v2.py` — Live camera inference with MediaPipe
- `src/export_onnx.py` — Export `.pth` → `.onnx` + numerical validation; auto-reads `input_features` from checkpoint
- `src/test_onnx.py` — Full-dataset accuracy + confusion matrix via ONNX Runtime; supports `--n_cameras`
- `src/kfold_cv.py` — 5-fold CV (partially executed)
- `src/add_foot_hold_gaps.py` — Insert 60-frame foot_hold gaps at gesture transitions
- `src/mirror_10joint.py` — X-flip mirror augmentation
- `data/skeletons/front/` — 76 CSVs (38 orig + 38 mirror), z=0, front camera only
- `data/labels/front/` — 76 JSONs with foot_hold gaps inserted, camera_id=0
- `data/skeletons/front_plus_45/` — 98 CSVs (76 front + 22 c2), z=0, camera_id in JSONs
- `data/labels/front_plus_45/` — 98 JSONs (c3=0, c2=1)
- `data/skeletons/merged_v1/` — All raw extracted skeletons including c4 files
- `data/labels/merged_v1/` — Label JSONs (camera_ids NOT set for c4)
- `models_final/` — Front-only model `.pth` + `.onnx`
- `models_final_v2/` — Phase 1 (c3+c2 with n_cameras=2) model `.pth` + `.onnx`

## Best Config (exp3)
```
cnn_out=64, lstm_hidden=0, dropout=0.6
lr=3e-4, weight_decay=1e-2, batch_size=32
window_size=60, stride=15
warmup_epochs=20, label_smoothing=0.1
moderate aug: time_warp ±10%, joint_dropout 20%, noise σ=0.005
```

## Key Commands
```powershell
# Train on all data (production model, front-only)
python src/train_v2.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --output_dir models_final --log_dir logs_final --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 4 --train_all

# Train Phase 1 (c3+c2 with camera-ID flag)
python src/train_v2.py --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --output_dir models_final_v2 --log_dir logs_final_v2 --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 4 --n_cameras 2 --train_all

# Resume training from checkpoint
python src/train_v2.py ... --resume models_final_v2/fern_v2_latest.pth

# Export to ONNX
python src/export_onnx.py --checkpoint_path models_final_v2/fern_v2_latest.pth --output_path models_final_v2/fern_v2.onnx

# Test ONNX accuracy + per-camera breakdown
python src/test_onnx.py --onnx_path models_final_v2/fern_v2.onnx --skeleton_dir data/skeletons/front_plus_45 --label_dir data/labels/front_plus_45 --n_cameras 2 --window_size 60 --stride 15

# 5-fold CV
python src/kfold_cv.py --skeleton_dir data/skeletons/front --label_dir data/labels/front --epochs 100 --warmup_epochs 15 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.6 --cnn_out 64 --k_folds 5 --device cuda --num_workers 4

# Live inference
python src/infer_v2.py --model models_final/fern_v2_latest.pth --camera_id 0 --window_size 60 --stride 10 --threshold 0.6
```

## Experiment Results
| Exp | Dataset | Arch | cnn_out | Params | Best Val Acc |
|-----|---------|------|---------|--------|-------------|
| BL | all | BiLSTM | 32 | 207K | 24.59% |
| F0 | front | BiLSTM | 32 | 207K | 24.80% |
| 1 | combined | BiLSTM | 32 | 207K | 35.12% |
| **2** | **front** | **CNN-only** | **32** | **33K** | **74.10%** |
| **3** | **front** | **CNN-only** | **64** | **132K** | **80.22%** |
| 4a | front | CNN-only | 64 | 132K | 78.06% |
| 4b | front | CNN-only | 64 | 132K | 80.22% |
| 5 | combined | CNN-only | 64 | 132K | 38.29% |
| 6 | front | CNN-only | 64 | 132K | 79.50% |
| 7 | front | CNN-only | 96 | 296K | 79.14% |
| Final | front | CNN-only | 64 | 132K | train_all |
| T1 | front+45 | CNN-only | 64 | 132K | **60.14%** (val) |
| T2 | front+45 | CNN-only | 64 | 132K | train_all |
| **P1** | **front+45** | **CNN-only** | **64** | **140K** | **train_all** (n_cameras=2) |

## Production Models
| Model | Path | Trained On | Input Dim | Front Acc | C2 Acc | Combined Acc |
|-------|------|-----------|-----------|-----------|--------|-------------|
| Front-only | `models_final/fern_v2.onnx` | front (76 files) | 30 | **70.64%** | 14.02% | 52.01% |
| Front+45 (no flag) | `models_final_v2/fern_v2.onnx` (old) | front+45 (98 files) | 30 | 55.85% | 49.43% | 52.01% |
| **Phase 1 (flag)** | `models_final_v2/fern_v2.onnx` (new) | front+45 (98 files) | 32 | 54.70% | **56.39%** | **55.71%** |

## Key Findings
1. **LSTM hurts**: CNN-only outperforms BiLSTM by 45%+ on this small dataset
2. **Front-only**: Combined camera angles drop accuracy from 80% → 38%
3. **cnn_out=64 is optimal**: 32→64 gains +6%, 64→96 loses
4. **Stable across seeds**: 78–80% (mean 79.5%, σ=1.25%)
5. **Unbiased CV estimate**: ~60.5% (5-fold CV)
6. **Augmentation**: Moderate is best (time warp ±10%, joint dropout 20%, noise σ=0.005)
7. **Geometric transform FAILED**: R_y(-θ) using MediaPipe z cannot recover front-view — angle sweep 0-90° all 14-15% (random). See `FERN_TRANSFORM_AGENT.md`.
8. **Camera-ID flag WORKS**: One-hot flag (+2 dims) gives +6.96 pp on c2 (49.43% → 56.39%) with only -1.15 pp on front (55.85% → 54.70%). Combined gain: +3.70 pp (52.01% → 55.71%).
9. **c4 excluded**: All 11 c4 files <30% detection. Best (p05_c4) has 28.4% detection → only 98 usable windows. Not viable for training.

## Critical Context
- **CUDA env**: `torch 2.2.2+cu121`, `onnxruntime-gpu 1.26.0`, RTX 3070 Laptop (8.6 GB)
- **protobuf lock**: mediapipe 0.10.35 requires protobuf <5.0.0
- **z=0**: All skeleton CSVs have z=0 (eliminates MediaPipe/YOLO depth conflict)
- **Foot_hold gaps**: 60-frame segments inserted at every gesture-group transition
- **Mirror pairing**: Originals and mirrors stay paired in same split (train/val/test)
- **Confidence threshold**: 0.6 recommended for production
- **Camera-ID encoding**: c3=[1,0], c2=[0,1] — appends 2 columns → input (T, 32)
- **Input feature dim**: 30 base + (n_cameras if >1 else 0) = 32 for Phase 1

## Resources
- Full report: `report.md`
- Phase 1 checkpoint: `models_final_v2/fern_v2_latest.pth`
- Phase 1 ONNX: `models_final_v2/fern_v2.onnx`
- Front-only checkpoint: `models_final/fern_v2_latest.pth`
- Front-only ONNX: `models_final/fern_v2.onnx`
- Camera flag plan: `CAMERA_FLAG_AGENT.md`
- Transform attempt: `docs/FERN_TRANSFORM_AGENT.md`
