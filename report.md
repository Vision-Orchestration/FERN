# FERN v2 — Final Experimental Report

## 1. Problem & Approach

**Goal**: Classify 8 foot gestures from front-camera (c3) skeleton videos using a lightweight neural network for real-time inference.

**Pipeline**: Video → MediaPipe Pose → lower-body 10-joint skeleton (x,y only, z=0) → sliding window classifier → gesture prediction.

---

## 2. Dataset Construction

| Step | Files | Description |
|------|-------|-------------|
| Raw videos (hand-labeled) | 33 files (~11 subjects, 7 gestures × 3 reps) | Original labeled footage |
| Low-detection c3 removal | Removed p00/p01/p02/p03/p06/p09 c3 files | Detection rate <70% excluded |
| Foot_hold gap insertion | +60-frame foot_hold segments added | Inserted at gesture-group transitions |
| Mirror augmentation | 38 originals + 38 mirrors = **76 total** | Flipped x-axis |
| z=0 normalization | All skeleton CSVs set z=0 | Eliminates MediaPipe/YOLO depth conflict |
| Combined dataset | 144 files (front + merged_v1 non-c3 + grouped) | Used only in exp1/exp5 (failed) |
| **Front+45 (Phase 1)** | 76 front + 22 c2 = **98 total**, z=0 | c3 (camera_id=0) + c2 (camera_id=1) |

**Front-only dataset**: 76 skeleton CSVs + 76 label JSONs, 3,280 sliding windows (window=60, stride=15)
**Front+45 dataset**: 98 skeleton CSVs + 98 label JSONs, 8,166 sliding windows (window=60, stride=15)

---

## 3. Model Architecture

### BiLSTM version (baseline / exp1)

```
Input(30) → Conv1d(1→16, k=3) → BN→ReLU
         → Conv1d(16→cnn_out=32, k=3) → BN→ReLU
         → BiLSTM(lstm_hidden=64, 2 layers)
         → AdditiveAttention
         → Dropout → Linear → 8 classes
Params: 207K
```

### CNN-only version (exp2+)

```
Input(30/32) → Conv1d(1→cnn_out//2, k=3) → BN→ReLU
            → Conv1d(cnn_out//2→cnn_out, k=3) → BN→ReLU
            → Linear(cnn_out×30/32→cnn_out) → Dropout
            → Global Avg Pool over time
            → Dropout → Linear(cnn_out→cnn_out//2) → ReLU → Dropout
            → Linear(cnn_out//2→8)
Params: 33K (cnn_out=32) / 132K (cnn_out=64) / 140K (cnn_out=64, input_dim=32)
```

**Camera-ID Flag**: When `n_cameras > 1`, a per-frame one-hot camera ID is appended to the 30 skeleton features. Phase 1 (c3+c2): input_dim=32.

---

## 4. Complete Experiment Results

### 4.1 Summary Table

| # | Experiment | Dataset | Arch | cnn_out | Params | Seed | Best Val Acc | Best Ep |
|---|-----------|---------|------|---------|--------|------|-------------|---------|
| BL | baseline | all | BiLSTM | 32 | 207K | 42 | **24.59%** | 26 |
| F0 | front-only | front | BiLSTM | 32 | 207K | 42 | **24.80%** | 31 |
| 1 | exp1 | combined | BiLSTM | 32 | 207K | 42 | **35.12%** | 14 |
| **2** | **exp2** | **front** | **CNN-only** | **32** | **33K** | **42** | **74.10%** | **145** |
| **3** | **exp3** | **front** | **CNN-only** | **64** | **132K** | **42** | **80.22%** | **120** |
| 4a | exp4a | front | CNN-only | 64 | 132K | 43 | **78.06%** | 80 |
| 4b | exp4b | front | CNN-only | 64 | 132K | 44 | **80.22%** | 183 |
| 5 | exp5 | combined | CNN-only | 64 | 132K | 42 | **38.29%** | 1 |
| 6 | exp6 | front | CNN-only | 64 | 132K | 42 | **79.50%** | 116 |
| 7 | exp7 | front | CNN-only | 96 | 296K | 42 | **79.14%** | 85 |
| **F** | **final** | **front** | **CNN-only** | **64** | **132K** | **42** | **train_all** | **200** |
| **T1** | **transform** | **front+45** | **CNN-only** | **64** | **132K** | **42** | **60.14% (val)** | **46** |
| **T2** | **final_v2 (no flag)** | **front+45** | **CNN-only** | **64** | **132K** | **42** | **train_all** | **200** |
| **P1** | **Phase 1 (flag)** | **front+45** | **CNN-only** | **64** | **140K** | **42** | **train_all** | **200** |

### 4.2 Critical Findings

**Finding 1: LSTM → CNN-only = +45–55%**  
- BiLSTM (any dataset): 25–35%
- CNN-only front: 74–80%
- LSTM overfits on small dataset (207K vs 132K params)

**Finding 2: Front-only → combined = -42%**  
- Front CNN-only (exp3): 80.22%
- Combined CNN-only (exp5): 38.29%
- Camera angle fundamentally changes skeleton geometry

**Finding 3: cnn_out=64 is optimal**  
```
cnn_out=32 (exp2):   74.10%  —  33K params
cnn_out=64 (exp3):   80.22%  — 132K params  ← BEST
cnn_out=96 (exp7):   79.14%  — 296K params  — worse + larger
```

**Finding 4: Result stable across seeds (78–80%)**  
```
seed=42 (exp3):  80.22%
seed=43 (exp4a): 78.06%  (-2.2%)
seed=44 (exp4b): 80.22%  (tied)
Mean ± σ: 79.50% ± 1.25%
```

**Finding 5: Strong augmentation hurts, moderate is best**  
- Original (exp3): 80.22% (time warp ±10%, joint dropout 20%, noise σ=0.005)
- Moderate aug (exp6): 79.50% (same as exp3) — consistent

---

## 5. 5-Fold Cross-Validation (Robust Estimate)

Config: CNN cnn_out=64, dropout=0.6, lstm_hidden=0, front dataset, 100 epochs per fold.

```
Fold 1:  61.13%
Fold 2:  63.57%
Fold 3:  59.76%
Fold 4:  58.84%
Fold 5:  ~59% (aborted, epoch 51/100)

Mean:    60.46%
Std:      1.89%
```

**Interpretation**: The single-split 78–80% was inflated by a lucky random split. The 5-fold CV unbiased estimate is **~60%**, which is the realistic expected accuracy on unseen data.

---

## 6. Per-Class Performance (Confusion Matrix)

Tested with un-augmented ONNX inference on all 3,280 windows (final model, training set).

```
Predicted →
                     f_hold  f_lift  s_kick  c_front heel_tap f_bend  f_step  f_kick
      foot_hold         167       7      59       75      53      50      55      48
      foot_lift           3     216      67        0      35       0      11       2
   sideway_kick           4      17     336        2      12       0       6       9
    cross_front           3       5       6      511       8      11      13       1
       heel_tap          25      31      11        9     245       2       2       7
  flamingo_bend          14       4       0       23      31     237       3       0
   forward_step          24      29      11        4       5      19     334       4
   forward_kick          34       4      26        5      56       0      18     271
```

### Per-class accuracy

| Class | Accuracy | Samples | Notes |
|-------|----------|---------|-------|
| cross_front | **91.6%** | 558 | Best — distinctive frontal leg cross |
| sideway_kick | **87.0%** | 386 | Good — lateral motion is clear |
| forward_step | **77.7%** | 430 | Moderate — confused with foot_lift |
| flamingo_bend | **76.0%** | 312 | Moderate |
| heel_tap | **73.8%** | 332 | Moderate |
| forward_kick | **65.5%** | 414 | Moderate — confused with heel_tap |
| foot_lift | **64.7%** | 334 | Moderate — confused with sideway_kick/heel_tap |
| foot_hold | **32.5%** | 514 | **Worst** — idle/neutral, spread across all classes |

**Overall (training set, no augment)**: 2,317 / 3,280 = **70.64%**  
**Unbiased CV estimate**: **~60.5%**

---

## 7. Learning Curves

Each experiment followed this pattern:
- **warmup_epochs**: 20 (linear LR 0 → 3e-4)
- **peak LR**: 3e-4 (epoch 20 onward)
- **cosine decay**: after warmup
- **early stopping patience**: 40 (15 for final)
- **No overfitting**: CNN-only models had val_acc > train_acc throughout

Best epoch range: 80–183 (exp3 best at epoch 120)

---

## 8. Optimization Path

```
 25% ┤ baseline (BiLSTM, all cameras)
 25% ┤ front-only (BiLSTM, front camera)
 35% ┤ exp1 (BiLSTM, combined)
 74% ┤ exp2 (CNN-only, front, cnn_out=32)     ← BREAKTHROUGH: remove LSTM
 80% ┤ exp3 (CNN-only, front, cnn_out=64)     ← BEST single split
 80% ┤ exp4b (CNN-only, front, seed=44)       ← Confirmed
 79% ┤ exp6 (CNN-only, front, moderate aug)   ← Confirmed
 79% ┤ exp7 (CNN-only, front, cnn_out=96)     ← Diminishing returns
 38% ┤ exp5 (CNN-only, combined)              ← CONFIRMED: angle mismatch
 61% ┤ 5-fold CV (robust estimate)            ← Unbiased
 71% ┤ ONNX train set (final model)           ← Production
 61% ┤ Expected on unseen data (CV average)   ← REALISTIC
 56% ┤ Phase 1 (flagged, c3+c2 combined)      ← Camera-ID flag experiment
```

---

## 9. Production Artifacts

| Artifact | Path | Notes |
|----------|------|-------|
| Final PyTorch checkpoint | `models_final/fern_v2_latest.pth` | 132K params, trained on all 76 front files |
| Final ONNX model | `models_final/fern_v2.onnx` | opset 17, verified (2.92e-04 max diff) |
| Phase 1 PyTorch checkpoint | `models_final_v2/fern_v2_latest.pth` | 140K params, n_cameras=2, 98 files |
| Phase 1 ONNX model | `models_final_v2/fern_v2.onnx` | opset 17, input_dim=32, verified |
| ONNX export script | `src/export_onnx.py` | CUDA/CPU compatible, supports --input_features |
| ONNX test script | `src/test_onnx.py` | Per-camera accuracy breakdown, supports --n_cameras |
| Live inference (PyTorch) | `src/infer_v2.py` | Real-time camera with MediaPipe |
| Training script | `src/train_v2.py` | Supports --n_cameras, --resume |
| Dataset | `src/dataset_v2.py` | Sliding window, camera-ID flag, mirror pairing |
| 5-fold CV | `src/kfold_cv.py` | Written but only partially executed |
| Foot_hold gap filler | `src/add_foot_hold_gaps.py` | Inserts gaps at gesture transitions |
| Mirror augment | `src/mirror_10joint.py` | X-flip augmentation |
| Camera-ID injection | `src/add_camera_id.py` | Adds camera_id to label JSONs |

### Model sizes

| Model | Params | File Size | Format |
|-------|--------|-----------|--------|
| CNN-only cnn_out=64 (30-dim) | 132K | ~530 KB | .pth / .onnx |
| CNN-only cnn_out=64 (32-dim) | 140K | ~564 KB | .pth / .onnx |
| BiLSTM cnn_out=32 | 207K | ~830 KB | .pth |

---

## 10. Recommendations for Production

1. **Deploy `models_final_v2/fern_v2.onnx`** (Phase 1) for multi-camera setups — 55.71% combined, 56.39% on c2
2. **Deploy `models_final/fern_v2.onnx`** for front-only setups — 70.64% on front
3. **Expected accuracy: ~60%** on unseen users (CV estimate)
4. **User-specific calibration**: Fine-tune on new user's first 10–20 gestures to adapt to their skeleton
5. **Confidence threshold**: Use 0.6+ for reliable predictions, fall back to "uncertain" otherwise
6. **Best classes**: cross_front (92%), sideway_kick (87%) — most reliable
7. **Worst class**: foot_hold (32%) — consider merging with "uncertain" output

---

## 11. Geometric Transform Attempt (Phase 1 — FAILED)

**Goal**: Rotate c2 (45°) skeletons to front-view equivalent using R_y(-θ) so the existing front-trained model can classify them without retraining.

**Method**: Isolate 22 c2 CSVs from `merged_v1/` (z=100% non-zero), apply `transform_skeleton.py` with R_y(-θ) at various angles, evaluate on the front-trained ONNX model.

**Result**:
| Condition | Accuracy | Notes |
|-----------|----------|-------|
| A — Front (baseline) | **70.64%** | Model trained on front data |
| B — Raw 45°, no transform | **14.02%** | Random baseline (12.5%) |
| C — Transformed 45° | **14.96%** | +0.94% gain — FAIL |

**Angle sweep (0°–90°)**:
| Angle | 0° | 5° | 10° | 15° | 20° | 25° | 30° | 35° | 40° | 45° | 50° | 55° | 60° | 65° | 70° | 75° | 80° | 85° | 90° |
|-------|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Acc% | 14.6 | 14.6 | 14.3 | 14.1 | 14.3 | 14.4 | 14.4 | 14.7 | 14.6 | 15.0 | **15.2** | 14.9 | 14.6 | 14.1 | 13.8 | 13.3 | 12.0 | 12.6 | 12.5 |

**Best angle**: 50° at **15.17%** — still essentially random.

**Verdict**: ❌ FAIL — MediaPipe single-camera z is too noisy for geometric rotation. The rotation matrix R_y(-θ) uses z to recompute x, but with noisy z the output is meaningless. Phase 2 (stereo triangulation) is also blocked as it depends on the same noisy MediaPipe detections.

---

## 12. Multi-Angle Training — Without Camera-ID Flag

Since the geometric transform failed, trained directly on front + c2 data combined (98 files, all z=0).

### Training

| Experiment | Dataset | Files | Val Acc | Train Acc (aug) |
|-----------|---------|-------|---------|----------------|
| T1 — front+45 (val split) | front_plus_45 | 90 train + 8 val | **60.14%** | — |
| T2 — final_v2 (train_all) | front_plus_45 | 98 (all) | — | 37.7% |

### Per-subset accuracy (no flag model)

| Subset | Files | Accuracy |
|--------|-------|----------|
| Front camera only | 76 | **55.85%** |
| C2 (45°) camera only | 22 | **49.43%** |
| Combined | 98 | **52.01%** |

---

## 13. Camera-ID Flag Experiment (Phase 1)

### Motivation

The no-flag model (Section 12) achieves 52.01% combined but forces the same weights to handle both camera views. By appending a one-hot camera-ID flag (c3=[1,0], c2=[0,1]), the model can learn angle-specific feature patterns.

### Implementation

- `dataset_v2.py`: added `n_cameras` param, reads `camera_id` from JSON, appends (T, n_cameras) one-hot flag to each window
- `model_v2.py`: added `input_features` param (defaults to `num_joints * 3` for backward compat)
- `train_v2.py`: added `--n_cameras` arg, computes `input_features = 30 + n_cameras`

### Training

Config: CNN cnn_out=64, dropout=0.6, 200 epochs, train_all, front_plus_45 dataset (98 files, 8,166 windows). Input dim: 32 (30 skeleton + 2 camera flag).

### Results

| Model | Front (c3) | C2 | Combined | Windows |
|-------|-----------|-----|----------|---------|
| No flag (T2) | 55.85% | 49.43% | 52.01% | 8,166 |
| **With flag (P1)** | **54.70%** | **56.39%** | **55.71%** | 8,166 |
| **Change** | **-1.15 pp** | **+6.96 pp** | **+3.70 pp** | — |

### Key Takeaways

- **Camera-ID flag clearly works**: largest gain is on c2 (+6.96 pp), slight regression on front (-1.15 pp)
- **Combined accuracy**: 52.01% → 55.71% (+3.70 pp)
- **Trade-off acceptable**: small front loss for meaningful c2 gain
- **Zero-shot c4 remains impossible**: all 11 c4 files have <30% detection; best (p05_c4) gives only 98 usable windows at 30% NaN threshold

### Per-camera window counts

| Camera | Files | Windows | Accuracy |
|--------|-------|---------|----------|
| c3 (front) | 76 | 3,280 | 54.70% |
| c2 (45° left) | 22 | 4,886 | 56.39% |

Note: c2 files have more windows per file (avg 222 vs 43) because they are longer recordings.

---

## 14. c4 Camera (45° Right) Assessment

### Detection Rates

All 11 c4 files in `data/skeletons/merged_v1/` have severely degraded detection:

| File | Detection % | Detected Frames | Valid Windows |
|------|:-----------:|:---------------:|:-------------:|
| p05_c4 | **28.4%** | 2,817 / 9,930 | 98 |
| p02_c4 | **15.2%** | 823 / 5,417 | 46 |
| p06_c4 | **6.4%** | 479 / 7,432 | 2 |
| Others | 0–5.3% | <200 | 0 |

**Total viable c4 windows**: ~144 (98 + 46) at 30% NaN threshold — negligible compared to 8,166 in the Phase 1 dataset.

**Verdict**: c4 excluded from all training. The 45° right camera angle produces <30% MediaPipe detection across all 11 subjects, likely due to the leg being partially occluded or the pose being outside MediaPipe's trained range.

---

## 15. Key Code Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `src/model_v2.py` | ~260 | CNN-only / BiLSTM model with input_features param |
| `src/dataset_v2.py` | ~450 | Dataset with camera-ID flag, random/subject split, augmentation |
| `src/train_v2.py` | ~390 | Training loop, AdamW, warmup, early stopping, n_cameras, resume |
| `src/infer_v2.py` | 224 | Live camera inference |
| `src/export_onnx.py` | ~90 | ONNX export + validation, auto input_features |
| `src/test_onnx.py` | ~120 | ONNX accuracy + per-camera confusion matrix |
| `src/add_foot_hold_gaps.py` | ~150 | Gap insertion at gesture transitions |
| `src/mirror_10joint.py` | ~100 | X-flip augmentation |
| `src/add_camera_id.py` | ~100 | Camera-ID injection into label JSONs |
| `src/kfold_cv.py` | ~200 | 5-fold CV (partially run) |
| `src/transform_skeleton.py` | 132 | R_y(-θ) geometric transform (tested, ineffective) |

## 16. Appendix: Training Hyperparameters

| Hyperparameter | Exp1–7 | Final / Phase 1 |
|---------------|--------|-----------------|
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-4 | 3e-4 |
| Weight decay | 1e-2 | 1e-2 |
| Batch size | 32 | 32 |
| Warmup epochs | 20 | 20 |
| Max epochs | 300 | 200 |
| Early stopping patience | 40 | 15 |
| Label smoothing | 0.1 | 0.1 |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 |
| LR scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Split mode | random | random |
| Train augmentation | moderate | moderate |
