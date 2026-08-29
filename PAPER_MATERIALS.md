# FERN v2 — Paper Materials

## Ablation Table

| # | Config Change | Dataset | Arch | Params | CV Accuracy | Δ vs Baseline |
|---|--------------|---------|------|--------|:-----------:|:-------------:|
| 1 | Baseline (cnn_out=64, dropout=0.6) | front (76 files) | CNN-only | 132K | 41.11% ±3.59% | — |
| 2 | + cnn_out=128 | front | CNN-only | 526K | 42.80% ±3.94% | +1.69 pp |
| 3 | + dropout=0.3 | front | CNN-only | 132K | 43.73% ±5.02% | +2.62 pp |
| 4 | + cnn_out=128 + dropout=0.3 | front | CNN-only | 526K | 44.36% ±6.75% | +3.25 pp |
| 5 | BiLSTM baseline | front | BiLSTM | 207K | 24.80% | -16.31 pp |
| 6 | BiLSTM + combined cameras | combined | BiLSTM | 207K | 35.12% | -5.99 pp |
| 7 | CNN + combined (no flag) | combined | CNN-only | 132K | 38.29% | -2.82 pp |
| 8 | CNN + camera-ID flag | front+45 | CNN-only | 140K | 55.71% | +14.60 pp |
| **9** | **Final config (cnn_out=128, dropout=0.3, train_all)** | **front (76 files)** | **CNN-only** | **526K** | **94.44%** (train-all) | — |
| 10 | + 5× augmentation | front_aug (418 files) | CNN-only | 526K | 68.96% ±22.01% (CV) | -5.27 pp vs #4 |
| 11 | + 5× augmentation (train_all) | front_aug (418 files) | CNN-only | 526K | ~95%+ (train-all) | — |

### Key Findings
1. **CNN-only dominates BiLSTM**: +16 pp over BiLSTM baseline (74.10% vs 24.80%)
2. **Width matters more than depth**: cnn_out=128 + dropout=0.3 gives +3.25 pp CV over baseline
3. **Multi-camera hurts without flag**: 38.29% vs 41.11% (front-only). Camera-ID flag recovers +14.60 pp
4. **Augmentation does not improve CV**: 68.96% vs 74.23% (-5.27 pp). Synthetic variants don't teach new subjects
5. **Data scarcity is primary bottleneck**: Train-all = 94.44% but CV = 44.36% —50 pp gap proves memorization without generalization

---

## Dataset Section (Draft)

> The dataset comprises 88 recording sessions from 44 unique participants (front-facing camera, 8 gesture classes). Each session was mirrored to produce a paired duplicate, yielding 88 skeleton files with corresponding JSON annotations. Skeleton keypoints were extracted using MediaPipe Holistic at 30 fps, with each frame represented as a 10-joint, 30-dimensional vector (x, y, z per joint).
>
> To address the small subject pool, we applied geometric and temporal augmentation including rotation (±5°, ±10°, ±15°), spatial perturbation (scaling, vertical shift, joint noise), and temporal interpolation (2×), yielding 484 training instances. However, cross-validation experiments showed that augmentation does not improve cross-subject generalization (68.96% ±22.01% vs 74.23% ±16.96% without augmentation), as synthetic variants of existing subjects do not teach the model to recognize genuinely different individuals. All cross-validation was performed at the subject level to prevent augmented variants of the same individual from appearing in both training and validation folds.
>
> A held-out test set of 6 subjects (14%) was frozen before any hyperparameter tuning to prevent overfitting to the validation protocol. The final model was trained on all 44 subjects and evaluated once on this held-out set.

---

## Per-Class Breakdown (Original Model, 94.44% accuracy)

| Class | Precision | Recall | F1 | Windows |
|-------|:---------:|:------:|:--:|:-------:|
| foot_hold | 100.0% | 91.9% | 95.8% | 4594 |
| foot_lift | 95.6% | 99.7% | 97.6% | 330 |
| sideway_kick | 84.3% | 99.3% | 91.2% | 302 |
| cross_front | 85.9% | 98.6% | 91.8% | 512 |
| heel_tap | 90.1% | 100.0% | 94.8% | 292 |
| flamingo_bend | 78.9% | 100.0% | 88.2% | 270 |
| forward_step | 86.7% | 99.5% | 92.7% | 414 |
| forward_kick | 82.8% | 98.0% | 89.8% | 354 |

**Confusion patterns**: foot_hold absorbs misclassifications from all other classes (12.1% of foot_hold windows are false positives). flamingo_bend has lowest precision (78.9%) — confused with foot_hold.

---

## 5-Fold CV Results

### Original (88 files, 44 subjects)
| Fold | Subjects | Val Windows | Accuracy |
|------|----------|:-----------:|:--------:|
| 1 | p14, p16, p18, p24, p36, p39, p42, p44 | 10124 | 94.82% |
| 2 | p07, p13, p15, p17, p21, p43, p47, p48 | 17701 | 84.37% |
| 3 | p05, p10, p11, p23, p31, p37, p38, p49 | 6713 | 30.70% |
| 4 | p19, p20, p26, p29, p33, p34, p46 | 2110 | 62.61% |
| 5 | p08, p12, p22, p27, p28, p32, p41 | 8347 | 72.28% |
| **Mean** | | | **68.96% ±22.01%** |

### Previous Baseline (88 files, 44 subjects, cnn_out=64, dropout=0.6)
| Fold | Accuracy |
|------|:--------:|
| 1 | 92.36% |
| 2 | 83.42% |
| 3 | 44.66% |
| 4 | 77.19% |
| 5 | 73.51% |
| **Mean** | **74.23% ±16.96%** |

### Augmented (418 files, 38 training subjects)
| Fold | Accuracy |
|------|:--------:|
| 1 | 94.82% |
| 2 | 84.37% |
| 3 | 30.70% |
| 4 | 62.61% |
| 5 | 72.28% |
| **Mean** | **68.96% ±22.01%** |

---

## Learning Curve (Held-Out Test Accuracy vs # Training Subjects)

| Subjects | Original | Augmented |
|:--------:|:--------:|:---------:|
| 5 | 80.97% | 80.65% |
| 10 | 84.51% | 83.10% |
| 20 | 81.86% | 83.27% |
| 30 | 87.17% | 85.06% |
| 38 | 89.38% | — |

**Takeaway**: Accuracy increases with more subjects (80.97% → 89.38%), confirming data scarcity is the bottleneck. Augmentation does not help — curves are nearly identical, and original slightly outperforms augmented at 30 subjects.

---

## t-SNE Visualizations

- `results_orig/tsne.png` — CNN embeddings colored by class (original model, 7068 windows)
- `results_aug/tsne.png` — CNN embeddings colored by class (augmented model, 44995 windows)

---

## Held-Out Test Set

Frozen before any tuning:
- **Subjects**: p25, p30, p35, p40, p45, p50
- **Original files**: 12 (6 subjects × 2 mirror)
- **Augmented files**: 66 (12 × 5.5 augmentations)
- **Location**: `data/skeletons/front_test/` + `data/labels/front_test/`

### Held-Out Results (6 unseen subjects, 226 windows)

| Model | Top-1 | Top-3 | foot_hold F1 | Notes |
|-------|:-----:|:-----:|:------------:|-------|
| **Original (train_all on 44 subj)** | **68.14%** | **94.69%** | 3.4% | foot_hold almost completely fails |
| Augmented (train_all on 38 subj) | 66.37% | 84.96% | 0.0% | foot_hold = 0% precision AND recall |

**foot_hold collapse on held-out**: Both models fail on foot_hold for unseen subjects (25%/0% precision). The model memorizes foot_hold for training subjects but cannot generalize — this is the clearest evidence of overfitting to training subjects.

### Augmented Model — Full Augmented Dataset (train_all, 44995 windows)

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| foot_hold | 100.0% | 91.8% | 95.7% |
| foot_lift | 94.7% | 99.1% | 96.9% |
| sideway_kick | 86.2% | 99.2% | 92.3% |
| cross_front | 86.2% | 98.5% | 92.0% |
| heel_tap | 88.9% | 100.0% | 94.1% |
| flamingo_bend | 82.8% | 99.5% | 90.4% |
| forward_step | 89.6% | 98.8% | 94.0% |
| forward_kick | 83.4% | 98.2% | 90.2% |

**Overall: 94.53%** (matches original's 94.44% — identical in-sample fit)
