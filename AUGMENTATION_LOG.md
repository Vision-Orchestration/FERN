# FERN v2 — Dataset Augmentation Summary

## Date: Aug 29, 2026

---

## What We Did

Added synthetic data augmentation to increase the training dataset from 88 files to 484 files (6.5× multiplier) without recording any new videos.

---

## The Problem

- 88 front-only skeleton CSVs (44 subjects × 2 mirror)
- 5-fold CV accuracy: 74.23% ±16.96%
- High variance — fold 3 dropped to 44.66% due to "hard" subjects
- No more time to record new videos

## The Solution

3 offline augmentation scripts + enhanced on-the-fly augmentation during training.

---

## New Augmentation Scripts

### 1. `augment_rotate.py` — Rotation Augmentation

Simulates slight camera angle variation by applying small Y-axis rotations.

```powershell
python src/augment_rotate.py --input_dir data/skeletons/front --label_dir data/labels/front --output_skel data/skeletons/front_aug --output_label data/labels/front_aug --angles 5 10 15 --mirror
```

- Rotates each skeleton by ±5°, ±10°, ±15° using `transform_skeleton.py`
- Generates mirror (X-flip) for each rotation
- **Result: 88 input → 264 new files**

### 2. `augment_spatial.py` — Spatial Augmentation

Simulates distance, camera height, and sensor noise variation.

```powershell
python src/augment_spatial.py --input_dir data/skeletons/front --label_dir data/labels/front --output_skel data/skeletons/front_aug --output_label data/labels/front_aug --variants 2 --mirror
```

Applies random combinations of:
- **Scaling**: 0.9-1.1 factor (simulates distance variation)
- **Vertical shift**: ±5% (simulates camera height difference)
- **Joint perturbation**: σ=0.01 noise (more aggressive than training-time σ=0.005)
- Generates mirror for each variant
- **Result: 88 input → 176 new files**

### 3. `augment_interpolate.py` — Temporal Interpolation

Doubles frame count by linear interpolation between consecutive frames.

```powershell
python src/augment_interpolate.py --input_dir data/skeletons/front --label_dir data/labels/front --output_skel data/skeletons/front_aug --output_label data/labels/front_aug --factor 2 --gestures foot_hold flamingo_bend
```

- Creates new frames by blending between real frames
- Adjusts label segment frame indices to match
- Best for slow, smooth gestures (foot_hold, flamingo_bend)
- **Result: 44 input → 44 new files (with doubled frame count)**

---

## Enhanced On-the-Fly Augmentation

Modified `dataset_v2.py` `_augment()` method to add 4 new transforms during training:

| Augmentation | Probability | What it does |
|-------------|:-----------:|-------------|
| Speed variation | 40% | Resample 80-120% of frames |
| Frame dropout | 30% | Drop 5-10% of frames (simulate missed detections) |
| Temporal shift | 30% | Crop/pad ±5-10 frames from edges |
| Random scaling | 30% | Scale coordinates by 0.9-1.1 |

These run on-the-fly during training — no new files needed, infinite variety.

---

## File Counts

| Dataset | Files | Location |
|---------|:-----:|----------|
| Original + mirror | 88 | `data/skeletons/front/` |
| Rotation augmented | 264 | `data/skeletons/front_aug/` |
| Spatial augmented | 176 | `data/skeletons/front_aug/` |
| Interpolation augmented | 44 | `data/skeletons/front_aug/` |
| **Total augmented** | **484** | `data/skeletons/front_aug/` |

---

## How to Train on Augmented Data

```powershell
python src/train_v2.py --skeleton_dir data/skeletons/front_aug --label_dir data/labels/front_aug --output_dir sweep_aug --epochs 200 --warmup_epochs 20 --batch_size 32 --window_size 60 --stride 15 --lr 3e-4 --weight_decay 1e-2 --dropout 0.3 --cnn_out 128 --lstm_hidden 0 --lstm_layers 1 --device cuda --num_workers 0 --train_all
```

Then run 5-fold CV with same params (minus `--train_all`) to get the unbiased number.

---

## Key Rules

1. **Always use `--num_workers 0`** on Windows (DataLoader hangs with >0)
2. **Never install `opencv-contrib-python`** (conflicts with `opencv-python`)
3. **Activate venv first**: `C:\fern\FERN_V2\venv\Scripts\Activate.ps1`
4. **Labeling tool uses single-pass mode**: `1`-`7` select gesture, `H` for foot_hold, `S`/`E` to mark

---

## Git History

- `89aa586` — Augmentation scripts + enhanced dataset_v2.py
- `0932fab` — Complete project documentation
