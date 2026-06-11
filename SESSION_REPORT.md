# FERN v2 — Session Report

## Overview

Fixed video concatenation corruption, completed grouped_by_gesture dataset pipeline, verified end-to-end training. **merged_v1 skeleton extraction never completed** — still needs to be done.

---

## 1. merge_v1_db.py — Fixed

### Problem
Concat demuxer (`-f concat -c copy`) reused decoder across clips with different codecs → corrupted output.

### Root Cause
SAR mismatch: clips with different DARs (16:9 vs 3:2) produced different SARs after `scale`, and the `concat` filter requires identical SAR across all inputs.

### Fix (`src/merge_v1_db.py:86`)
```python
[{i}:v]scale={w}:{h}:flags=bilinear,setsar=1[s{i}]
```

### Result
49/50 groups succeeded. p02_c1 still fails (mixed-resolution MOV→MP4 conversions).

---

## 2. group_by_gesture.py — Fixed & Completed (9/9 Gestures)

### Problem
SAR + frame rate + time base mismatches between clips from different persons.
- p01: 1280×720 @ 14.35 fps, DAR 16:9
- p02: 1080×720 @ 30 fps, DAR 3:2

### Fix (`src/group_by_gesture.py:61-92`)
Full normalization per input:
```
[{i}:v]settb=1/{fps},setpts=PTS-STARTPTS,
scale={w}:{h}:flags=bilinear,setsar=1,
fps={fps},format=yuv420p[s{i}]
```

Chunked (25 clips/chunk) + concat demuxer merge for large groups (max 283 clips).

### Label Names Fixed
| V1_TO_V2 | DEFAULT_CLASSES |
|----------|-----------------|
| heeltap | heel_tap |
| flamingo | flamingo_bend |
| lateral_slide | sideway_kick |
| foot_hold | **(needs adding — see §5)** |
| neutral | (ignored) |

---

## 3. Skeleton Extraction

### grouped_by_gesture ✅ Done
- 9 gesture videos → 9 CSVs via MediaPipe Pose
- 64–75% detection rate
- 10 lower-body joints (normalized coordinates)

### merged_v1 ❌ NOT DONE
`data/skeletons/merged_v1/` is **empty**. Extraction was started but never completed. Still needs ~4-6 hours.

### protobuf Issue ⚠️
Current: protobuf 7.35.0 — **will crash** mediapipe at runtime. Fix:
```powershell
cd C:\fern\FERN_V2
.\venv\Scripts\Activate.ps1
pip install "protobuf>=4.21,<5.0"
```

---

## 4. Mirror Augmentation

### Bundled `mirror_skeletons.py` ❌ BROKEN
Expects 33 MediaPipe joints (99 columns). Our CSVs have 10 joints (30 columns). **Do not use.**

### Custom `mirror_10joint.py` ✅
Written last session, correct for 10-joint format:
1. Negate X for all joints
2. Swap left/right pairs (5 pairs)
3. Copy label JSON with `mirrored: true`

Run after skeleton extraction:
```powershell
python src\mirror_10joint.py ^
    --skeleton_dir data\skeletons\merged_v1 ^
    --label_dir    data\labels\merged_v1 ^
    --output_skel  data\skeletons\merged_v1 ^
    --output_label data\labels\merged_v1
```

---

## 5. Code Fixes Applied (3 files)

### `src/dataset_v2.py` — added foot_hold as idle class
```python
DEFAULT_CLASSES = [
    "foot_hold",       # class 0 — idle/null gesture
    "foot_lift",
    "sideway_kick",
    "cross_front",
    "heel_tap",
    "flamingo_bend",
    "forward_step",
    "forward_kick",
]
```

### `src/model_v2.py` — updated default num_classes=8
```python
class FERNv2(nn.Module):
    def __init__(self, ..., num_classes: int = 8, ...):
```

### `src/train_v2.py` — CUDA optimizations
- `torch.backends.cudnn.benchmark = True` (was False)
- `torch.backends.cudnn.deterministic = False` (was True)
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- `torch.set_float32_matmul_precision("high")`

---

## 6. Label Fixes (2 scripts)

### `src/fix_labels.py` — quick fixes (55 JSONs updated)
| Fix | merged_v1 | grouped_by_gesture |
|-----|-----------|-------------------|
| `neutral` → `foot_hold` in segments | 47 files | 2 files (neutral.json + mirror) |
| `heeltap` → `heel_tap` in gesture_order | 36 files | 2 files |
| `lateral_slide` → `sideway_kick` in gesture_order | 36 files | 2 files |
| `flamingo` → `flamingo_bend` in gesture_order | 47 files | 2 files |

### `src/refine_labels.py` — skeleton-based boundary refinement
- Computes foot velocity from ankle/heel/foot_index joints
- Snaps segment boundaries to motion onset/offset frames
- Adaptive threshold based on motion percentile (default 70th)
- Runtime: ~1s per video (CPU)
- Applied to all 18 grouped_by_gesture labels (total shift varies per gesture)
- Ready for merged_v1 once skeletons are extracted

---

## 7. Environment

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.2.2+cu121 | ✅ CUDA — verified working |
| torchvision | 0.17.2+cu121 | ✅ |
| mediapipe | 0.10.35 | ✅ protobuf 4.25.9 compatible |
| protobuf | 4.25.9 | ✅ downgraded from 7.35.0 |
| pandas | 2.2.2 | ✅ |
| tensorboard | 2.16.2 | ✅ |

Venv: `C:\fern\FERN_V2\venv\`

---

## 8. Directory Status

```
C:\fern\FERN_V2\data\
├── grouped_by_gesture\   ← 9 MP4s + 9 JSONs ✅
├── labels\
│   ├── grouped_by_gesture\  ← 18 JSONs (refined, names fixed) ✅
│   └── merged_v1\           ← 50 JSONs (names fixed, not refined) ⚠️
├── merged_v1\             ← 49/50 MP4s + labels/ ✅ (missing p02_c1.mp4)
├── skeletons\
│   ├── grouped_by_gesture\  ← 9 CSV + 9 mirror CSVs ✅
│   └── merged_v1\           ← EMPTY ❌ (extraction pending)
├── raw\                   ← 33 test videos (no labels)
└── v1_clips\              ← original dataset by person
```

---

## 9. Training Progress

### CUDA Smoke Test (2 epochs, RTX 3070, 8 classes)
```
Device : cuda
GPU    : NVIDIA GeForce RTX 3070 Laptop GPU
VRAM   : 8.6 GB
Model parameters: 823,368
Speed : ~10s/epoch  (4.8x vs CPU at ~48s/epoch)
```

### grouped_by_gesture dataset (after fix + refinement)
| Class | Windows |
|-------|---------|
| foot_hold | 1,840 |
| foot_lift | 1,800 |
| sideway_kick | 2,734 |
| cross_front | 2,576 |
| heel_tap | 2,600 |
| flamingo_bend | 2,290 |
| forward_step | 2,632 |
| forward_kick | 2,470 |
| **Total** | **18,942** |

---

## 10. Remaining Work (ordered)

### merged_v1 pipeline (blocked on skeleton extraction)
1. **Finish merged_v1 skeleton extraction** — 27/50 CSVs done. Remaining: p07-p11 (20 videos, ~50-72 MB each) + testing_* (3 videos, 125-804 MB). MediaPipe CPU is slow on large files (~1-2 min per video).
2. **Refine merged_v1 labels** — `refine_labels.py` ready, just needs skeletons to exist
3. **Mirror merged_v1 skeletons** — `mirror_10joint.py` ready
4. **Combine into `data/skeletons/all/` + `data/labels/all/`** and train

### Priority tasks (from AI agent checklist)
5. **P1: infer_v2.py robustness** — handle `pose_landmarks=None`, confidence-based fallback for torso normalization
6. **P2: Deployment** — `export_onnx.py`, stateful `FERNInferencer` class with sliding window buffer
7. **P3: Reproducibility** — auto-download pretrained weights, `generate_dummy_data.py`
8. **P4: UX** — unified CLI, README with GIF and accuracy clarification, attention viz

### Known issues
- **p02_c1 merge still fails** — mixed-resolution MOV→MP4 clips
- **`data/raw/` (33 test videos)** — no skeletons or labels, not integrated
- **`mirror_skeletons.py`** bundled version expects 33 joints (broken for our 10-joint format) — use `mirror_10joint.py` instead
