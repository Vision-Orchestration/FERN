# FERN v2 — Final Project Report

## 1. Project Overview

**FERN (Foot-Equipped Recognition Network) v2** is a deep-learning system for real-time foot gesture recognition using skeleton-based pose estimation. The pipeline processes raw video clips, extracts lower-body skeletons (10 joints), labels gesture segments, augments via left-right mirroring, and trains an 8-class CNN-BiLSTM classifier.

**Repository root:** `C:\fern\FERN_V2\`

**Key numbers:**
- 10 persons (p00–p11, excluding p04), 5 camera angles (c1–c5) = ~50 video sources
- 2,165 raw clips in `data/v1_clips/`
- 8 gesture classes: `foot_hold`, `heel_tap`, `forward_kick`, `foot_lift`, `sideway_kick`, `forward_step`, `cross_front`, `flamingo_bend`
- Model: 823,368 parameters, CNN → BiLSTM → Classifier head
- Target: real-time inference on RTX 3070 Laptop GPU

---

## 2. Dataset Pipeline Architecture

```
v1_clips/  (2,165 raw clips, 10 persons × ~5 cameras)
    │
    ├── merge_v1_db.py  ────────────────────────────────┐
    │   Concat clips into per-person-per-camera MP4s     │
    │   50 videos: p00_c1.mp4 … p11_c4.mp4               │
    │   (p04 excluded — no clips)                        │
    ├── group_by_gesture.py  ──────────────────────────┐ │
    │   Group clips by gesture label across all persons │ │
    │   → 9 gesture-group MP4s (one per gesture class)  │ │
    │                                                    │ │
    ├── extract_skeleton.py / extract_skeleton_gpu.py   ▼ ▼
    │   MediaPipe Pose (CPU) / YOLOv8m-pose (GPU)        │
    │   → 10-joint CSV per video                          │
    │                                                      │
    ├── fix_labels.py  ── normalise gesture names ────────┤
    ├── repair_labels.py  ── fix start > end corruption ──┤
    ├── refine_labels.py  ── boundary refinement ─────────┤
    │                                                      │
    ├── mirror_10joint.py  ── left-right mirror ──────────┤
    │                                                      │
    └── train_v2.py  ── combined dataset training ────────┘
```

---

## 3. Video Concatenation

### 3.1 `merge_v1_db.py` — Per-person-per-camera videos

**Purpose:** Concatenates all clips for one person+camera into a single video + generates label JSON with per-clip segment boundaries.

**Fixes applied:**

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Corrupted output (green frames) | SAR mismatch: clips with DAR 16:9 vs 3:2 produce different SAR after `scale`; `concat` filter requires identical SAR | Added `setsar=1` to every input stream: `[{i}:v]scale={w}:{h}:flags=bilinear,setsar=1[s{i}]` |
| Output stuck at first clip | `concat` protocol mismatch — using `-f concat -c copy` (concat demuxer) instead of concat filter | Switched to `-filter_complex` concat filter |
| Motion interpolation artefacts | Using `minterpolate` filter between clips | Removed `minterpolate` |

**Result:** 49/50 groups succeeded. `p02_c1` still fails due to mixed-resolution MOV→MP4 source clips that cannot be consistently scaled.

**Label generation (`build_label_json`):** Parses the clip manifest to create sequential segment boundaries with gesture names, FPS, total_frames. Buffered clips (buf=3 frames) trimmed from segments. **Bug:** when `total_frames < buf*2`, segments with `end_frame < start_frame` are produced. Fixed by `repair_labels.py`.

### 3.2 `group_by_gesture.py` — Gesture-grouped videos

**Purpose:** Groups all clips of the same gesture class across all persons/cameras into one video per gesture (9 videos total: heeltap, forward_kick, forward_step, foot_lift, sideway_kick, cross_front, flamingo, neutral, foot_hold).

**Fixes applied:**

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| FFmpeg hang | 283 clips of heeltap exceeds max input limit | Chunked into 25-clip batches, then concat demuxer merge |
| SAR/fps mismatch | p01 (1280×720 @ 14.35 fps, DAR 16:9) vs p02 (1080×720 @ 30 fps, DAR 3:2) | Full per-input normalisation: `settb=1/{fps},setpts=PTS-STARTPTS,scale={w}:{h}:flags=bilinear,setsar=1,fps={fps},format=yuv420p` |
| Codec conflict | Concatenating AVC (H.264) with MJPEG clips | `format=yuv420p` ensures consistent pixel format |

**Results:** All 9 gesture groups merged successfully. Gesture-vs-gesture balance varies from 212 clips (foot_lift) to 283 clips (heeltap and flamingo).

| Gesture | Clips | Filename |
|---------|-------|----------|
| heeltap | 283 | `heeltap.mp4` |
| forward_kick | 232 | `forward_kick.mp4` |
| forward_step | 222 | `forward_step.mp4` |
| foot_lift | 212 | `foot_lift.mp4` |
| sideway_kick | 220 | `lateral_slide.mp4` |
| cross_front | 228 | `cross_front.mp4` |
| flamingo_bend | 283 | `flamingo.mp4` |
| foot_hold | 256 | `neutral.mp4` (renamed in labels) |

---

## 4. Skeleton Extraction

### 4.1 Format

All skeletons use a consistent 10-joint lower-body format:

| Index | Joint | Source (MediaPipe COCO) |
|-------|-------|------------------------|
| 0 | left_hip | MP 23 |
| 1 | right_hip | MP 24 |
| 2 | left_knee | MP 25 |
| 3 | right_knee | MP 26 |
| 4 | left_ankle | MP 27 |
| 5 | right_ankle | MP 28 |
| 6 | left_heel | MP 29 |
| 7 | right_heel | MP 30 |
| 8 | left_foot_index | MP 31 |
| 9 | right_foot_index | MP 32 |

**Normalisation (applied to every frame):**
1. Centre all joints relative to mid-hip (avg of left/right hip)
2. Scale by torso length (distance mid-hip → mid-shoulder)
3. Store `mid_hip_x, mid_hip_y, mid_hip_z, torso_length` as metadata columns

**CSV columns (36 total):**
```
frame_idx, pose_detected,
left_hip_{x,y,z}, right_hip_{x,y,z},
left_knee_{x,y,z}, right_knee_{x,y,z},
left_ankle_{x,y,z}, right_ankle_{x,y,z},
left_heel_{x,y,z}, right_heel_{x,y,z},
left_foot_index_{x,y,z}, right_foot_index_{x,y,z},
mid_hip_{x,y,z}, torso_length
```

### 4.2 `extract_skeleton.py` — MediaPipe Pose (CPU)

**Used for:** `grouped_by_gesture` (9 videos) and initial merged_v1 batch (p00–p06, 25 videos).

- Model: `mediapipe.pose_landmarker` with `BlazePose GPU Heavy`
- Static image mode: True (per-frame detection for accuracy)
- 10-joint extraction from 33 MediaPipe landmarks
- Speed: ~1–2 min per video (variable, depends on resolution)
- Detection rate: 64–75% on grouped_by_gesture

**Limitations:**
- Very slow on large merged_v1 videos (50–800 MB each, ~4–6 hours total)
- c4 camera angle consistently fails (<30% detection) due to extreme viewing angle

### 4.3 `extract_skeleton_gpu.py` — YOLOv8m-pose GPU

**Used for:** merged_v1 videos p07–p11 + testing (25 videos).

**Why GPU:** MediaPipe CPU was too slow for remaining large videos (estimated ~4–6 additional hours). Switched to YOLOv8m-pose via onnxruntime-gpu with CUDA + TensorRT providers.

- Model: YOLOv8m-pose (medium) — 46.4% average detection rate
- Framework: onnxruntime-gpu 1.26.0 with `TensorrtExecutionProvider` + `CUDAExecutionProvider` + `CPUExecutionProvider`
- Input: 640×640 RGB, full-image inference
- Post-processing: NMS (IoU=0.5, conf=0.3)
- Keypoint mapping: YOLO COCO keypoints (0–16) → our 10 lower-body joints
  - YOLO 11(R-hip) → right_hip, 12(R-knee) → right_knee, etc.
  - Heel approximated: `heel_x = ankle_x - 0.15 * (knee_x - ankle_x)` (behind ankle)
  - Foot index approximated: `foot_index_x = ankle_x + 0.15 * (knee_x - ankle_x)` (in front)
- Normalisation: same mid-hip centering + torso-length scaling as MediaPipe
- YOLO has no `z` coordinate → all `z` set to 0
- Speed: ~33 min total for 25 videos (RTX 3070)

**Detection rate breakdown:**
- ≥90%: 6 videos (p08_c2, p09_c4, p10_c1/2/3, p11_c4)
- 50–90%: 12 videos
- 30–50%: 4 videos
- <30%: 3 videos (all c4 camera or testing)

**Comparison with YOLOv9/10/11 (also tried):**
- YOLOv9t-pose: 46.6% avg detection (similar, ~same speed)
- YOLOv10m-pose: 38.8% avg detection
- YOLOv11m-pose: 39.5% avg detection
- YOLOv8x-pose: 0% on worst-case videos (too rigid NMS)
- **Winner:** YOLOv8m-pose (best combination of speed, detection rate, reliability)

**Why not MoveNet:** Thunder model gave very jittery keypoints; Lightning was inaccurate.

### 4.4 Summary table

| Dataset | Method | Videos | Avg Detection | Time |
|---------|--------|--------|---------------|------|
| grouped_by_gesture | MediaPipe CPU | 9 | ~70% | ~15 min |
| merged_v1 p00–p06 | MediaPipe CPU | 25 | ~55% | ~2–3 hours |
| merged_v1 p07–p11 + testing | YOLOv8m-pose GPU | 25 | ~46% | ~33 min |
| **Total** | | **59** | | |

---

## 5. Label Pipeline

### 5.1 Raw labels (from `label_videos_v3.py`)

Generated by the original `label_videos_v3.py` which annotated merged_v1 clips with heuristic per-clip gesture labels and segment boundaries.

### 5.2 `fix_labels.py` — Gesture name normalisation

**Problem:** The original labels used inconsistent naming:
- `heeltap` → should be `heel_tap` (snake_case)
- `lateral_slide` → should be `sideway_kick` (actual gesture name)
- `flamingo` → should be `flamingo_bend` (full name)
- `neutral` → should be `foot_hold` (descriptive name for idle)

**Fix:** Applied in-place to both `gesture_order` arrays and individual `segments[].gesture` fields. Removes duplicate `foot_hold` entries when both `neutral` and `foot_hold` existed.

**Files updated:** 55 total
- merged_v1: 47 files with neutral→foot_hold, 36 files with heeltap/flamingo/lateral_slide fixes
- grouped_by_gesture: all 18 files (9 original + 9 mirror)

### 5.3 `repair_labels.py` — Corruption fix

**Problem:** 12 merged_v1 label files had segments where `start_frame > end_frame` due to a bug in `merge_v1_db.py`'s `build_label_json`: when a clip's effective frames (after buf=3 trim) were ≤ 0, the `end_frame` was set to the same value as the previous corrupted segment, propagating backwards through all subsequent segments.

**12 corrupted files:** p00_c1, p00_c4, p02_c3, p05_c3, p06_c3, p06_c5, p07_c1, p07_c4, p08_c1, p09_c1, p09_c3, testing_timestamp

**Fix:** For each corrupted file:
1. Sort segments by `start_frame`
2. Set `end_frame = next_segment.start_frame - 1` for each segment (or `total_frames - 1` for last)
3. If `end <= start` (segment too short after repair), give it `start + max(30, (total - start) // 10)` frames
4. Add metadata field `repaired_by: "repair_labels.py"`

**Downstream interaction with refine_labels.py:** The first run of `refine_labels.py` after repair re-introduced corruption because its motion-based boundary snapping could push `end_frame` before `start_frame` when skeleton data was shorter than the video timeline (common on low-detection videos). Fixed with a safety guard in `refine_segments()` that enforces `end >= start` after all boundary adjustments (see §5.4).

### 5.4 `refine_labels.py` — Motion-based boundary refinement

**Purpose:** Adjust heuristic clip-boundary-based segment start/end frames to snap to actual foot-motion onset/offset, using skeleton motion signals.

**Algorithm:**

```
1. Load skeleton CSV → extract foot joint coordinates (6 joints × 3 axes = 18 features)
2. Compute per-frame foot speed: L2 displacement of foot joints between consecutive frames
3. Smooth with moving-average (window = fps/6, minimum 3)
4. Compute adaptive threshold: 70th percentile of motion between 30th and 95th percentiles,
   with FPS-dependent floor at 0.002 × fps
5. For each segment:
   foot_hold (idle): start→where motion drops below threshold, end→where motion rises above
   active gestures: start→where motion rises above, end→where motion drops below
6. Apply safety guard: enforce end ≥ start, clamp to T-1, prevent overlap with previous segment
```

**Safety guard fix (critical):** The initial implementation had a logic bug where `end` was clamped to `T-1` (skeleton length) AFTER being set to `start + 30`, so if `T` was shorter than `start`, the clamp undid the fix. Corrected order: `if e < s → e = s + 30`, then `if e >= T → e = max(s, T-1)`. This was essential for videos where the detected skeleton was significantly shorter than the full video timeline (common with c4 camera footage at <30% detection).

**Results:**
- **grouped_by_gesture:** All 18 labels refined (9 original + 9 mirror), total shift varies per gesture
- **merged_v1 (after repair):** 49/50 refined (testing_vid has no skeleton → skipped), total shifts:
  - Low-detection files (c4 cameras) had large shifts due to safety guard enforcing minimum segment lengths
  - p00_c1: 6,233 frames total shift (safety guard fixed many corrupted segments)
  - p06_c5: 7,017 frames (same)
  - Normal files: 10–90 frames total shift (minor boundary adjustments)

### 5.5 Label pipeline order (critical)

The correct sequence is:
1. Generate labels (from `merge_v1_db.py` or `label_videos_v3.py`)
2. `fix_labels.py` — normalise names
3. `repair_labels.py` — fix start_frame > end_frame corruption
4. `refine_labels.py` — motion-based boundary refinement (requires skeleton CSVs)

**Do not re-order.** If `refine` is run before `repair`, the corrupted segments get modified in unpredictable ways. If `refine` is run before `fix`, the old gesture names cause incorrect thresholding (foot_hold gets special idle handling).

---

## 6. Mirror Augmentation

### 6.1 `mirror_skeletons.py` (bundled, BROKEN)

Expects 33 MediaPipe joints (99 coordinate columns). Our CSVs use 10 joints (30 columns). **Do not use.**

### 6.2 `mirror_10joint.py` (created for 10-joint format)

**Operation for each frame:**
1. Negate X coordinate of every joint (horizontal flip)
2. Swap left↔right joint pairs (5 pairs: hip, knee, ankle, heel, foot_index)
3. Negate `mid_hip_x` (reference point flips too)
4. Copy label JSON with `mirror_of: <original_stem>` metadata

**Output naming:** `<stem>_mirror.csv` and `<stem>_mirror.json` in the same directory.

**Applied to:**
- merged_v1: 50 original → 50 mirror (100 total)
- grouped_by_gesture: mirror files already existed from a prior session (verified correct)

---

## 7. Model and Training Configuration

### 7.1 `dataset_v2.py` — 8-class dataset

**Default classes (DEFAULT_CLASSES):**
```python
["foot_hold", "heel_tap", "forward_kick", "foot_lift",
 "sideway_kick", "forward_step", "cross_front", "flamingo_bend"]
```

- `foot_hold` added as class 0 (idle/null gesture), remapped from old `neutral`
- Window extraction: `window_size=64` frames, `stride=32` frames
- Per-window normalisation: mid-hip centering + torso scaling (already in CSV)
- NaN frames: linear interpolation, remaining NaN → 0.0

### 7.2 `model_v2.py` — Architecture

```
Input: (batch, 64, 30)  ← 64 frames × (10 joints × 3 coords)
  │
  ├─ Conv1D(30 → 64, kernel=3, pad=1) + ReLU
  ├─ Conv1D(64 → 128, kernel=3, pad=1) + ReLU
  ├─ Conv1D(128 → 256, kernel=3, pad=1) + ReLU
  ├─ AdaptiveAvgPool1D(1) → (batch, 256)
  │
  ├─ BiLSTM(256 → 128, 2 layers, dropout=0.2) → (batch, 256)
  ├─ Dropout(0.2)
  ├─ Linear(256 → 128) + ReLU
  ├─ Dropout(0.2)
  └─ Linear(128 → 8)  ← 8 gesture classes
```

**Parameters:** 823,368

### 7.3 `train_v2.py` — CUDA optimisations

**Performance-critical changes:**

| Setting | Before | After | Effect |
|---------|--------|-------|--------|
| `cudnn.benchmark` | False | True | Auto-tune cuDNN kernels for input sizes |
| `cudnn.deterministic` | True | False | Non-deterministic but faster |
| `TF32 matmul` | off | on (`allow_tf32`) | ~2× faster float32 matmuls |
| `TF32 cuDNN` | off | on | Faster convolutions |
| `float32_matmul_precision` | "highest" | "high" | Enables TF32 |

**Training speed:** ~10s/epoch on RTX 3070 Laptop GPU (4.8× faster than CPU at ~48s/epoch)

---

## 8. Combined Dataset

### 8.1 Composition (112 files)

| Source | Original | Mirror | Total |
|--------|----------|--------|-------|
| grouped_by_gesture | 9 | 9 | 18 |
| merged_v1 | 50 | 50 | 100 |
| testing_* (removed) | 3 | 3 | — |
| **Total** | **59** | **59** | **112** |

Testing files (testing_p101, testing_timestamp, testing_vid) were **excluded** because:
- Labels contain `"unknown"` gesture class (not in 8-class set)
- Skeleton detection <30% (c4 camera footage)
- `testing_vid` has no skeleton → cannot be refined, labels unreliable

### 8.2 Final quality check

```
Total labels:  112
Total skeletons: 112
Gesture classes: 8 (cross_front, flamingo_bend, foot_hold, foot_lift,
                      forward_kick, forward_step, heel_tap, sideway_kick)
Total segments: 7,872
Corrupted segments (start > end): 0
Refined labels: 110 / 112 (testing_vid + mirror unrefined — no skeleton)
```

### 8.3 Directory layout

```
C:\fern\FERN_V2\data\
├── skeletons\
│   ├── all\                  ← 112 CSVs (combined training input)
│   ├── grouped_by_gesture\   ← 18 CSVs (9 + 9 mirror)
│   └── merged_v1\            ← 100 CSVs (50 + 50 mirror)
├── labels\
│   ├── all\                  ← 112 JSONs (combined training input)
│   ├── grouped_by_gesture\   ← 18 JSONs (refined, names fixed)
│   └── merged_v1\            ← 100 JSONs (repaired, refined, names fixed)
├── merged_v1\                ← 49 MP4s + original labels
├── grouped_by_gesture\       ← 9 MP4s
├── raw\                      ← 33 raw test videos (not yet processed)
└── v1_clips\                 ← 2,165 original clips by person
```

---

## 9. Source Scripts (complete reference)

| Script | Purpose | Status |
|--------|---------|--------|
| `dataset_v2.py` | Data loader, windowing, 8-class mapping | ✅ Modified (foot_hold added) |
| `model_v2.py` | CNN-BiLSTM architecture, 8-class output | ✅ Modified (num_classes=8) |
| `train_v2.py` | Training loop, CUDA optimisations | ✅ Modified (TF32, benchmark) |
| `merge_v1_db.py` | FFmpeg concat into person-camera videos | ✅ Fixed (setsar=1, no interp) |
| `group_by_gesture.py` | FFmpeg concat into gesture videos | ✅ Fixed (norm, chunking) |
| `extract_skeleton.py` | MediaPipe CPU skeleton extraction | ✅ Used for grouped + early merged |
| `extract_skeleton_gpu.py` | YOLOv8m-pose GPU extraction | ✅ Created (25 videos, ~33 min) |
| `fix_labels.py` | Gesture name normalisation | ✅ Created (55 files fixed) |
| `repair_labels.py` | start > end corruption fix | ✅ Created (12 files repaired) |
| `refine_labels.py` | Motion-based boundary refinement | ✅ Created (+ safety guard fix) |
| `mirror_10joint.py` | 10-joint mirror augmentation | ✅ Created (50+50 mirrored) |
| `mirror_skeletons.py` | 33-joint mirror (BROKEN) | ❌ Not usable |
| `infer_v2.py` | Inference entry point | ⚠️ Needs robustness (P1) |
| `evaluate_v2.py` | Evaluation script | ✅ Exists |
| `label_videos_v3.py` | Original video labelling | ✅ Used |
| `check_corrupt.py` | Corruption checker utility | ✅ Created |
| `check_labels.py` | Label validator | ✅ Created |

---

## 10. Environment

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11.5 | |
| torch | 2.2.2+cu121 | CUDA 12.1 |
| torchvision | 0.17.2+cu121 | |
| mediapipe | 0.10.35 | BlazePose GPU |
| onnxruntime-gpu | 1.26.0 | CUDA + TensorRT providers |
| protobuf | 4.25.9 | **Required downgrade** — 7.x breaks mediapipe |
| pandas | 2.2.2 | |
| tensorboard | 2.16.2 | |
| opencv-python | 4.10.x | |
| numpy | 1.26.x | |
| ultralytics | 8.x | YOLO model loading |
| FFmpeg | 7.x | Video concatenation |

**CUDA devices:**
- GPU 0: NVIDIA GeForce RTX 3070 Laptop GPU (8.6 GB shared)
- CUDA capability: 8.6
- Provider list: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

**Venv:** `C:\fern\FERN_V2\venv\`

---

## 11. Known Issues

### P0 — Data quality
- **p02_c1 merge fails** — mixed-resolution MOV→MP4 clips cannot be consistently scaled; no merged video available for this source
- **c4 camera angle** — consistently produces <30% skeleton detection across all methods (MediaPipe, YOLOv8m, YOLOv9/10/11); these videos contribute limited training signal
- **Testing videos** (testing_p101, testing_timestamp, testing_vid) — excluded from training due to "unknown" class and poor detection; could be used as evaluation set but have no ground-truth labels

### P1 — `infer_v2.py` robustness
- Does not handle `pose_landmarks=None` (frames with no person detected)
- No confidence-based fallback for torso normalisation
- Missing sliding-window buffer for real-time inference

### P2 — Deployment
- No `export_onnx.py` script for ONNX runtime export
- No stateful `FERNInferencer` class

### P3 — Reproducibility
- No `generate_dummy_data.py` for testing without real data
- No auto-download of pretrained weights

### P4 — UX
- No unified CLI
- No README with usage examples or accuracy clarification
- No attention visualisation

### Infrastructure
- **protobuf version lock:** mediapipe 0.10.35 requires protobuf <5.0.0; pip may auto-upgrade to 7.x on fresh install. Fix: `pip install "protobuf>=4.21,<5.0"` before installing mediapipe.

---

## 12. Session Timeline

| Date | Session | Work Done |
|------|---------|-----------|
| Session 1 | Initial setup | Environment setup, merge_v1_db.py SAR fix, group_by_gesture.py normalisation+chunking, 8-class config, CUDA opts, skeleton extraction start |
| Session 2 | merged_v1 skeletons | GPU extraction with YOLOv8m-pose via onnxruntime-gpu — all 50 CSVs extracted (~33 min). fix_labels.py. refine_labels.py on grouped_by_gesture. |
| Session 3 | Repair + refine cycle | repair_labels.py (fix 12 corrupted files). refine_labels.py bug fix (safety guard for end < start). Re-refine all merged_v1. mirror_10joint.py on merged_v1. Combine datasets. Remove testing files. Final validation. | C:\fern\FERN_V2\SESSION_REPORT.md