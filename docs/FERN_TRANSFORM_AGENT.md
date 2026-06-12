# FERN v2 — Geometric Transform & Stereo Triangulation Agent

## UPDATE: This approach FAILED Phase 1

**Phase 1 was executed on 2026-06-06. Result: FAIL (14.96% vs 70.64% front baseline).**

Angle sweep (0-90deg, step 5) showed ALL angles produce 14-15% accuracy — essentially
random (12.5% baseline for 8 classes). The geometric transform R_y(-theta) does NOT
work because MediaPipe single-camera z values, while non-zero, are too noisy to recover
a meaningful front-view equivalent via rotation.

**Follow-up (successful)**: Trained a front+45 combined model (`models_final_v2/fern_v2.onnx`)
which achieves 55.85% on front and 49.43% on c2 — balanced but lower per-angle accuracy.
See report.md section 12 for details.

This document is preserved for reference but DO NOT re-run Phase 1 or Phase 2.

---

## Your Job (historical — do not execute)

Execute Phase 1 and Phase 2 in order. Phase 2 only runs if Phase 1 passes its
success threshold. Read every section before writing any code.

---

## Critical Context

```
Model    : models_final/fern_v2.onnx   (132K CNN-only, opset17)
Training : front camera (c3) only, z=0, 76 CSVs, 8 classes, ~60% unbiased
Venv     : C:\fern\FERN_V2\venv\
PYTHONPATH: C:\fern\FERN_V2\src
Working dir: C:\fern\FERN_V2\
```

**8 gesture classes (DEFAULT_CLASSES order matters):**
```python
["foot_hold", "foot_lift", "sideway_kick", "cross_front",
 "heel_tap", "flamingo_bend", "forward_step", "forward_kick"]
```

**10-joint skeleton CSV format (36 columns total):**
```
frame_idx, pose_detected,
left_hip_x/y/z, right_hip_x/y/z,
left_knee_x/y/z, right_knee_x/y/z,
left_ankle_x/y/z, right_ankle_x/y/z,
left_heel_x/y/z, right_heel_x/y/z,
left_foot_index_x/y/z, right_foot_index_x/y/z,
mid_hip_x/y/z, torso_length
```
Feature columns = the 30 joint x/y/z columns only (not frame_idx, pose_detected,
mid_hip_*, torso_length).

**Camera naming convention:** c1=unknown, c2=unknown, c3=front (trained on),
c4=overhead (excluded, <30% detection), c5=unknown.
**The 45° camera is c2.** All paths below assume c2. If this is wrong, replace
c2 with the correct angle camera throughout.

**Why this works geometrically:**
A skeleton from camera B at angle θ is the same skeleton as front-view rotated
by θ around the Y-axis (vertical). Applying R_y(-θ) recovers the front-view
equivalent. The model never knows which camera it came from.

---

## Directory Structure to Create

```
C:\fern\FERN_V2\
├── src/
│   ├── transform_skeleton.py       [CREATE — Phase 1, core math]
│   ├── apply_transform_batch.py    [CREATE — Phase 1, batch processing]
│   ├── test_transform.py           [CREATE — Phase 1, evaluation]
│   ├── calibrate_stereo.py         [CREATE — Phase 2]
│   ├── extract_stereo_3d.py        [CREATE — Phase 2]
│   └── test_stereo.py              [CREATE — Phase 2]
├── data/
│   ├── skeletons/
│   │   ├── front/                  [EXISTING — 76 CSVs, z=0]
│   │   ├── raw_45/                 [CREATE — c2 skeletons, z preserved]
│   │   └── transformed_45/         [CREATE — after R_y(-45°) transform]
│   └── labels/
│       ├── front/                  [EXISTING]
│       ├── raw_45/                 [CREATE — copied from merged_v1 labels]
│       └── transformed_45/         [CREATE — same as raw_45]
└── calibration/
    └── stereo_calibration.json     [CREATE — Phase 2, after calibration session]
```

---

## PHASE 1 — Geometric Transform (MediaPipe z)

### Step 0 — Check z data in existing merged_v1 skeletons

Run this before writing any new code:

```powershell
cd C:\fern\FERN_V2
.\venv\Scripts\Activate.ps1
python - << 'EOF'
import pandas as pd, glob, numpy as np
files = glob.glob("data/skeletons/merged_v1/*c2*.csv")
print(f"c2 skeleton files found: {len(files)}")
if files:
    df = pd.read_csv(files[0])
    z_cols = [c for c in df.columns if c.endswith('_z') and 'mid_hip' not in c]
    z_vals = df[z_cols].values
    nonzero = np.count_nonzero(~np.isnan(z_vals) & (z_vals != 0.0))
    total   = np.count_nonzero(~np.isnan(z_vals))
    print(f"z columns: {z_cols[:3]} ...")
    print(f"Non-zero z values: {nonzero}/{total} ({100*nonzero/max(total,1):.1f}%)")
    print(f"z sample: {z_vals[z_vals != 0][:5]}")
EOF
```

**Decision:**
- If `Non-zero z values > 5%` → existing CSVs have valid z, go to Step 1.
- If `Non-zero z values == 0%` → z was zeroed during extraction, go to Step 0b.

### Step 0b — Re-extract c2 skeletons WITH z (only if Step 0 shows z=0)

Only run if Step 0 shows all-zero z. This re-extracts from source videos.

```powershell
$env:PYTHONPATH = "C:\fern\FERN_V2\src"
python src\extract_skeleton.py `
    --video_dir  data\merged_v1 `
    --output_dir data\skeletons\raw_45_extracted `
    --pattern    "*c2*.mp4"
```

If extract_skeleton.py has no `--pattern` arg, add it:

```python
# In extract_skeleton.py, in the argument parser section, add:
p.add_argument("--pattern", default="*.mp4",
               help="Glob pattern to filter videos (e.g. '*c2*.mp4')")

# In the file-discovery loop, change:
#   video_files = sorted(video_dir.glob("*.mp4"))
# to:
video_files = sorted(video_dir.glob(args.pattern))
```

After re-extraction, verify z is non-zero with the same check from Step 0
(pointing at `data/skeletons/raw_45_extracted/`). If still zero, look inside
`extract_skeleton.py` for any line doing `df[z_cols] = 0` or `z = 0` and
remove it.

### Step 1 — Isolate c2 skeleton and label files into raw_45/

```powershell
New-Item -ItemType Directory -Force -Path data\skeletons\raw_45
New-Item -ItemType Directory -Force -Path data\labels\raw_45

# Copy c2 skeleton CSVs (with z intact)
Get-ChildItem data\skeletons\merged_v1\*c2*.csv |
    Copy-Item -Destination data\skeletons\raw_45\

# Copy matching label JSONs
Get-ChildItem data\labels\merged_v1\*c2*.json |
    Copy-Item -Destination data\labels\raw_45\

# Verify counts match
$s = (Get-ChildItem data\skeletons\raw_45\*.csv).Count
$l = (Get-ChildItem data\labels\raw_45\*.json).Count
Write-Host "Skeletons: $s   Labels: $l"
# They must be equal. If not, find and remove unpaired files.
```

### Step 2 — Create `src/transform_skeleton.py`

Create this file exactly as written:

```python
"""
transform_skeleton.py
Geometric transformation: rotate a skeleton from an angled camera
back to front-view equivalent using R_y(-theta).
"""
import numpy as np

JOINT_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
N_JOINTS = 10


def rotation_y(theta_deg: float) -> np.ndarray:
    """3×3 rotation matrix around Y-axis (vertical)."""
    t = np.radians(theta_deg)
    return np.array([
        [ np.cos(t), 0, np.sin(t)],
        [ 0,         1, 0        ],
        [-np.sin(t), 0, np.cos(t)],
    ], dtype=np.float32)


def transform_to_front(
    skeleton: np.ndarray,
    camera_angle_deg: float = 45.0,
    zero_z_after: bool = True,
) -> np.ndarray:
    """
    Transform (T, 30) skeleton from angled camera to front-view equivalent.

    Parameters
    ----------
    skeleton         : (T, 30) — feature columns only, no metadata
    camera_angle_deg : angle of camera relative to subject front (degrees)
                       positive = camera is to the subject's right
    zero_z_after     : True  → set z=0 after transform (matches training data)
                       False → keep real depth (for Phase 2 / stereo)

    Returns
    -------
    (T, 30) — front-equivalent skeleton
    """
    if skeleton.shape[1] != N_JOINTS * 3:
        raise ValueError(
            f"Expected {N_JOINTS * 3} feature cols, got {skeleton.shape[1]}"
        )
    T = skeleton.shape[0]
    R = rotation_y(-camera_angle_deg)              # invert to undo camera rotation

    joints   = skeleton.reshape(T, N_JOINTS, 3)                      # (T, 10, 3)
    rotated  = (R @ joints.transpose(0, 2, 1)).transpose(0, 2, 1)    # (T, 10, 3)

    if zero_z_after:
        rotated[:, :, 2] = 0.0    # discard depth — matches z=0 training format

    return rotated.reshape(T, N_JOINTS * 3)


def transform_csv(
    csv_path: str,
    output_path: str,
    camera_angle_deg: float = 45.0,
    zero_z_after: bool = True,
) -> dict:
    """
    Load a skeleton CSV, apply transform, write to output_path.
    Preserves all non-feature columns (frame_idx, pose_detected, metadata).
    Returns stats dict.
    """
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(csv_path)

    # Identify the 30 feature columns (joint_x/y/z, not metadata)
    feature_cols = [
        c for c in df.columns
        if any(c.startswith(j) for j in JOINT_NAMES)
        and c.endswith(("_x", "_y", "_z"))
    ]
    if len(feature_cols) != N_JOINTS * 3:
        raise ValueError(
            f"{csv_path}: found {len(feature_cols)} feature cols, "
            f"expected {N_JOINTS * 3}"
        )

    raw = df[feature_cols].values.astype(np.float32)

    # Preserve NaN mask; fill NaN→0 before rotation (avoids NaN propagation)
    nan_mask  = np.isnan(raw)
    raw_clean = np.where(nan_mask, 0.0, raw)

    transformed = transform_to_front(raw_clean, camera_angle_deg, zero_z_after)

    # Restore NaN where original had no detection
    transformed[nan_mask] = np.nan

    df[feature_cols] = transformed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_no_detect = int((df["pose_detected"] == 0).sum()) \
                  if "pose_detected" in df.columns else 0
    return {
        "input":      csv_path,
        "output":     output_path,
        "frames":     len(df),
        "no_detect":  n_no_detect,
        "detect_pct": 100 * (1 - n_no_detect / max(len(df), 1)),
    }


if __name__ == "__main__":
    # Quick smoke test
    dummy = np.random.randn(60, 30).astype(np.float32)
    out   = transform_to_front(dummy, camera_angle_deg=45.0, zero_z_after=True)
    assert out.shape == (60, 30), "Shape mismatch"
    assert np.all(out[:, 2::3] == 0.0), "z not zeroed"
    print("transform_skeleton.py: smoke test passed.")
```

### Step 3 — Create `src/apply_transform_batch.py`

```python
"""
apply_transform_batch.py
Batch-apply geometric transform to all CSVs in a directory.
Copies label JSONs unchanged (geometry change does not affect labels).

Usage:
    python src/apply_transform_batch.py \
        --input_skel   data/skeletons/raw_45 \
        --input_label  data/labels/raw_45 \
        --output_skel  data/skeletons/transformed_45 \
        --output_label data/labels/transformed_45 \
        --angle        45.0
"""
import argparse
import shutil
from pathlib import Path

# Must be on PYTHONPATH
from transform_skeleton import transform_csv


def main(args):
    in_skel   = Path(args.input_skel)
    in_label  = Path(args.input_label)
    out_skel  = Path(args.output_skel)
    out_label = Path(args.output_label)
    out_skel.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_skel.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {in_skel}")

    print(f"Transforming {len(csv_files)} files  (angle={args.angle}°) ...")
    ok, failed = 0, []

    for csv_path in csv_files:
        out_csv = out_skel / csv_path.name
        try:
            s = transform_csv(
                str(csv_path), str(out_csv),
                camera_angle_deg=args.angle,
                zero_z_after=True,
            )
            print(f"  OK  {csv_path.name:40s} "
                  f"{s['frames']:5d} frames  "
                  f"{s['detect_pct']:5.1f}% detect")
            ok += 1
        except Exception as e:
            print(f"  ERR {csv_path.name}: {e}")
            failed.append(csv_path.name)

        # Labels don't change — just copy
        json_src = in_label / csv_path.with_suffix(".json").name
        json_dst = out_label / csv_path.with_suffix(".json").name
        if json_src.exists():
            shutil.copy2(json_src, json_dst)
        else:
            print(f"  WARN no label for {csv_path.name}")

    print(f"\nResult: {ok}/{len(csv_files)} succeeded.")
    if failed:
        print(f"Failed files: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_skel",   required=True)
    p.add_argument("--input_label",  required=True)
    p.add_argument("--output_skel",  required=True)
    p.add_argument("--output_label", required=True)
    p.add_argument("--angle", type=float, default=45.0,
                   help="Camera angle in degrees (positive = right of front)")
    main(p.parse_args())
```

### Step 4 — Create `src/test_transform.py`

```python
"""
test_transform.py
Evaluate the existing ONNX model on three conditions:
  A) Front camera — baseline
  B) Raw 45° skeleton — no transform
  C) Transformed 45° skeleton — after R_y(-45°)

Prints accuracy per condition and per class, then emits a Phase 1 verdict.

Usage:
    python src/test_transform.py
    python src/test_transform.py --angle 45.0 --model models_final/fern_v2.onnx
"""
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Must be on PYTHONPATH
from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES


def evaluate(model_path: str, skeleton_dir: str, label_dir: str,
             window_size: int = 60, stride: int = 15) -> dict:
    """Run ONNX model on a full dataset directory. Returns accuracy dict."""
    if not Path(skeleton_dir).exists():
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess       = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    ds = SkeletonWindowDataset(
        skeleton_dir=skeleton_dir,
        label_dir=label_dir,
        window_size=window_size,
        stride=stride,
        split="all",
        augment=False,
    )
    if len(ds) == 0:
        print(f"  WARNING: 0 windows in {skeleton_dir}")
        return None

    nc = len(DEFAULT_CLASSES)
    correct_per_class = np.zeros(nc, dtype=int)
    total_per_class   = np.zeros(nc, dtype=int)

    for x, y in ds:
        inp    = x.numpy().astype(np.float32)[np.newaxis]   # (1, T, 30)
        logits = sess.run(None, {input_name: inp})[0]
        pred   = int(np.argmax(logits))
        total_per_class[y]   += 1
        if pred == y:
            correct_per_class[y] += 1

    total   = int(total_per_class.sum())
    correct = int(correct_per_class.sum())
    return {
        "acc":     correct / total,
        "total":   total,
        "correct": correct,
        "per_class_acc":   correct_per_class / np.maximum(total_per_class, 1),
        "per_class_total": total_per_class,
    }


def print_result(label: str, r: dict):
    if r is None:
        print(f"  {label}: SKIPPED (directory not found)")
        return
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Overall accuracy: {r['acc']*100:.2f}%  "
          f"({r['correct']}/{r['total']} windows)")
    print(f"  {'Class':<22} {'Acc':>8} {'Windows':>9}")
    for i, cls in enumerate(DEFAULT_CLASSES):
        print(f"  {cls:<22} {r['per_class_acc'][i]*100:>7.1f}% "
              f"{r['per_class_total'][i]:>9}")


def main(args):
    print(f"\nModel: {args.model}")

    results = {
        "A_front": evaluate(
            args.model,
            args.front_skel, args.front_label,
            args.window_size, args.stride,
        ),
        "B_raw45": evaluate(
            args.model,
            args.raw45_skel, args.raw45_label,
            args.window_size, args.stride,
        ),
        "C_tf45": evaluate(
            args.model,
            args.tf45_skel, args.tf45_label,
            args.window_size, args.stride,
        ),
    }

    print_result("A — Front camera (baseline)", results["A_front"])
    print_result("B — Raw 45° (no transform)",  results["B_raw45"])
    print_result("C — Transformed 45°",          results["C_tf45"])

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  {'Condition':<35} {'Accuracy':>10}")
    for key, label in [
        ("A_front", "A — Front (baseline)"),
        ("B_raw45", "B — Raw 45°, no transform"),
        ("C_tf45",  "C — Transformed 45°"),
    ]:
        r = results[key]
        if r:
            print(f"  {label:<35} {r['acc']*100:>9.2f}%")

    # Verdict
    r_b = results["B_raw45"]
    r_c = results["C_tf45"]
    print(f"\n{'='*60}")
    print("  PHASE 1 VERDICT")
    if r_b and r_c:
        gain = (r_c["acc"] - r_b["acc"]) * 100
        print(f"  Transform gain over raw 45°: {gain:+.2f}%")
        if r_c["acc"] >= 0.65:
            print("  ✅ STRONG PASS  (≥65%) — Phase 2 strongly recommended.")
            print("     MediaPipe z has enough signal. Stereo will improve further.")
        elif r_c["acc"] >= 0.50:
            print("  ✅ PASS  (≥50%) — Proceed to Phase 2.")
            print("     Transform works. Real z from stereo will close the gap.")
        elif r_c["acc"] >= 0.35:
            print("  ⚠️  MARGINAL  (35–50%) — MediaPipe z is noisy.")
            print("     Phase 2 still worth attempting. Stereo z may fix it.")
        else:
            print("  ❌ FAIL  (<35%) — Transform did not help meaningfully.")
            print("     Do NOT proceed to Phase 2 without investigating.")
            print("     Check: is the camera angle correct? Is z truly non-zero?")
    else:
        print("  Could not compare — one or more conditions missing.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="models_final/fern_v2.onnx")
    p.add_argument("--front_skel",   default="data/skeletons/front")
    p.add_argument("--front_label",  default="data/labels/front")
    p.add_argument("--raw45_skel",   default="data/skeletons/raw_45")
    p.add_argument("--raw45_label",  default="data/labels/raw_45")
    p.add_argument("--tf45_skel",    default="data/skeletons/transformed_45")
    p.add_argument("--tf45_label",   default="data/labels/transformed_45")
    p.add_argument("--window_size",  type=int, default=60)
    p.add_argument("--stride",       type=int, default=15)
    p.add_argument("--angle",        type=float, default=45.0)
    main(p.parse_args())
```

### Step 5 — Run Phase 1 pipeline

Run these commands in order. Stop if any step fails.

```powershell
cd C:\fern\FERN_V2
$env:PYTHONPATH = "C:\fern\FERN_V2\src"
.\venv\Scripts\Activate.ps1

# 5.1 Smoke test the transform module
python src\transform_skeleton.py

# 5.2 Apply transform to all c2 skeletons
python src\apply_transform_batch.py `
    --input_skel   data\skeletons\raw_45 `
    --input_label  data\labels\raw_45 `
    --output_skel  data\skeletons\transformed_45 `
    --output_label data\labels\transformed_45 `
    --angle        45.0

# 5.3 Verify output
python -c "
from pathlib import Path
s = list(Path('data/skeletons/transformed_45').glob('*.csv'))
l = list(Path('data/labels/transformed_45').glob('*.json'))
print(f'Transformed skeletons: {len(s)}')
print(f'Labels: {len(l)}')
assert len(s) == len(l), 'MISMATCH — check for missing labels'
print('OK')
"

# 5.4 Run evaluation
python src\test_transform.py 2>&1 | Tee-Object -FilePath logs\phase1_result.txt
```

### Step 6 — Read the verdict

Open `logs/phase1_result.txt`. Find the PHASE 1 VERDICT section.
- ≥50% → proceed to Phase 2
- <50% → stop, report result, investigate before proceeding

---

## PHASE 2 — Stereo Triangulation (run only if Phase 1 ≥ 50%)

Phase 2 replaces MediaPipe's noisy z with geometrically correct z computed
from two camera views. The same front-view transform is then applied, but
with real depth values, closing the remaining gap to front-camera accuracy.

### Step 7 — Create `src/calibrate_stereo.py`

This script runs a live calibration session using a printed checkerboard.
It must be run once with both phones physically present.

```python
"""
calibrate_stereo.py
Stereo camera calibration using a printed checkerboard.

BEFORE RUNNING:
  1. Print a 9×6 checkerboard (inner corners). A4 paper is fine.
     Download from: https://calib.io/pages/camera-calibration-pattern-generator
  2. Set up both cameras (phone as DroidCam, or direct USB)
  3. Note the camera IDs (run with --list_cameras first)

Usage:
    python src/calibrate_stereo.py --list_cameras
    python src/calibrate_stereo.py --cam_front 0 --cam_45 1 --angle 45.0
    python src/calibrate_stereo.py --from_images --img_dir calibration/frames
"""
import argparse
import json
import os
import time
import cv2
import numpy as np
from pathlib import Path


CHESSBOARD = (9, 6)    # inner corners (columns, rows)
SQUARE_MM  = 25.0      # physical square size in mm — measure your printout!
N_REQUIRED = 20        # minimum good frames for calibration


def list_cameras(max_id: int = 8):
    print("Scanning camera IDs ...")
    for i in range(max_id):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            print(f"  Camera {i}: {'OK' if ret else 'opened but no frame'}")
            cap.release()
        else:
            print(f"  Camera {i}: not available")


def capture_calibration_frames(cam_id_a: int, cam_id_b: int,
                                save_dir: str, n_required: int = N_REQUIRED):
    """
    Interactive frame capture. Press SPACE to capture, Q to quit.
    Both cameras must see the checkerboard clearly for a frame to count.
    """
    dir_a = Path(save_dir) / "cam_a"
    dir_b = Path(save_dir) / "cam_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    cap_a = cv2.VideoCapture(cam_id_a)
    cap_b = cv2.VideoCapture(cam_id_b)
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError(f"Could not open cameras {cam_id_a} and {cam_id_b}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    count = 0
    print(f"\nCapturing calibration frames (need {n_required}).")
    print("SPACE = capture  |  Q = quit early\n")
    print("Move the checkerboard slowly to different positions and angles.")

    while count < n_required:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            continue

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        found_a, corners_a = cv2.findChessboardCorners(gray_a, CHESSBOARD, None)
        found_b, corners_b = cv2.findChessboardCorners(gray_b, CHESSBOARD, None)

        # Draw detection feedback
        disp_a = frame_a.copy()
        disp_b = frame_b.copy()
        cv2.drawChessboardCorners(disp_a, CHESSBOARD, corners_a, found_a)
        cv2.drawChessboardCorners(disp_b, CHESSBOARD, corners_b, found_b)

        status = f"Captured: {count}/{n_required}  " \
                 f"A:{'OK' if found_a else '--'}  B:{'OK' if found_b else '--'}"
        cv2.putText(disp_a, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        combined = np.hstack([
            cv2.resize(disp_a, (640, 360)),
            cv2.resize(disp_b, (640, 360)),
        ])
        cv2.imshow("Calibration (A | B)  SPACE=capture  Q=quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and found_a and found_b:
            # Subpixel refinement
            corners_a = cv2.cornerSubPix(gray_a, corners_a, (11,11), (-1,-1), criteria)
            corners_b = cv2.cornerSubPix(gray_b, corners_b, (11,11), (-1,-1), criteria)
            cv2.imwrite(str(dir_a / f"frame_{count:04d}.png"), frame_a)
            cv2.imwrite(str(dir_b / f"frame_{count:04d}.png"), frame_b)
            print(f"  Captured frame {count+1}/{n_required}")
            count += 1
        elif key == ord(' ') and not (found_a and found_b):
            print("  Not captured — checkerboard not detected in both cameras")

    cap_a.release()
    cap_b.release()
    cv2.destroyAllWindows()
    return count


def run_calibration(img_dir: str, output_path: str, angle_deg: float):
    """
    Run stereo calibration from captured image pairs.
    Saves calibration JSON to output_path.
    """
    dir_a = Path(img_dir) / "cam_a"
    dir_b = Path(img_dir) / "cam_b"

    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_MM

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points = []
    img_points_a = []
    img_points_b = []

    frames_a = sorted(dir_a.glob("frame_*.png"))
    frames_b = sorted(dir_b.glob("frame_*.png"))
    assert len(frames_a) == len(frames_b), "Frame count mismatch"

    h, w = None, None
    good = 0
    for fa, fb in zip(frames_a, frames_b):
        img_a = cv2.imread(str(fa))
        img_b = cv2.imread(str(fb))
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        h, w = gray_a.shape[:2]

        found_a, ca = cv2.findChessboardCorners(gray_a, CHESSBOARD, None)
        found_b, cb = cv2.findChessboardCorners(gray_b, CHESSBOARD, None)
        if not (found_a and found_b):
            continue

        ca = cv2.cornerSubPix(gray_a, ca, (11,11), (-1,-1), criteria)
        cb = cv2.cornerSubPix(gray_b, cb, (11,11), (-1,-1), criteria)

        obj_points.append(objp)
        img_points_a.append(ca)
        img_points_b.append(cb)
        good += 1

    print(f"Good frame pairs: {good}/{len(frames_a)}")
    if good < 10:
        raise RuntimeError(f"Only {good} good pairs — need at least 10. Recapture.")

    print("Running stereo calibration (may take a minute) ...")

    flags = cv2.CALIB_FIX_INTRINSIC  # calibrate intrinsics first, then stereo
    # Individual calibration first
    _, K_a, D_a, _, _ = cv2.calibrateCamera(obj_points, img_points_a, (w, h), None, None)
    _, K_b, D_b, _, _ = cv2.calibrateCamera(obj_points, img_points_b, (w, h), None, None)

    rms, K_a, D_a, K_b, D_b, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_a, img_points_b,
        K_a, D_a, K_b, D_b,
        (w, h),
        criteria=criteria,
        flags=flags,
    )

    print(f"Stereo calibration RMS error: {rms:.4f} px")
    if rms > 1.5:
        print("WARNING: RMS > 1.5 px — recapture with more diverse positions.")

    T_norm = np.linalg.norm(T)
    print(f"Baseline (camera separation): {T_norm:.1f} mm")

    calib = {
        "angle_deg":  angle_deg,
        "rms_px":     float(rms),
        "image_size": [w, h],
        "square_mm":  SQUARE_MM,
        "K_a":        K_a.tolist(),
        "D_a":        D_a.tolist(),
        "K_b":        K_b.tolist(),
        "D_b":        D_b.tolist(),
        "R":          R.tolist(),     # rotation from cam_a to cam_b
        "T":          T.tolist(),     # translation from cam_a to cam_b (mm)
        "E":          E.tolist(),
        "F":          F.tolist(),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration → {output_path}")
    return calib


def main(args):
    if args.list_cameras:
        list_cameras()
        return

    if not args.from_images:
        n = capture_calibration_frames(
            args.cam_front, args.cam_45,
            args.img_dir, N_REQUIRED,
        )
        if n < 10:
            print(f"Only {n} frames captured — need ≥10. Run again.")
            return

    run_calibration(args.img_dir, args.output, args.angle)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list_cameras",  action="store_true")
    p.add_argument("--cam_front",     type=int, default=0)
    p.add_argument("--cam_45",        type=int, default=1)
    p.add_argument("--angle",         type=float, default=45.0)
    p.add_argument("--img_dir",       default="calibration/frames")
    p.add_argument("--output",        default="calibration/stereo_calibration.json")
    p.add_argument("--from_images",   action="store_true",
                   help="Skip capture, run calibration from existing images in img_dir")
    main(p.parse_args())
```

### Step 8 — Create `src/extract_stereo_3d.py`

This script takes synchronized video pairs and produces skeletons with real
3D coordinates by triangulating MediaPipe detections from both views.

```python
"""
extract_stereo_3d.py
Extract real-3D skeletons from synchronized front + 45° video pairs.
Uses stereo calibration to triangulate joint positions.

Usage:
    python src/extract_stereo_3d.py \
        --front_video  data/merged_v1/p08_c3.mp4 \
        --angle_video  data/merged_v1/p08_c2.mp4 \
        --calib        calibration/stereo_calibration.json \
        --output_csv   data/skeletons/stereo_3d/p08_c3_stereo.csv \
        --output_label data/labels/stereo_3d/p08_c3_stereo.json

    # Batch all matched pairs:
    python src/extract_stereo_3d.py --batch \
        --front_dir  data/merged_v1 \
        --angle_dir  data/merged_v1 \
        --front_suffix _c3 \
        --angle_suffix _c2 \
        --calib      calibration/stereo_calibration.json \
        --output_skel  data/skeletons/stereo_3d \
        --output_label data/labels/stereo_3d
"""
import argparse
import json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

LOWER_BODY_INDICES = {
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}
JOINT_NAMES = list(LOWER_BODY_INDICES.keys())


def load_calibration(path: str) -> dict:
    with open(path) as f:
        c = json.load(f)
    return {
        "K_a": np.array(c["K_a"]),
        "D_a": np.array(c["D_a"]),
        "K_b": np.array(c["K_b"]),
        "D_b": np.array(c["D_b"]),
        "R":   np.array(c["R"]),
        "T":   np.array(c["T"]),
    }


def build_projection_matrices(calib: dict):
    """Compute 3×4 projection matrices for both cameras."""
    K_a, K_b = calib["K_a"], calib["K_b"]
    R,   T   = calib["R"],   calib["T"]

    # Camera A is world origin
    P_a = K_a @ np.hstack([np.eye(3), np.zeros((3,1))])
    # Camera B is rotated/translated from A
    P_b = K_b @ np.hstack([R, T.reshape(3,1)])
    return P_a, P_b


def triangulate_joints(pts_a: np.ndarray, pts_b: np.ndarray,
                       P_a: np.ndarray, P_b: np.ndarray) -> np.ndarray:
    """
    Triangulate N joints from two sets of 2D points.
    pts_a, pts_b: (N, 2) in pixel coordinates
    Returns: (N, 3) in world coordinates (same units as calibration, mm)
    """
    N = pts_a.shape[0]
    pts3d = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        p4d = cv2.triangulatePoints(
            P_a, P_b,
            pts_a[i].reshape(2,1).astype(np.float64),
            pts_b[i].reshape(2,1).astype(np.float64),
        )
        pts3d[i] = (p4d[:3] / p4d[3]).flatten()
    return pts3d


def detect_landmarks(frame: np.ndarray, pose) -> np.ndarray | None:
    """
    Run MediaPipe Pose on a frame.
    Returns (33, 2) pixel coordinates or None if not detected.
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe not installed")
    h, w = frame.shape[:2]
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark
    return np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)


def normalize_skeleton(joints_3d: np.ndarray) -> tuple:
    """
    Center on mid-hip, scale by torso length.
    joints_3d: (10, 3)
    Returns: (normalized (10,3), mid_hip (3,), torso_length float)
    """
    mid_hip = (joints_3d[0] + joints_3d[1]) / 2.0   # left+right hip
    centered = joints_3d - mid_hip

    # Torso length: distance mid-hip to mid-shoulder
    # We don't have shoulders in lower-body set — use hip-to-knee distance as proxy
    knee_mid = (joints_3d[2] + joints_3d[3]) / 2.0
    torso_len = float(np.linalg.norm(knee_mid - mid_hip))
    if torso_len < 1e-6:
        torso_len = 1.0

    normalized = centered / torso_len
    return normalized, mid_hip, torso_len


def process_video_pair(front_path: str, angle_path: str,
                       calib: dict, output_csv: str):
    """
    Extract synchronized stereo-3D skeleton from a video pair.
    front_path  = front camera (c3) video
    angle_path  = 45° camera (c2) video
    output_csv  = where to write the 10-joint skeleton CSV
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe required for Phase 2")

    mp_pose = mp.solutions.pose

    P_a, P_b = build_projection_matrices(calib)

    cap_a = cv2.VideoCapture(front_path)
    cap_b = cv2.VideoCapture(angle_path)

    rows = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        while True:
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()
            if not ret_a or not ret_b:
                break

            lm_a = detect_landmarks(frame_a, pose)
            lm_b = detect_landmarks(frame_b, pose)

            row = {"frame_idx": frame_idx, "pose_detected": 0}
            for jname in JOINT_NAMES:
                for ax in ("x", "y", "z"):
                    row[f"{jname}_{ax}"] = np.nan

            if lm_a is not None and lm_b is not None:
                # Extract lower-body landmarks from both views
                indices = [LOWER_BODY_INDICES[j] for j in JOINT_NAMES]
                pts_a = lm_a[indices]   # (10, 2)
                pts_b = lm_b[indices]   # (10, 2)

                # Triangulate to real 3D
                joints_3d = triangulate_joints(pts_a, pts_b, P_a, P_b)  # (10, 3)

                # Normalize
                normed, mid_hip, torso_len = normalize_skeleton(joints_3d)

                row["pose_detected"] = 1
                for i, jname in enumerate(JOINT_NAMES):
                    row[f"{jname}_x"] = float(normed[i, 0])
                    row[f"{jname}_y"] = float(normed[i, 1])
                    row[f"{jname}_z"] = float(normed[i, 2])

                row["mid_hip_x"]    = float(mid_hip[0])
                row["mid_hip_y"]    = float(mid_hip[1])
                row["mid_hip_z"]    = float(mid_hip[2])
                row["torso_length"] = torso_len

            rows.append(row)
            frame_idx += 1

    cap_a.release()
    cap_b.release()

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    n_detected = int(df["pose_detected"].sum())
    pct = 100 * n_detected / max(len(df), 1)
    print(f"  {Path(output_csv).name}: {len(df)} frames, "
          f"{n_detected} detected ({pct:.1f}%)")
    return {"frames": len(df), "detected": n_detected, "pct": pct}


def main(args):
    calib = load_calibration(args.calib)
    print(f"Calibration loaded from {args.calib}")

    if not args.batch:
        # Single pair
        process_video_pair(
            args.front_video, args.angle_video,
            calib, args.output_csv,
        )
        # Copy label JSON if available
        label_src = Path(args.front_video).with_suffix(".json")
        if label_src.exists() and args.output_label:
            import shutil
            shutil.copy2(label_src, args.output_label)
    else:
        # Batch: find all front videos and match with angle videos
        front_dir = Path(args.front_dir)
        angle_dir = Path(args.angle_dir)
        out_skel  = Path(args.output_skel)
        out_label = Path(args.output_label)
        out_skel.mkdir(parents=True, exist_ok=True)
        out_label.mkdir(parents=True, exist_ok=True)

        front_vids = sorted(front_dir.glob(f"*{args.front_suffix}.mp4"))
        print(f"Found {len(front_vids)} front videos")

        ok, skipped = 0, 0
        for fv in front_vids:
            stem_base = fv.stem.replace(args.front_suffix, "")
            av = angle_dir / f"{stem_base}{args.angle_suffix}.mp4"
            if not av.exists():
                print(f"  SKIP {fv.name} — no matching angle video")
                skipped += 1
                continue

            out_csv   = out_skel  / f"{stem_base}_stereo3d.csv"
            out_json  = out_label / f"{stem_base}_stereo3d.json"
            label_src = front_dir / f"{fv.stem}.json"

            try:
                process_video_pair(str(fv), str(av), calib, str(out_csv))
                if label_src.exists():
                    import shutil
                    shutil.copy2(label_src, out_json)
                ok += 1
            except Exception as e:
                print(f"  ERR {fv.name}: {e}")
                skipped += 1

        print(f"\nDone: {ok} processed, {skipped} skipped.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calib",          default="calibration/stereo_calibration.json")
    # Single mode
    p.add_argument("--front_video",    default=None)
    p.add_argument("--angle_video",    default=None)
    p.add_argument("--output_csv",     default=None)
    p.add_argument("--output_label",   default=None)
    # Batch mode
    p.add_argument("--batch",          action="store_true")
    p.add_argument("--front_dir",      default="data/merged_v1")
    p.add_argument("--angle_dir",      default="data/merged_v1")
    p.add_argument("--front_suffix",   default="_c3")
    p.add_argument("--angle_suffix",   default="_c2")
    p.add_argument("--output_skel",    default="data/skeletons/stereo_3d")
    p.add_argument("--output_label",   default="data/labels/stereo_3d")
    main(p.parse_args())
```

### Step 9 — Create `src/test_stereo.py`

This evaluates whether real-3D stereo skeletons match front-camera accuracy.

```python
"""
test_stereo.py
Compare stereo-3D skeleton accuracy against front-camera and transformed-45 baselines.
Also optionally retrains on stereo data and reports generalization.

Usage:
    python src/test_stereo.py --mode eval
    python src/test_stereo.py --mode retrain
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Reuse evaluate() from test_transform.py
sys.path.insert(0, str(Path(__file__).parent))
from test_transform import evaluate, print_result, DEFAULT_CLASSES


def main(args):
    model = args.model
    print(f"\nModel: {model}")

    # Evaluate stereo-3D skeletons against existing baselines
    results = {}

    if Path(args.front_skel).exists():
        results["front"]     = evaluate(model, args.front_skel, args.front_label)
    if Path(args.tf45_skel).exists():
        results["tf45"]      = evaluate(model, args.tf45_skel,  args.tf45_label)
    if Path(args.stereo_skel).exists():
        results["stereo_3d"] = evaluate(model, args.stereo_skel, args.stereo_label)

    print_result("A — Front (baseline)",       results.get("front"))
    print_result("B — Transformed 45° (Phase1)", results.get("tf45"))
    print_result("C — Stereo 3D (Phase2)",      results.get("stereo_3d"))

    # Summary
    print(f"\n{'='*60}")
    print("  PHASE 2 VERDICT")
    r_c = results.get("stereo_3d")
    r_b = results.get("tf45")
    r_a = results.get("front")
    if r_c and r_b and r_a:
        gain_over_tf45 = (r_c["acc"] - r_b["acc"]) * 100
        gap_from_front = (r_a["acc"] - r_c["acc"]) * 100
        print(f"  Stereo vs Transformed-45°: {gain_over_tf45:+.2f}%")
        print(f"  Gap from front camera:     {gap_from_front:+.2f}%")
        if r_c["acc"] >= r_a["acc"] * 0.90:
            print("  ✅ PHASE 2 SUCCESS — stereo reaches ≥90% of front accuracy.")
            print("     The transform layer is production-ready for the 45° camera.")
        elif r_c["acc"] > r_b["acc"]:
            print("  ✅ IMPROVEMENT — stereo z helps over Phase 1.")
            print("     Consider retraining on stereo data to close remaining gap.")
        else:
            print("  ⚠️  No improvement over Phase 1.")
            print("     Check calibration RMS — may need recapture.")

    if args.mode == "retrain":
        print("\nRetraining on stereo_3d data ...")
        cmd = [
            sys.executable, "src/train_v2.py",
            "--skeleton_dir", args.stereo_skel,
            "--label_dir",    args.stereo_label,
            "--output_dir",   "models_stereo",
            "--log_dir",      "logs_stereo",
            "--epochs",       "200",
            "--warmup_epochs","20",
            "--batch_size",   "32",
            "--cnn_out",      "64",
            "--lstm_hidden",  "0",
            "--dropout",      "0.6",
            "--lr",           "3e-4",
            "--device",       "cuda",
            "--num_workers",  "4",
            "--train_all",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="models_final/fern_v2.onnx")
    p.add_argument("--front_skel",   default="data/skeletons/front")
    p.add_argument("--front_label",  default="data/labels/front")
    p.add_argument("--tf45_skel",    default="data/skeletons/transformed_45")
    p.add_argument("--tf45_label",   default="data/labels/transformed_45")
    p.add_argument("--stereo_skel",  default="data/skeletons/stereo_3d")
    p.add_argument("--stereo_label", default="data/labels/stereo_3d")
    p.add_argument("--mode",         default="eval", choices=["eval","retrain"])
    main(p.parse_args())
```

### Step 10 — Run Phase 2 pipeline

```powershell
cd C:\fern\FERN_V2
$env:PYTHONPATH = "C:\fern\FERN_V2\src"
.\venv\Scripts\Activate.ps1

# --- CALIBRATION SESSION (do this once, physically with both cameras) ---

# 10.1 Find your camera IDs
python src\calibrate_stereo.py --list_cameras

# 10.2 Run calibration session
#      Front camera ID and 45° camera ID from step 10.1
python src\calibrate_stereo.py `
    --cam_front 0 `
    --cam_45    1 `
    --angle     45.0 `
    --img_dir   calibration\frames `
    --output    calibration\stereo_calibration.json

# IMPORTANT: Check the RMS error printed at the end.
# RMS < 1.0 px = excellent
# RMS 1.0–1.5 px = acceptable
# RMS > 1.5 px = recapture (more diverse checkerboard positions)

# --- STEREO EXTRACTION ---

# 10.3 Test on one pair first
python src\extract_stereo_3d.py `
    --front_video data\merged_v1\p08_c3.mp4 `
    --angle_video data\merged_v1\p08_c2.mp4 `
    --calib       calibration\stereo_calibration.json `
    --output_csv  data\skeletons\stereo_3d\p08_stereo3d.csv `
    --output_label data\labels\stereo_3d\p08_stereo3d.json

# 10.4 If test looks good, run batch
python src\extract_stereo_3d.py --batch `
    --front_dir   data\merged_v1 `
    --angle_dir   data\merged_v1 `
    --front_suffix _c3 `
    --angle_suffix _c2 `
    --calib       calibration\stereo_calibration.json `
    --output_skel  data\skeletons\stereo_3d `
    --output_label data\labels\stereo_3d

# 10.5 Evaluate
python src\test_stereo.py 2>&1 | Tee-Object -FilePath logs\phase2_result.txt
```

---

## Decision Tree & Success Criteria

```
Phase 1 result (test_transform.py):
│
├── C (transformed 45°) >= 65%  →  ✅ Strong pass
│                                    Proceed to Phase 2
│                                    MediaPipe z is good enough
│
├── C >= 50%             →  ✅ Pass
│                             Proceed to Phase 2
│                             Stereo z will improve further
│
├── C 35–50%             →  ⚠️ Marginal
│                             Investigate camera angle accuracy
│                             Proceed to Phase 2 cautiously
│
└── C < 35%              →  ❌ STOP
                              Do NOT proceed to Phase 2
                              Debug: check z values are non-zero,
                              verify camera angle is exactly 45°,
                              check JOINT_NAMES order in CSVs


Phase 2 result (test_stereo.py):
│
├── stereo >= 90% of front  →  ✅ Production ready
│                               The transform layer can be used in infer_v2.py
│
├── stereo > Phase 1        →  ✅ Improvement
│                               Retrain model on stereo data for better results
│
└── stereo <= Phase 1       →  ⚠️ No gain
                                Check calibration RMS, recapture if >1.5 px
```

---

## Files Created Summary

| File | Phase | Purpose |
|------|-------|---------|
| `src/transform_skeleton.py` | 1 | Core R_y(-θ) math + CSV transform |
| `src/apply_transform_batch.py` | 1 | Batch process directory of CSVs |
| `src/test_transform.py` | 1 | A/B/C evaluation + verdict |
| `src/calibrate_stereo.py` | 2 | Interactive checkerboard calibration |
| `src/extract_stereo_3d.py` | 2 | Triangulate real-3D joints from video pairs |
| `src/test_stereo.py` | 2 | Compare stereo vs Phase 1 vs front |

## Important Notes for Agent

1. `transform_skeleton.py` must be importable from PYTHONPATH (`C:\fern\FERN_V2\src`).
   All other scripts import from it.

2. Do not modify `dataset_v2.py` or `model_v2.py`. The model takes z=0 input.
   The transform pipeline produces z=0 output. They are compatible.

3. `test_transform.py` uses `SkeletonWindowDataset` with `split="all"` —
   this is intentional (we want to see how the model handles the full 45° set,
   not just a held-out split).

4. Phase 2 requires both cameras to be recording simultaneously and in sync.
   If testing on pre-recorded videos that were not synchronized, triangulation
   will produce junk. Sync matters.

5. The calibration session (Step 10.2) requires physical presence with both
   cameras and a printed checkerboard. It cannot be skipped or mocked.
   The agent should stop at Step 10.2 and wait for the user to run it manually.
   Resume from Step 10.3 after `calibration/stereo_calibration.json` exists.
