"""
augment_interpolate.py — Generate temporally-interpolated skeleton CSVs.

Creates new frames by linearly interpolating between consecutive real
frames. Useful for slow gestures (foot_hold, flamingo_bend) where the
motion is smooth and interpolation adds realistic variation.

Usage:
    python src/augment_interpolate.py \\
        --input_dir  data/skeletons/front \\
        --label_dir  data/labels/front \\
        --output_skel data/skeletons/front_aug \\
        --output_label data/labels/front_aug \\
        --factor 2 \\
        --gestures foot_hold flamingo_bend \\
        --label_dir_check data/labels/front
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transform_skeleton import JOINT_NAMES, N_JOINTS

SUPPORTED = {'.csv'}


def interpolate_frames(data, factor):
    """Interpolate to increase frame count by factor.

    factor=2 means 1 new frame between each pair of real frames.
    """
    T, F = data.shape
    new_T = (T - 1) * factor + 1
    result = np.zeros((new_T, F), dtype=np.float32)

    for i in range(new_T):
        # Map new index back to original time
        t_orig = i / factor
        lo = int(t_orig)
        hi = min(lo + 1, T - 1)
        alpha = t_orig - lo
        result[i] = data[lo] * (1 - alpha) + data[hi] * alpha

    return result


def _should_augment(csv_path, label_path, gestures):
    """Check if this file's gesture matches the target list."""
    if not gestures:
        return True  # augment everything
    if not label_path or not Path(label_path).exists():
        return False
    with open(label_path) as f:
        data = json.load(f)
    segs = data.get("segments", [])
    if not segs:
        return False
    file_gestures = set(s.get("gesture", "") for s in segs)
    return bool(file_gestures & set(gestures))


def augment_one(csv_path, label_path, output_skel, output_label, factor):
    """Generate an interpolated version of one CSV."""
    df = pd.read_csv(csv_path)
    stem = Path(csv_path).stem

    feature_cols = [
        c for c in df.columns
        if any(c.startswith(j) for j in JOINT_NAMES)
        and c.endswith(("_x", "_y", "_z"))
    ]

    raw = df[feature_cols].values.astype(np.float32)
    nan_mask = np.isnan(raw)
    raw_clean = np.where(nan_mask, 0.0, raw)

    # Interpolate
    interp = interpolate_frames(raw_clean, factor)
    # Propagate nan mask to interpolated frames (interpolate as float, threshold back)
    interp_mask = interpolate_frames(nan_mask.astype(np.float32), factor) > 0.5
    interp[interp_mask] = np.nan

    # Build new DataFrame with adjusted frame_idx
    T_new = len(interp)
    out_df = pd.DataFrame()
    out_df["frame_idx"] = np.arange(T_new)

    # pose_detected: repeat each value `factor` times
    if "pose_detected" in df.columns:
        out_df["pose_detected"] = np.repeat(
            df["pose_detected"].values, factor)[:T_new]
    else:
        out_df["pose_detected"] = 1

    out_df[feature_cols] = interp

    # Copy non-feature, non-frame columns
    meta_cols = [c for c in df.columns
                 if c not in feature_cols
                 and c not in ("frame_idx", "pose_detected")]
    for col in meta_cols:
        out_df[col] = np.repeat(df[col].values, factor)[:T_new] \
            if len(df[col].values) > 1 else df[col].values[0]

    out_name = f"{stem}_interp{factor}.csv"
    out_path = os.path.join(output_skel, out_name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Adjust label segment frame indices
    if label_path and Path(label_path).exists():
        with open(label_path) as f:
            lbl = json.load(f)
        lbl_new = dict(lbl)
        lbl_new["total_frames"] = T_new
        new_segs = []
        for seg in lbl.get("segments", []):
            new_segs.append({
                "gesture":     seg["gesture"],
                "start_frame": seg["start_frame"] * factor,
                "end_frame":   min(seg["end_frame"] * factor, T_new - 1),
            })
        lbl_new["segments"] = new_segs

        lbl_out = os.path.join(output_label,
                               str(Path(out_name).with_suffix('.json')))
        Path(lbl_out).parent.mkdir(parents=True, exist_ok=True)
        with open(lbl_out, 'w') as f:
            json.dump(lbl_new, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Temporal interpolation augmentation")
    p.add_argument("--input_dir",   required=True)
    p.add_argument("--label_dir",   default=None)
    p.add_argument("--output_skel", required=True)
    p.add_argument("--output_label", default=None)
    p.add_argument("--factor", type=int, default=2,
                   help="Interpolation factor (2 = double frames)")
    p.add_argument("--gestures", nargs="*", default=None,
                   help="Only augment files containing these gestures "
                        "(default: augment all)")
    args = p.parse_args()

    label_dir = args.label_dir or args.input_dir.replace("skeletons", "labels")
    output_label = args.output_label or args.output_skel.replace("skeletons", "labels")

    csvs = sorted(Path(args.input_dir).glob("*.csv"))
    print(f"Found {len(csvs)} CSVs in {args.input_dir}")
    print(f"Factor: {args.factor}x  Target gestures: {args.gestures or 'all'}")

    total = 0
    skipped = 0
    for csv_path in csvs:
        stem = csv_path.stem
        if "_interp" in stem or "_mirror" in stem or "_rot" in stem or "_sp" in stem:
            continue

        label_path = Path(label_dir) / f"{stem}.json"

        if not _should_augment(str(csv_path), label_path, args.gestures):
            skipped += 1
            continue

        augment_one(
            str(csv_path),
            str(label_path) if label_path.exists() else None,
            args.output_skel, output_label,
            args.factor,
        )
        total += 1
        print(f"  {stem} -> interp{args.factor}")

    print(f"\nDone. {total} augmented, {skipped} skipped")
    print(f"Skeletons: {args.output_skel}")
    print(f"Labels:    {output_label}")


if __name__ == '__main__':
    main()
