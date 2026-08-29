"""
augment_rotate.py — Generate rotation-augmented skeleton CSVs.

Applies small random Y-axis rotations to existing front-view skeletons
to simulate slight camera angle variation. Uses transform_skeleton.py.

Usage:
    python src/augment_rotate.py \\
        --input_dir  data/skeletons/front \\
        --label_dir  data/labels/front \\
        --output_skel data/skeletons/front_aug \\
        --output_label data/labels/front_aug \\
        --angles 5 10 15 \\
        --mirror

Output: for each input CSV, generates len(angles) rotated versions
        plus their mirrors if --mirror is set.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transform_skeleton import (
    JOINT_NAMES, N_JOINTS, transform_to_front,
)

SUPPORTED = {'.csv'}


def augment_one(csv_path, label_path, output_skel, output_label,
                angles, make_mirror):
    """Read one CSV, generate rotated versions, copy labels."""
    df = pd.read_csv(csv_path)
    total_frames = len(df)

    feature_cols = [
        c for c in df.columns
        if any(c.startswith(j) for j in JOINT_NAMES)
        and c.endswith(("_x", "_y", "_z"))
    ]

    raw = df[feature_cols].values.astype(np.float32)
    nan_mask = np.isnan(raw)
    raw_clean = np.where(nan_mask, 0.0, raw)

    stem = Path(csv_path).stem

    for angle in angles:
        rotated = transform_to_front(raw_clean, camera_angle_deg=float(angle),
                                     zero_z_after=True)
        rotated[nan_mask] = np.nan

        out_df = df.copy()
        out_df[feature_cols] = rotated

        out_name = f"{stem}_rot{angle}.csv"
        out_path = os.path.join(output_skel, out_name)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        # Copy label
        if label_path and os.path.exists(label_path):
            lbl_out = os.path.join(output_label,
                                   str(Path(out_name).with_suffix('.json')))
            Path(lbl_out).parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(label_path, lbl_out)

        if make_mirror:
            mirror_name = f"{stem}_rot{angle}_mirror.csv"
            mirror_path = os.path.join(output_skel, mirror_name)
            _write_mirror(out_df, feature_cols, mirror_path)

            if label_path and os.path.exists(label_path):
                lbl_mirror = os.path.join(
                    output_label,
                    str(Path(mirror_name).with_suffix('.json')))
                shutil.copy2(label_path, lbl_mirror)


def _write_mirror(df, feature_cols, out_path):
    """Negate X, swap left/right joints, write mirror CSV."""
    mirror_df = df.copy()
    data = mirror_df[feature_cols].values.copy().astype(np.float32)

    # LR swap pairs by joint index (each joint = 3 cols)
    LR_SWAP = [
        (0, 1),   # left_hip ↔ right_hip
        (2, 3),   # left_knee ↔ right_knee
        (4, 5),   # left_ankle ↔ right_ankle
        (6, 7),   # left_heel ↔ right_heel
        (8, 9),   # left_foot_index ↔ right_foot_index
    ]

    for l_idx, r_idx in LR_SWAP:
        l_start, r_start = l_idx * 3, r_idx * 3
        # swap and negate X
        tmp_x = data[:, l_start].copy()
        data[:, l_start] = -data[:, r_start]
        data[:, r_start] = -tmp_x
        # swap Y, Z
        for offset in [1, 2]:
            tmp = data[:, l_start + offset].copy()
            data[:, l_start + offset] = data[:, r_start + offset]
            data[:, r_start + offset] = tmp

    # Negate mid_hip_x
    mid_hip_cols = [c for c in feature_cols if c.startswith("mid_hip")]
    if mid_hip_cols:
        mid_hip_x_col = [c for c in mid_hip_cols if c.endswith("_x")]
        if mid_hip_x_col:
            mirror_df[mid_hip_x_col[0]] = -mirror_df[mid_hip_x_col[0]]

    mirror_df[feature_cols] = data
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mirror_df.to_csv(out_path, index=False)


def main():
    p = argparse.ArgumentParser(description="Rotation augmentation")
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--label_dir",  default=None)
    p.add_argument("--output_skel", required=True)
    p.add_argument("--output_label", default=None)
    p.add_argument("--angles", nargs="+", type=float, default=[5, 10, 15],
                   help="Rotation angles in degrees")
    p.add_argument("--mirror", action="store_true",
                   help="Also generate mirrored versions")
    args = p.parse_args()

    label_dir = args.label_dir or args.input_dir.replace("skeletons", "labels")
    output_label = args.output_label or args.output_skel.replace("skeletons", "labels")

    csvs = sorted(Path(args.input_dir).glob("*.csv"))
    print(f"Found {len(csvs)} CSVs in {args.input_dir}")
    print(f"Angles: {args.angles}  Mirror: {args.mirror}")

    total = 0
    for csv_path in csvs:
        stem = csv_path.stem
        # Skip already-augmented files
        if "_rot" in stem or "_mirror" in stem:
            continue

        label_path = Path(label_dir) / f"{stem}.json"
        augment_one(
            str(csv_path), str(label_path) if label_path.exists() else None,
            args.output_skel, output_label,
            args.angles, args.mirror,
        )
        total += 1
        print(f"  {stem} -> {len(args.angles)} rotations"
              f"{' + mirrors' if args.mirror else ''}")

    n_new = total * len(args.angles) * (2 if args.mirror else 1)
    print(f"\nDone. {total} input -> {n_new} new files")
    print(f"Skeletons: {args.output_skel}")
    print(f"Labels:    {output_label}")


if __name__ == '__main__':
    main()
