"""
augment_spatial.py — Generate spatially-augmented skeleton CSVs.

Applies random scaling, vertical shift, and joint perturbation to
existing skeleton CSVs. Generates multiple variants per input.

Usage:
    python src/augment_spatial.py \\
        --input_dir  data/skeletons/front \\
        --label_dir  data/labels/front \\
        --output_skel data/skeletons/front_aug \\
        --output_label data/labels/front_aug \\
        --variants 2 \\
        --mirror
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transform_skeleton import JOINT_NAMES, N_JOINTS

SUPPORTED = {'.csv'}


def _scale(data, factor):
    """Scale all coordinates by factor (simulates distance)."""
    return data * factor


def _vertical_shift(data, max_shift=0.05):
    """Shift all Y coordinates by random offset (simulates camera height)."""
    shift = np.random.uniform(-max_shift, max_shift)
    # Y coords are every 3rd column starting at index 1 (left_hip_y, etc.)
    y_cols = np.arange(1, data.shape[1], 3)
    data[:, y_cols] += shift
    return data


def _joint_perturbation(data, sigma=0.01):
    """Add Gaussian noise to individual joints."""
    noise = np.random.normal(0, sigma, size=data.shape).astype(np.float32)
    return data + noise


def _write_mirror(df, feature_cols, out_path):
    """Negate X, swap left/right joints, write mirror CSV."""
    mirror_df = df.copy()
    data = mirror_df[feature_cols].values.copy().astype(np.float32)

    LR_SWAP = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for l_idx, r_idx in LR_SWAP:
        l_start, r_idx_s = l_idx * 3, r_idx * 3
        tmp_x = data[:, l_start].copy()
        data[:, l_start] = -data[:, r_idx_s]
        data[:, r_idx_s] = -tmp_x
        for offset in [1, 2]:
            tmp = data[:, l_start + offset].copy()
            data[:, l_start + offset] = data[:, r_idx_s + offset]
            data[:, r_idx_s + offset] = tmp

    mid_hip_x_col = [c for c in feature_cols
                      if c.startswith("mid_hip") and c.endswith("_x")]
    if mid_hip_x_col:
        mirror_df[mid_hip_x_col[0]] = -mirror_df[mid_hip_x_col[0]]

    mirror_df[feature_cols] = data
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mirror_df.to_csv(out_path, index=False)


def augment_one(csv_path, label_path, output_skel, output_label,
                n_variants, make_mirror):
    """Generate spatial variants of one CSV."""
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

    for v in range(n_variants):
        aug = raw_clean.copy()

        # 1. Random scale (0.9 – 1.1)
        scale = np.random.uniform(0.9, 1.1)
        aug = _scale(aug, scale)

        # 2. Random vertical shift
        if np.random.random() < 0.5:
            aug = _vertical_shift(aug, max_shift=0.05)

        # 3. Joint perturbation
        if np.random.random() < 0.5:
            aug = _joint_perturbation(aug, sigma=0.01)

        aug[nan_mask] = np.nan

        out_df = df.copy()
        out_df[feature_cols] = aug

        out_name = f"{stem}_sp{v}.csv"
        out_path = os.path.join(output_skel, out_name)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        if label_path and os.path.exists(label_path):
            lbl_out = os.path.join(output_label,
                                   str(Path(out_name).with_suffix('.json')))
            Path(lbl_out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, lbl_out)

        if make_mirror:
            mirror_name = f"{stem}_sp{v}_mirror.csv"
            mirror_path = os.path.join(output_skel, mirror_name)
            _write_mirror(out_df, feature_cols, mirror_path)
            if label_path and os.path.exists(label_path):
                lbl_mirror = os.path.join(
                    output_label,
                    str(Path(mirror_name).with_suffix('.json')))
                shutil.copy2(label_path, lbl_mirror)


def main():
    p = argparse.ArgumentParser(description="Spatial augmentation")
    p.add_argument("--input_dir",   required=True)
    p.add_argument("--label_dir",   default=None)
    p.add_argument("--output_skel", required=True)
    p.add_argument("--output_label", default=None)
    p.add_argument("--variants", type=int, default=2,
                   help="Number of spatial variants per input")
    p.add_argument("--mirror", action="store_true",
                   help="Also generate mirrored versions")
    args = p.parse_args()

    label_dir = args.label_dir or args.input_dir.replace("skeletons", "labels")
    output_label = args.output_label or args.output_skel.replace("skeletons", "labels")

    csvs = sorted(Path(args.input_dir).glob("*.csv"))
    print(f"Found {len(csvs)} CSVs in {args.input_dir}")
    print(f"Variants per file: {args.variants}  Mirror: {args.mirror}")

    total = 0
    for csv_path in csvs:
        stem = csv_path.stem
        if "_sp" in stem or "_mirror" in stem or "_rot" in stem:
            continue

        label_path = Path(label_dir) / f"{stem}.json"
        augment_one(
            str(csv_path), str(label_path) if label_path.exists() else None,
            args.output_skel, output_label,
            args.variants, args.mirror,
        )
        total += 1
        print(f"  {stem} -> {args.variants} variants"
              f"{' + mirrors' if args.mirror else ''}")

    n_new = total * args.variants * (2 if args.mirror else 1)
    print(f"\nDone. {total} input -> {n_new} new files")
    print(f"Skeletons: {args.output_skel}")
    print(f"Labels:    {output_label}")


if __name__ == '__main__':
    main()
