"""
Mirror augmentation for 10-joint FERN v2 skeleton CSVs + label JSONs.

For each skeleton CSV + matching label JSON:
  1. Negate X coordinate of every joint
  2. Swap left↔right joint pairs
  3. Negate mid_hip_x
  4. Write mirrored skeleton CSV as <stem>_mirror.csv
  5. Copy label JSON as <stem>_mirror.json (frames unchanged)
"""

import argparse
import csv
import json
import os
from pathlib import Path

# 10-joint left↔right pairs (by column index in the CSV, skipping frame_idx, pose_detected)
# Each tuple is (left_col_start, right_col_start) — 3 coords (x,y,z) each
# Column layout after frame_idx(0), pose_detected(1):
#   0-1  : left_hip_x,y,z
#   3-5  : right_hip_x,y,z
#   6-8  : left_knee_x,y,z
#   9-11 : right_knee_x,y,z
#   12-14: left_ankle_x,y,z
#   15-17: right_ankle_x,y,z
#   18-20: left_heel_x,y,z
#   21-23: right_heel_x,y,z
#   24-26: left_foot_index_x,y,z
#   27-29: right_foot_index_x,y,z
#   30-32: mid_hip_x,y,z
#   33   : torso_length
LR_SWAP_PAIRS = [
    (0, 3),   # hip
    (6, 9),   # knee
    (12, 15), # ankle
    (18, 21), # heel
    (24, 27), # foot_index
]

def mirror_skeleton_csv(in_path: str, out_path: str):
    """Mirror a skeleton CSV: negate X, swap left↔right."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(in_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if len(row) < 34:
                continue
            # Coords start at index 2
            coords = [float(v) if v else 0.0 for v in row[2:]]
            # Negate all X coords (every 3rd starting from 0)
            for i in range(0, len(coords), 3):
                coords[i] = -coords[i]
            # Swap left↔right
            for l, r in LR_SWAP_PAIRS:
                for off in range(3):
                    coords[l + off], coords[r + off] = coords[r + off], coords[l + off]
            # Write mirrored row
            out_row = row[:2] + [f"{v:.12f}" for v in coords]
            writer.writerow(out_row)

def main():
    ap = argparse.ArgumentParser(description="Mirror 10-joint skeletons and labels")
    ap.add_argument("--skeleton_dir", default="data/skeletons/merged_v1")
    ap.add_argument("--label_dir", default="data/labels/merged_v1")
    ap.add_argument("--output_skel", default="data/skeletons/merged_v1")
    ap.add_argument("--output_label", default="data/labels/merged_v1")
    args = ap.parse_args()

    skel_dir = Path(args.skeleton_dir)
    label_dir = Path(args.label_dir)
    out_skel = Path(args.output_skel)
    out_label = Path(args.output_label)

    csv_paths = sorted(skel_dir.glob("*.csv"))
    mirrored = 0
    skipped = 0

    for csv_path in csv_paths:
        stem = csv_path.stem
        if stem.endswith("_mirror"):
            continue
        label_path = label_dir / f"{stem}.json"
        if not label_path.exists():
            skipped += 1
            continue

        out_csv = out_skel / f"{stem}_mirror.csv"
        out_json = out_label / f"{stem}_mirror.json"

        mirror_skeleton_csv(str(csv_path), str(out_csv))

        # Copy label JSON unchanged
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(label_path) as f:
            label = json.load(f)
        label["mirror_of"] = stem
        with open(out_json, "w") as f:
            json.dump(label, f, indent=2)

        mirrored += 1
        print(f"  Mirrored: {stem}")

    print(f"\nDone. Mirrored: {mirrored}  Skipped (no label): {skipped}")

if __name__ == "__main__":
    main()
