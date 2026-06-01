"""
FERN v2 — Dataset augmentation: left-right skeleton mirror.

For every skeleton CSV + label JSON in the new database, this script:
  1. Negates the normalised X coordinate of every landmark
  2. Swaps left/right joint index pairs (MediaPipe BlazePose convention)
  3. Writes a mirrored CSV  → <stem>_mirror.csv
  4. Copies the label JSON  → <stem>_mirror.json  (frame indices unchanged)

Usage
-----
python src/mirror_dataset.py \
    --skeleton_dir data/skeletons \
    --label_dir    data/labels \
    --output_skel  data/skeletons \
    --output_label data/labels

Only files whose label JSON exists are mirrored (skips un-labeled skeletons).
Already-mirrored files (_mirror suffix) are skipped to avoid double-mirroring.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

# ── MediaPipe BlazePose left↔right joint swap map ─────────────────────────────
# Each tuple is (left_idx, right_idx).  We swap coordinates for every pair.
LR_SWAP_PAIRS = [
    (1,  4),   # inner eye
    (2,  5),   # eye
    (3,  6),   # outer eye
    (7,  8),   # ear
    (9,  10),  # mouth corner
    (11, 12),  # shoulder
    (13, 14),  # elbow
    (15, 16),  # wrist
    (17, 18),  # pinky
    (19, 20),  # index finger
    (21, 22),  # thumb
    (23, 24),  # hip
    (25, 26),  # knee
    (27, 28),  # ankle
    (29, 30),  # heel
    (31, 32),  # foot index
]

N_LANDMARKS = 33
# Each landmark has x, y, z → 3 values.  CSV columns after frame/detected:
# x0 y0 z0  x1 y1 z1 … x32 y32 z32  ref_hip_x ref_hip_y ref_torso_len visibility_mean
COORD_START  = 2                    # column index where x0 begins
COORDS_PER_LM = 3                   # x, y, z
N_COORD_COLS  = N_LANDMARKS * COORDS_PER_LM   # 99 columns
REF_COLS      = 4                   # normalisation reference columns at the end


def _swap_and_mirror(row_vals: list) -> list:
    """
    row_vals: the numeric part of one CSV row (everything after frame/detected).
    Returns a new list with X negated and left/right pairs swapped.
    """
    coords = np.array(row_vals[:N_COORD_COLS], dtype=float)
    ref    = row_vals[N_COORD_COLS:]          # keep reference columns as-is

    # Reshape to (33, 3) for easy indexing
    lm = coords.reshape(N_LANDMARKS, COORDS_PER_LM).copy()

    # 1. Negate X for all landmarks
    lm[:, 0] *= -1.0

    # 2. Swap left↔right pairs
    for l_idx, r_idx in LR_SWAP_PAIRS:
        lm[l_idx], lm[r_idx] = lm[r_idx].copy(), lm[l_idx].copy()

    return lm.flatten().tolist() + ref


def mirror_skeleton(src_csv: str, dst_csv: str) -> int:
    """Mirror one skeleton CSV.  Returns number of rows written."""
    with open(src_csv, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)

    with open(dst_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            frame_idx = row[0]
            detected  = row[1]
            numeric   = [float(v) if v not in ('', 'nan') else float('nan')
                         for v in row[COORD_START:]]

            if detected == '1' and not any(
                    v != v for v in numeric[:N_COORD_COLS]):  # no NaN
                mirrored = _swap_and_mirror(numeric)
            else:
                mirrored = numeric   # keep NaN rows as-is

            writer.writerow([frame_idx, detected] + mirrored)

    return len(rows)


def mirror_label(src_json: str, dst_json: str, mirror_video_path: str):
    """Copy label JSON, updating video_path to the mirror name."""
    with open(src_json) as f:
        data = json.load(f)

    data['video_path'] = mirror_video_path
    data['mirrored']   = True

    with open(dst_json, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description='Mirror skeleton CSVs and label JSONs left↔right.')
    p.add_argument('--skeleton_dir',  required=True,
                   help='Folder containing skeleton CSVs.')
    p.add_argument('--label_dir',     required=True,
                   help='Folder containing label JSONs.')
    p.add_argument('--output_skel',   required=True,
                   help='Output folder for mirrored CSVs (can be same as input).')
    p.add_argument('--output_label',  required=True,
                   help='Output folder for mirrored JSONs (can be same as input).')
    args = p.parse_args()

    os.makedirs(args.output_skel,  exist_ok=True)
    os.makedirs(args.output_label, exist_ok=True)

    csv_files = sorted(Path(args.skeleton_dir).glob('*.csv'))
    skipped   = []
    processed = []

    for csv_path in csv_files:
        stem = csv_path.stem

        # Skip already-mirrored files
        if stem.endswith('_mirror'):
            skipped.append(str(csv_path.name) + '  (already mirrored)')
            continue

        # Skip if no matching label JSON
        json_path = Path(args.label_dir) / (stem + '.json')
        if not json_path.exists():
            skipped.append(str(csv_path.name) + '  (no label JSON)')
            continue

        dst_csv  = Path(args.output_skel)  / (stem + '_mirror.csv')
        dst_json = Path(args.output_label) / (stem + '_mirror.json')

        n_rows = mirror_skeleton(str(csv_path), str(dst_csv))
        mirror_label(str(json_path), str(dst_json),
                     mirror_video_path=stem + '_mirror')

        processed.append((csv_path.name, n_rows))
        print(f'  Mirrored  {csv_path.name}  ({n_rows} frames)')

    print()
    print(f'Done.  {len(processed)} file(s) mirrored, '
          f'{len(skipped)} skipped.')
    if skipped:
        print('Skipped:')
        for s in skipped:
            print(f'  {s}')


if __name__ == '__main__':
    main()
