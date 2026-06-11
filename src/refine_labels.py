"""
FERN v2 — Refine label boundaries using skeleton motion signals.

Reads skeleton CSVs and their paired label JSONs, then adjusts segment
boundaries to snap to actual foot-motion onset/offset frames rather than
heuristic clip-boundary estimates.

Method:
  1. Compute per-frame foot speed from ankle/heel/foot_index joints.
  2. Smooth with a moving-average filter.
  3. For each segment boundary, search ±search_window frames for the
     point where speed crosses a threshold.
  4. Snap the boundary to that frame.

Usage:
    python src/refine_labels.py \
        --skeleton_dir data/skeletons/merged_v1 \
        --label_dir    data/labels/merged_v1 \
        --output_dir   data/labels/merged_v1_refined
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Joint names in our 10-joint skeleton CSVs.
FOOT_JOINTS = [
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

FEATURE_COLS = []
JOINT_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
for name in JOINT_NAMES:
    for axis in ("x", "y", "z"):
        FEATURE_COLS.append(f"{name}_{axis}")


def compute_motion_score(csv_path: str, fps: float) -> np.ndarray:
    """
    Load skeleton CSV, compute per-frame foot speed.

    Returns array of shape (T,) where each value is the summed
    L2 displacement of all foot joints between consecutive frames.
    NaN frames are linearly interpolated first.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: CSV missing columns: {missing}", file=sys.stderr)
        return None

    data = df[FEATURE_COLS].values.astype(np.float32)

    # NaN interpolation (same as dataset_v2.py).
    if "pose_detected" in df.columns:
        not_detected = df["pose_detected"].values == 0
        data[not_detected] = np.nan

    data_df = pd.DataFrame(data, columns=FEATURE_COLS)
    data_df.interpolate(method="linear", axis=0, inplace=True)
    data_df.fillna(0.0, inplace=True)

    data = data_df.values.astype(np.float32)

    # Get column indices for foot joints.
    foot_cols = []
    for joint in FOOT_JOINTS:
        for axis in ("x", "y", "z"):
            foot_cols.append(FEATURE_COLS.index(f"{joint}_{axis}"))

    foot_data = data[:, foot_cols]

    # Frame-to-frame displacement (L2 norm per joint, summed).
    diff = np.diff(foot_data, axis=0)
    speed = np.sqrt((diff ** 2).sum(axis=1))  # (T-1,)

    # Pad first frame (speed=0).
    speed = np.concatenate([[0.0], speed])

    return speed


def smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def refine_segments(segments, motion, fps, search=10, threshold_percentile=70):
    """
    Adjust start/end frames of each segment to snap to motion boundaries.

    Parameters
    ----------
    segments : list[dict]
        Each with keys: gesture, start_frame, end_frame.
    motion : np.ndarray
        Per-frame foot speed (T,).
    fps : float
        Frames per second (for threshold scaling).
    search : int
        Frames to search around each boundary.
    threshold_percentile : float
        Percentile of motion signal to use as activity threshold.

    Returns
    -------
    list[dict]
        Refined segments.
    """
    T = len(motion)
    if T == 0:
        return segments

    # Compute threshold: only consider frames between 0 and 95th percentile
    # to avoid outlier spikes.
    low = np.percentile(motion, 30)
    high = np.percentile(motion, 95)
    candidate = motion[(motion >= low) & (motion <= high)]
    if len(candidate) == 0:
        threshold = np.percentile(motion, threshold_percentile)
    else:
        threshold = np.percentile(candidate, threshold_percentile)

    # Minimum threshold floor (fps-dependent).
    threshold = max(threshold, 0.002 * fps)

    new_segments = []
    for seg in segments:
        s = max(0, seg["start_frame"])
        e = min(T - 1, seg["end_frame"])

        # For foot_hold (idle), snap start to where motion DROPS below threshold
        # and end to where motion RISES above threshold.
        if seg["gesture"] == "foot_hold":
            # Search backward from end for motion onset (leave idle).
            lo_end = max(s, e - search)
            if lo_end < e:
                region = motion[lo_end:e + 1]
                # Find first frame where motion crosses above threshold.
                above = np.where(region > threshold)[0]
                if len(above) > 0:
                    e = lo_end + above[0] - 1
                    e = max(s, e)

            # Search forward from start for motion offset (enter idle).
            hi_start = min(e, s + search)
            if s < hi_start:
                region = motion[s:hi_start + 1]
                below = np.where(region < threshold)[0]
                if len(below) > 0:
                    s = s + below[-1]
                    s = min(s, e)

        else:
            # Active gesture: snap start to where motion RISES above threshold.
            lo_start = max(0, s - search)
            if lo_start < s:
                region = motion[lo_start:s + 1]
                above = np.where(region > threshold)[0]
                if len(above) > 0:
                    s = lo_start + above[0]
                    s = max(0, s)

            # Snap end to where motion DROPS below threshold.
            hi_end = min(T - 1, e + search)
            if e < hi_end:
                region = motion[e:hi_end + 1]
                below = np.where(region < threshold)[0]
                if len(below) > 0:
                    e = e + below[0]
                    e = min(T - 1, e)

        new_segments.append({
            "gesture": seg["gesture"],
            "start_frame": int(s),
            "end_frame": int(e),
        })

    # Safety: ensure no segment has end < start (motion signal can cause
    # boundary crossing on low-detection videos).
    for i in range(len(new_segments)):
        s = new_segments[i]["start_frame"]
        e = new_segments[i]["end_frame"]
        if e < s:
            e = s + 30
        if e >= T:
            e = max(s, T - 1)
        if e < s:
            e = s  # absolute fallback — 1-frame segment
        new_segments[i]["end_frame"] = e
        new_segments[i]["start_frame"] = s
        if i > 0 and s < new_segments[i-1]["end_frame"]:
            new_segments[i]["start_frame"] = new_segments[i-1]["end_frame"] + 1

    return new_segments


def refine_one(skeleton_csv: str, label_json: str, output_json: str,
               search: int = 10, threshold_percentile: float = 70):
    """Refine labels for one video using its skeleton CSV."""
    # Load label JSON.
    with open(label_json) as f:
        label = json.load(f)

    fps = label.get("fps", 30.0)

    # Load skeleton and compute motion.
    if not os.path.exists(skeleton_csv):
        print(f"  SKIP: no skeleton at {skeleton_csv}")
        return False

    motion = compute_motion_score(skeleton_csv, fps)
    if motion is None or len(motion) < 5:
        print(f"  SKIP: insufficient motion data in {skeleton_csv}")
        return False

    # Smooth.
    motion = smooth(motion, window=max(3, int(fps / 6)))

    # Refine segments.
    segments = label.get("segments", [])
    if not segments:
        print(f"  SKIP: no segments in {label_json}")
        return False

    old_frames = [(s["start_frame"], s["end_frame"]) for s in segments]
    new_segments = refine_segments(
        segments, motion, fps,
        search=search,
        threshold_percentile=threshold_percentile,
    )
    new_frames = [(s["start_frame"], s["end_frame"]) for s in new_segments]

    # Check if anything changed.
    if old_frames == new_frames:
        return False  # no change

    label["segments"] = new_segments
    label["refined_by"] = "refine_labels.py"
    label["refinement_search"] = search
    label["refinement_threshold_pct"] = threshold_percentile

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(label, f, indent=2)

    total_shift = sum(
        abs(ns - os) + abs(ne - oe)
        for (os, oe), (ns, ne) in zip(old_frames, new_frames)
    )
    print(f"  Refined: {os.path.basename(output_json)}  "
          f"(total shift: {total_shift} frames)")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Refine label boundaries using skeleton motion signals."
    )
    p.add_argument("--skeleton_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--search", type=int, default=10,
                    help="Frames to search around each boundary (default: 10)")
    p.add_argument("--threshold_percentile", type=float, default=70,
                    help="Motion percentile for activity threshold (default: 70)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all label JSONs.
    label_paths = sorted(Path(args.label_dir).glob("*.json"))
    if not label_paths:
        # Also check for nested structure.
        label_paths = sorted(Path(args.label_dir).rglob("*.json"))

    print(f"Found {len(label_paths)} label JSON(s).\n")

    refined = 0
    unchanged = 0
    skipped = 0

    for lpath in label_paths:
        # Corresponding skeleton CSV.
        rel = lpath.relative_to(args.label_dir)
        csv_name = rel.with_suffix(".csv")
        csv_path = Path(args.skeleton_dir) / csv_name

        out_path = Path(args.output_dir) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if refine_one(str(csv_path), str(lpath), str(out_path),
                      search=args.search,
                      threshold_percentile=args.threshold_percentile):
            refined += 1
        elif not csv_path.exists():
            skipped += 1
        else:
            unchanged += 1

    print(f"\nDone.  Refined: {refined}  Unchanged: {unchanged}  "
          f"Skipped (no skeleton): {skipped}")


if __name__ == "__main__":
    main()
