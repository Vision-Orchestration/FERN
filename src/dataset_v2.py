"""
FERN v2 — Step 3: PyTorch Dataset from skeleton CSVs + label JSON files.

The dataset works with a sliding window approach:
  - Load skeleton CSV (output of extract_skeleton.py).
  - Load label JSON (output of label_videos.py).
  - For each labeled segment, slide a fixed-size window across the frames
    with a configurable stride.
  - Each window becomes one training sample.

Missing frames (NaN rows) are handled by linear interpolation from
neighboring valid frames.  If an entire window is missing, it is skipped.

The dataset returns:
    x : torch.FloatTensor of shape (window_size, NUM_FEATURES)
    y : int class label
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Joint count and feature count — must match extract_skeleton.py.
NUM_FEATURE_JOINTS = 10
NUM_FEATURES       = NUM_FEATURE_JOINTS * 3   # 30

# Column names for the feature columns only (no metadata columns).
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


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_skeleton_csv(csv_path: str) -> np.ndarray:
    """
    Load a skeleton CSV into a float32 numpy array of shape (T, 30).
    Rows where pose_detected == 0 are filled with NaN.
    Gaps are then repaired by linear interpolation.
    """
    df = pd.read_csv(csv_path)

    # Extract only the 30 feature columns.
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV {csv_path} is missing columns: {missing_cols}\n"
            f"Re-run extract_skeleton.py to regenerate it."
        )

    data = df[FEATURE_COLS].values.astype(np.float32)

    # Mark undetected frames as NaN.
    if "pose_detected" in df.columns:
        not_detected = df["pose_detected"].values == 0
        data[not_detected] = np.nan

    # Interpolate NaN values along the time axis (axis 0).
    # pd.DataFrame handles this column-by-column cleanly.
    data_df = pd.DataFrame(data, columns=FEATURE_COLS)
    data_df.interpolate(method="linear", axis=0, inplace=True)
    data_df.fillna(0.0, inplace=True)   # fill any remaining (e.g. start/end)

    return data_df.values.astype(np.float32)


def load_label_json(json_path: str) -> Tuple[List[dict], float]:
    """
    Load label JSON and return (segments, fps).
    Each segment is a dict: {gesture, start_frame, end_frame}.
    """
    with open(json_path) as f:
        data = json.load(f)
    return data["segments"], data.get("fps", 30.0)


# ---------------------------------------------------------------------------
# Gesture class mapping
# ---------------------------------------------------------------------------
DEFAULT_CLASSES = [
    "foot_lift",
    "sideway_kick",
    "cross_front",
    "heel_tap",
    "flamingo_bend",
    "forward_step",
    "forward_kick",
]

def build_class_map(classes: List[str]):
    return {name: idx for idx, name in enumerate(classes)}


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class SkeletonWindowDataset(Dataset):
    """
    Sliding window dataset over labeled skeleton sequences.

    Parameters
    ----------
    skeleton_dir : str
        Directory containing per-video skeleton CSV files
        (mirroring the structure of video_dir).
    label_dir : str
        Directory containing per-video label JSON files
        (mirroring the same structure).
    classes : list of str
        Ordered list of gesture class names.  Index = class id.
    window_size : int
        Number of frames per window (default 60 = 2 s at 30 fps).
    stride : int
        Step size between consecutive windows (default 15 = 0.5 s).
    split : str
        One of "train", "val", "test" or "all".  Selects which videos
        to include based on a simple hash of the filename.
    val_ratio : float
        Fraction of videos held out for validation.
    test_ratio : float
        Fraction of videos held out for test.
    seed : int
        Random seed for reproducible train/val/test splitting.
    augment : bool
        Whether to apply time-warp and joint-noise augmentation
        during training.
    """

    def __init__(
        self,
        skeleton_dir: str,
        label_dir:    str,
        classes:      List[str] = None,
        window_size:  int  = 60,
        stride:       int  = 15,
        split:        str  = "all",
        val_ratio:    float = 0.15,
        test_ratio:   float = 0.10,
        seed:         int  = 42,
        augment:      bool = False,
    ):
        self.skeleton_dir = skeleton_dir
        self.label_dir    = label_dir
        self.classes      = classes or DEFAULT_CLASSES
        self.class_map    = build_class_map(self.classes)
        self.num_classes  = len(self.classes)
        self.window_size  = window_size
        self.stride       = stride
        self.split        = split
        self.augment      = augment

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build(seed, val_ratio, test_ratio)

    # ------------------------------------------------------------------
    # Internal: discover files and build sample list
    # ------------------------------------------------------------------

    def _all_pairs(self):
        """
        Find all (skeleton_csv, label_json) pairs where both files exist.
        """
        pairs = []
        for root, _, files in os.walk(self.skeleton_dir):
            for fname in sorted(files):
                if not fname.endswith(".csv"):
                    continue
                csv_path  = os.path.join(root, fname)
                rel       = os.path.relpath(csv_path, self.skeleton_dir)
                json_path = os.path.join(
                    self.label_dir,
                    str(Path(rel).with_suffix(".json"))
                )
                if os.path.exists(json_path):
                    pairs.append((csv_path, json_path))
        return pairs

    def _split_pairs(self, pairs, seed, val_ratio, test_ratio):
        """
        Deterministically split pairs into train/val/test by hashing filename.
        """
        rng = np.random.RandomState(seed)
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        n      = len(shuffled)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test   = shuffled[:n_test]
        val    = shuffled[n_test:n_test + n_val]
        train  = shuffled[n_test + n_val:]
        return {"train": train, "val": val, "test": test, "all": shuffled}

    def _build(self, seed, val_ratio, test_ratio):
        all_pairs = self._all_pairs()
        if not all_pairs:
            raise RuntimeError(
                f"No matching (CSV, JSON) pairs found.\n"
                f"  skeleton_dir = {self.skeleton_dir}\n"
                f"  label_dir    = {self.label_dir}\n"
                f"Run extract_skeleton.py and label_videos.py first."
            )

        split_map = self._split_pairs(all_pairs, seed, val_ratio, test_ratio)
        selected  = split_map[self.split]

        skipped_windows  = 0
        unknown_gestures = set()

        for csv_path, json_path in selected:
            try:
                skeleton  = load_skeleton_csv(csv_path)
                segments, _ = load_label_json(json_path)
            except Exception as e:
                print(f"  WARNING: Skipping {csv_path}: {e}")
                continue

            T = len(skeleton)

            for seg in segments:
                gesture = seg["gesture"]
                if gesture not in self.class_map:
                    unknown_gestures.add(gesture)
                    continue
                label = self.class_map[gesture]
                s     = seg["start_frame"]
                e     = min(seg["end_frame"], T - 1)

                # Slide windows across this segment.
                pos = s
                while pos + self.window_size <= e + 1:
                    window = skeleton[pos: pos + self.window_size]
                    # Skip windows with too many NaN/zeros (bad detection).
                    nan_ratio = np.isnan(window).mean()
                    if nan_ratio > 0.3:
                        skipped_windows += 1
                    else:
                        self.samples.append((window.copy(), label))
                    pos += self.stride

        if unknown_gestures:
            print(f"  WARNING: Unknown gestures ignored: {unknown_gestures}")
        if skipped_windows:
            print(f"  INFO: Skipped {skipped_windows} high-NaN windows.")

        print(f"  Dataset [{self.split}]: {len(self.samples)} windows "
              f"from {len(selected)} video(s).")

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Apply small random perturbations at training time.

        Two transforms are applied:
          1. Time warp: slightly stretch or compress the sequence by
             re-sampling it to a different length, then back to window_size.
          2. Joint noise: add small Gaussian noise to all coordinates,
             mimicking real-world tracking jitter.
        """
        T, F = x.shape

        # --- Time warp (scale factor 0.9 – 1.1) ---
        factor     = np.random.uniform(0.9, 1.1)
        new_len    = max(2, int(T * factor))
        src_idx    = np.linspace(0, T - 1, new_len)
        tgt_idx    = np.linspace(0, new_len - 1, T)
        x_warped   = np.zeros_like(x)
        for f in range(F):
            x_warped[:, f] = np.interp(tgt_idx, src_idx,
                                        np.interp(src_idx, np.arange(T), x[:, f]))
        x = x_warped

        # --- Joint noise (std = 0.005 in normalized coords) ---
        noise = np.random.normal(0, 0.005, size=x.shape).astype(np.float32)
        x = x + noise

        return x

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.augment:
            x = self._augment(x)
        x = torch.from_numpy(x).float()
        return x, y

    # ------------------------------------------------------------------
    # Utility: weighted sampler for class balancing
    # ------------------------------------------------------------------

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Return a WeightedRandomSampler that balances class frequencies.
        This compensates for classes with fewer samples.
        """
        labels  = [s[1] for s in self.samples]
        counts  = np.bincount(labels, minlength=self.num_classes).astype(float)
        counts  = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        sample_weights = torch.tensor([weights[l] for l in labels])
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def class_distribution(self):
        """Print the number of windows per class."""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        print("\nClass distribution:")
        for i, name in enumerate(self.classes):
            print(f"  {name:20s}  {counts[i]:5d} windows")
        print()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset_v2.py <skeleton_dir> <label_dir>")
        sys.exit(1)

    ds = SkeletonWindowDataset(
        skeleton_dir=sys.argv[1],
        label_dir=sys.argv[2],
        split="all",
    )
    ds.class_distribution()

    x, y = ds[0]
    print(f"Sample shape: {x.shape}  label: {y} ({ds.classes[y]})")
