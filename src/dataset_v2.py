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
import re
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

def load_skeleton_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a skeleton CSV into a float32 numpy array of shape (T, 30).

    Returns (data, detected) where:
      data     — (T, 30) float32, interpolated + fillna(0)
      detected — (T,) bool: True where pose_detected == 1
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

    if "pose_detected" in df.columns:
        detected = df["pose_detected"].values != 0
        data[~detected] = np.nan
    else:
        detected = np.ones(len(df), dtype=bool)

    # Interpolate NaN values along the time axis (axis 0).
    data_df = pd.DataFrame(data, columns=FEATURE_COLS)
    data_df.interpolate(method="linear", axis=0, inplace=True)
    data_df.fillna(0.0, inplace=True)   # fill any remaining (e.g. start/end)

    return data_df.values.astype(np.float32), detected


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
    "foot_hold",        # class 0 — idle/neutral stance (G07)
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
        split_mode:   str  = "random",
        val_ratio:    float = 0.15,
        test_ratio:   float = 0.10,
        seed:         int  = 42,
        augment:      bool = False,
        n_cameras:    int  = 1,
    ):
        self.skeleton_dir = skeleton_dir
        self.label_dir    = label_dir
        self.classes      = classes or DEFAULT_CLASSES
        self.class_map    = build_class_map(self.classes)
        self.num_classes  = len(self.classes)
        self.window_size  = window_size
        self.stride       = stride
        self.split        = split
        self.split_mode   = split_mode
        self.augment      = augment
        self.n_cameras    = n_cameras

        self.samples: List[Tuple[np.ndarray, int]] = []
        self.cam_ids: List[int] = []
        self.sample_sources: List[str] = []
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
                    pairs.append((Path(csv_path), Path(json_path)))
        return pairs

    def _split_pairs(self, pairs, seed, val_ratio, test_ratio):
        if self.split_mode == "random":
            return self._random_split_pairs(pairs, seed, val_ratio, test_ratio)
        return self._subject_split_pairs(pairs, seed, val_ratio, test_ratio)

    def _random_split_pairs(self, pairs, seed, val_ratio, test_ratio):
        """
        Random video-level split: mirrors stay paired with their originals.
        No subject tracking — each video is independent.
        """
        rng = np.random.RandomState(seed)

        # Group originals + mirrors together
        groups = {}
        for csv_path, json_path in pairs:
            stem = Path(csv_path).stem
            base = stem.replace("_mirror", "")
            groups.setdefault(base, []).append((csv_path, json_path))

        keys = sorted(groups.keys())
        rng.shuffle(keys)

        n      = len(keys)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))

        splits = {"train": [], "val": [], "test": [], "all": []}
        for i, base in enumerate(keys):
            for pair in groups[base]:
                splits["all"].append(pair)
            if i < n_test:
                splits["test"].extend(groups[base])
            elif i < n_test + n_val:
                splits["val"].extend(groups[base])
            else:
                splits["train"].extend(groups[base])

        return splits

    def _subject_split_pairs(self, pairs, seed, val_ratio, test_ratio):
        """
        Subject-independent split: all camera angles + mirrors of the same
        subject stay in the same split. Cross-subject files (grouped_by_gesture)
        always go to train.
        """
        rng = np.random.RandomState(seed)

        # Separate subject-based (merged_v1: p{NN}_c{N}) from
        # cross-subject (grouped_by_gesture: gesture names).
        subject_map = {}
        cross_subject = []

        for csv_path, json_path in pairs:
            stem = Path(csv_path).stem
            base = stem.replace("_mirror", "")
            m = re.match(r"^(p\d{2})_c\d", base)
            if m:
                subject_map.setdefault(m.group(1), []).append((csv_path, json_path))
            else:
                cross_subject.append((csv_path, json_path))

        subjects = sorted(subject_map.keys())
        rng.shuffle(subjects)

        n      = len(subjects)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))

        train = list(cross_subject)   # grouped_by_gesture → train only
        val   = []
        test  = []

        for s in subjects[:n_test]:
            test.extend(subject_map[s])
        for s in subjects[n_test:n_test + n_val]:
            val.extend(subject_map[s])
        for s in subjects[n_test + n_val:]:
            train.extend(subject_map[s])

        shuffled = list(cross_subject)
        for s in subjects:
            shuffled.extend(subject_map[s])

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
                skeleton, detected = load_skeleton_csv(csv_path)
                with open(json_path) as fh:
                    label_data = json.load(fh)
                segments  = label_data["segments"]
            except Exception as e:
                print(f"  WARNING: Skipping {csv_path}: {e}")
                continue

            cam_id = int(label_data.get("camera_id", 0))
            if cam_id >= self.n_cameras:
                print(f"  WARNING: cam_id={cam_id} >= n_cameras={self.n_cameras} in {csv_path.name}, setting to 0")
                cam_id = 0
            T = len(skeleton)

            segments = sorted(segments, key=lambda s: s["start_frame"])
            short_count = 0
            prev_end = -1

            for seg in segments:
                gesture = seg["gesture"]
                if gesture not in self.class_map:
                    unknown_gestures.add(gesture)
                    continue
                label = self.class_map[gesture]
                s     = seg["start_frame"]
                e     = seg["end_frame"]

                if s >= T:
                    print(f"  WARNING: start_frame={s} >= T={T} in {csv_path.name}, skipping segment")
                    continue
                if s > e:
                    print(f"  WARNING: start_frame={s} > end_frame={e} in {csv_path.name}, skipping segment")
                    continue

                seg_len = e - s + 1
                if seg_len < self.window_size:
                    short_count += 1

                if s <= prev_end:
                    print(f"  WARNING: Overlapping segments in {csv_path.name}: "
                          f"'{gesture}' starts at {s} but previous ends at {prev_end}")
                prev_end = e

                e = min(e, T - 1)

                pos = s
                while pos + self.window_size <= e + 1:
                    window = skeleton[pos: pos + self.window_size]
                    det_ratio = detected[pos: pos + self.window_size].mean()
                    if det_ratio < 0.7:
                        skipped_windows += 1
                    else:
                        self.samples.append((window.copy(), label))
                        self.cam_ids.append(cam_id)
                        self.sample_sources.append(csv_path)
                    pos += self.stride

        if unknown_gestures:
            print(f"  WARNING: Unknown gestures ignored: {unknown_gestures}")
        if short_count:
            print(f"  WARNING: {short_count} segments < window_size={self.window_size}.")
        if skipped_windows:
            print(f"  INFO: Skipped {skipped_windows} high-NaN windows.")

        print(f"  Dataset [{self.split}]: {len(self.samples)} windows "
              f"from {len(selected)} video(s).")

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Apply random perturbations at training time.

        Transforms:
          1. Time warp (±10%)
          2. Speed variation (resample 80-120% of frames)
          3. Frame dropout (drop 5-10% of frames)
          4. Temporal shift (crop/pad 5-10 frames from edges)
          5. Joint dropout: randomly zero out 1 joint
          6. Joint noise (σ=0.005)
          7. Random scaling (0.9-1.1)
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

        # --- Speed variation (resample 80-120% of frames) ---
        if np.random.random() < 0.4:
            speed_factor = np.random.uniform(0.8, 1.2)
            new_T = max(2, int(T * speed_factor))
            src = np.linspace(0, T - 1, new_T)
            x_speed = np.zeros((new_T, F), dtype=x.dtype)
            for f in range(F):
                x_speed[:, f] = np.interp(
                    np.linspace(0, new_T - 1, T), src,
                    np.interp(src, np.arange(T), x[:, f]))
            # Pad or crop back to T
            if new_T > T:
                x = x_speed[:T]
            else:
                pad = np.zeros((T - new_T, F), dtype=x.dtype)
                x = np.concatenate([x_speed, pad], axis=0)

        # --- Frame dropout (drop 5-10% of frames, replace with zero) ---
        if np.random.random() < 0.3:
            n_drop = int(T * np.random.uniform(0.05, 0.10))
            drop_idx = np.random.choice(T, size=n_drop, replace=False)
            x[drop_idx] = 0.0

        # --- Temporal shift (crop/pad 5-10 frames from edges) ---
        if np.random.random() < 0.3:
            shift = np.random.randint(5, 11)
            if np.random.random() < 0.5:
                # crop from start, pad end
                x = np.concatenate([x[shift:],
                                    np.zeros((shift, F), dtype=x.dtype)], axis=0)
            else:
                # crop from end, pad start
                x = np.concatenate([np.zeros((shift, F), dtype=x.dtype),
                                    x[:T - shift]], axis=0)

        # --- Joint dropout: zero out 1 random joint's 3 coords ---
        if np.random.random() < 0.2:
            joint = np.random.randint(0, 10)
            x[:, joint*3:(joint+1)*3] = 0.0

        # --- Joint noise (σ=0.005) ---
        noise = np.random.normal(0, 0.005, size=x.shape).astype(np.float32)
        x = x + noise

        # --- Random scaling (0.9-1.1) ---
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

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

        if self.n_cameras > 1:
            cam_id  = self.cam_ids[idx]
            one_hot = np.zeros(self.n_cameras, dtype=np.float32)
            one_hot[cam_id] = 1.0
            flag    = np.tile(one_hot, (self.window_size, 1))
            x       = np.concatenate([x, flag], axis=-1)

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
