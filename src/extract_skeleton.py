"""
FERN v2 — Step 1: Extract skeleton sequences from raw videos.

This script runs MediaPipe Pose on every video file in a source directory
and writes one CSV file per video.  Each row in the CSV is one frame.
Each column is one joint coordinate.

We use only the 10 lower-body joints (indices 23-32 in MediaPipe Pose):
    23 left_hip         24 right_hip
    25 left_knee        26 right_knee
    27 left_ankle       28 right_ankle
    29 left_heel        30 right_heel
    31 left_foot_index  32 right_foot_index

Plus we read joints 11 (left_shoulder) and 12 (right_shoulder) to compute
the torso length for scale normalization — but we do NOT keep them as model
input features.

Column naming convention: joint_name_x, joint_name_y, joint_name_z
Frame rows where MediaPipe fails to detect a pose are filled with NaN.

Usage
-----
python src/extract_skeleton.py \
    --video_dir  data/raw \
    --output_dir data/skeletons \
    --show       (optional: display annotated video while processing)
"""

import argparse
import os
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Joint configuration
# ---------------------------------------------------------------------------

# Indices of joints we keep as model input features.
FEATURE_JOINT_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Human-readable names aligned with the indices above.
FEATURE_JOINT_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Additional joints read only for normalization (not model input).
NORM_LEFT_SHOULDER  = 11
NORM_RIGHT_SHOULDER = 12

NUM_FEATURE_JOINTS = len(FEATURE_JOINT_INDICES)   # 10
NUM_FEATURES       = NUM_FEATURE_JOINTS * 3        # 30 (x, y, z per joint)

# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------

def build_csv_header():
    header = ["frame_idx", "pose_detected"]
    for name in FEATURE_JOINT_NAMES:
        for axis in ("x", "y", "z"):
            header.append(f"{name}_{axis}")
    # Also store the normalization reference values for diagnostics.
    header += ["mid_hip_x", "mid_hip_y", "mid_hip_z", "torso_length"]
    return header


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def compute_mid_hip(landmarks):
    """Return the midpoint between left_hip (23) and right_hip (24)."""
    lh = landmarks[23]
    rh = landmarks[24]
    return (
        (lh.x + rh.x) / 2.0,
        (lh.y + rh.y) / 2.0,
        (lh.z + rh.z) / 2.0,
    )


def compute_torso_length(landmarks):
    """
    Torso length = distance from mid-hip to mid-shoulder.
    Used as a scale reference so that joint coordinates are independent
    of how far the person stands from the camera.
    Returns 1.0 if the value would be zero (defensive).
    """
    mid_hip_x, mid_hip_y, mid_hip_z = compute_mid_hip(landmarks)

    ls = landmarks[NORM_LEFT_SHOULDER]
    rs = landmarks[NORM_RIGHT_SHOULDER]
    mid_shoulder = (
        (ls.x + rs.x) / 2.0,
        (ls.y + rs.y) / 2.0,
        (ls.z + rs.z) / 2.0,
    )

    dx = mid_shoulder[0] - mid_hip_x
    dy = mid_shoulder[1] - mid_hip_y
    dz = mid_shoulder[2] - mid_hip_z
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    return length if length > 1e-6 else 1.0


def normalize_landmarks(landmarks):
    """
    Center on mid-hip and scale by torso length.
    Returns a flat list: [j0_x, j0_y, j0_z, j1_x, ...] for FEATURE joints,
    plus (mid_hip_x, mid_hip_y, mid_hip_z, torso_length).
    """
    mx, my, mz = compute_mid_hip(landmarks)
    torso      = compute_torso_length(landmarks)

    coords = []
    for idx in FEATURE_JOINT_INDICES:
        lm = landmarks[idx]
        coords.append((lm.x - mx) / torso)
        coords.append((lm.y - my) / torso)
        coords.append((lm.z - mz) / torso)

    return coords, (mx, my, mz, torso)


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_video(video_path: str, output_csv: str, show: bool = False):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    MODEL_PATH = os.path.join(
        os.path.expanduser("~"),
        ".cache", "mediapipe", "models", "pose_landmarker_heavy.task"
    )
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        return 0, 0.0

    options = PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0, 0.0

    fps       = cap.get(cv2.CAP_PROP_FPS)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected  = 0
    frame_idx = 0
    header    = build_csv_header()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp  = int(frame_idx * 1000 / fps)
                result     = landmarker.detect_for_video(mp_image, timestamp)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lms    = result.pose_landmarks[0]
                    coords, norm_ref = normalize_landmarks(lms)
                    row    = [frame_idx, 1] + coords + list(norm_ref)
                    detected += 1
                else:
                    nan_coords = [float("nan")] * NUM_FEATURES
                    nan_ref    = [float("nan")] * 4
                    row = [frame_idx, 0] + nan_coords + nan_ref

                writer.writerow(row)

                if show:
                    h, w = frame.shape[:2]
                    if h > 800:
                        scale = 800 / h
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    cv2.imshow("FERN v2 - Extraction", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()

    detection_rate = detected / frame_idx if frame_idx > 0 else 0.0
    return frame_idx, detection_rate
    

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


def find_videos(video_dir: str):
    """Recursively find all video files under video_dir."""
    videos = []
    for root, _, files in os.walk(video_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                videos.append(os.path.join(root, fname))
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe skeleton sequences from raw videos."
    )
    parser.add_argument("--video_dir",  required=True,
                        help="Root folder of raw videos.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write skeleton CSV files.")
    parser.add_argument("--show", action="store_true",
                        help="Display annotated video during processing.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    videos = find_videos(args.video_dir)
    if not videos:
        print(f"No videos found under {args.video_dir}")
        return

    print(f"Found {len(videos)} video(s).  Starting extraction...")
    print()

    total_frames    = 0
    total_detected  = 0

    for i, vpath in enumerate(videos, 1):
        # Mirror the subfolder structure so labels are preserved.
        rel       = os.path.relpath(vpath, args.video_dir)
        csv_rel   = str(Path(rel).with_suffix(".csv"))
        csv_path  = os.path.join(args.output_dir, csv_rel)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        print(f"[{i}/{len(videos)}] {rel}")
        frames, rate = extract_video(vpath, csv_path, show=args.show)
        print(f"         {frames} frames  |  detection rate: {rate*100:.1f}%")

        if rate < 0.5:
            print(f"  WARNING: Low detection rate.  Check camera angle and lighting.")

        total_frames   += frames
        total_detected += int(rate * frames)

    overall_rate = total_detected / total_frames if total_frames > 0 else 0
    print()
    print(f"Done.  {total_frames} total frames, "
          f"overall detection rate: {overall_rate*100:.1f}%")
    print(f"CSV files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
