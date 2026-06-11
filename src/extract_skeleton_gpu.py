"""
FERN v2 — GPU-accelerated skeleton extraction using PyTorch (YOLOv8 pose).

Runs on RTX 3070 via PyTorch CUDA. Falls back to CPU if no GPU.

Usage:
    python src/extract_skeleton_gpu.py \
        --video_dir  data/merged_v1 \
        --output_dir data/skeletons/merged_v1
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# YOLOv8-pose keypoint indices (COCO 17)
# 0:nose 1:L-eye 2:R-eye 3:L-ear 4:R-ear
# 5:L-shoulder 6:R-shoulder 7:L-elbow 8:R-elbow 9:L-wrist 10:R-wrist
# 11:L-hip 12:R-hip 13:L-knee 14:R-knee 15:L-ankle 16:R-ankle
COCO_LOWER = [
    (11, "left_hip"), (12, "right_hip"),
    (13, "left_knee"), (14, "right_knee"),
    (15, "left_ankle"), (16, "right_ankle"),
]

JOINT_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

NUM_FEATURES = 30


def build_csv_header():
    header = ["frame_idx", "pose_detected"]
    for name in JOINT_NAMES:
        for axis in ("x", "y", "z"):
            header.append(f"{name}_{axis}")
    header += ["mid_hip_x", "mid_hip_y", "mid_hip_z", "torso_length"]
    return header


def compute_torso_length(kps, w, h):
    """Torso = distance mid-shoulder to mid-hip in pixels."""
    ls = kps[5][:2]; rs = kps[6][:2]
    lh = kps[11][:2]; rh = kps[12][:2]
    mid_s = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_h = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    dx = (mid_s[0] - mid_h[0]); dy = (mid_s[1] - mid_h[1])
    return math.sqrt(dx * dx + dy * dy) or 1.0


def extract_video_gpu(video_path, output_csv, model, device, show=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected = 0
    frame_idx = 0
    header = build_csv_header()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # YOLOv8 expects BGR, 640x640 with letterbox
            results = model(frame, imgsz=640, verbose=False)

            if (len(results) > 0 and results[0].keypoints is not None and
                results[0].keypoints.xy is not None and
                len(results[0].keypoints.xy) > 0 and
                results[0].keypoints.conf is not None and
                len(results[0].keypoints.conf) > 0):
                kps = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2)
                confs = results[0].keypoints.conf[0].cpu().numpy()  # (17,)
                avg_conf = confs.mean()

                if avg_conf > 0.3 and len(kps) >= 17:
                    detected += 1
                    mid_hip_x = (kps[11][0] + kps[12][0]) / (w * 2)
                    mid_hip_y = (kps[11][1] + kps[12][1]) / (h * 2)
                    torso = compute_torso_length(kps, w, h)

                    coords = []
                    for idx, name in COCO_LOWER:
                        kx = (kps[idx][0] / w - mid_hip_x) / torso * w
                        ky = (kps[idx][1] / h - mid_hip_y) / torso * h
                        coords.extend([kx, ky, 0.0])

                    # Approximate heel: behind ankle
                    la = coords[12:15]; ra = coords[15:18]
                    lh = [la[0] - 0.02, la[1] + 0.01, 0.0]
                    rh = [ra[0] - 0.02, ra[1] + 0.01, 0.0]
                    li = [la[0] + 0.03, la[1] + 0.02, 0.0]
                    ri = [ra[0] + 0.03, ra[1] + 0.02, 0.0]

                    all_c = coords + lh + rh + li + ri
                    row = [frame_idx, 1] + all_c + [mid_hip_x, mid_hip_y, 0.0, torso]
                else:
                    row = [frame_idx, 0] + [float("nan")] * NUM_FEATURES + [float("nan")] * 4
            else:
                row = [frame_idx, 0] + [float("nan")] * NUM_FEATURES + [float("nan")] * 4

            writer.writerow(row)
            frame_idx += 1

            if show and frame_idx % 30 == 0:
                disp = frame.copy()
                if len(results) > 0 and results[0].keypoints is not None:
                    kps = results[0].keypoints.xy[0].cpu().numpy()
                    for kp in kps:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
                cv2.imshow("FERN - GPU Extract", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    rate = detected / frame_idx if frame_idx > 0 else 0.0
    return frame_idx, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--model", default="yolov8n-pose",
                        help="yolov8n-pose / yolov8s-pose / yolov8m-pose")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    from ultralytics import YOLO
    print(f"Loading {args.model}...", end=" ", flush=True)
    t0 = time.time()
    model = YOLO(f"{args.model}.pt")
    model.to(device)
    print(f"{time.time()-t0:.0f}s")

    found = []
    for root, _, files in os.walk(args.video_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in {".mp4", ".mov", ".avi"}:
                found.append(os.path.join(root, fname))

    done = {p.stem for p in Path(args.output_dir).glob("*.csv")}
    found = [v for v in found if Path(v).stem not in done]

    if not found:
        print("Done — no remaining videos.")
        return

    print(f"Remaining: {len(found)} videos\n")

    total_fr = 0
    total_det = 0
    for i, vp in enumerate(found, 1):
        rel = os.path.relpath(vp, args.video_dir)
        csv_p = os.path.join(args.output_dir, str(Path(rel).with_suffix(".csv")))
        os.makedirs(os.path.dirname(csv_p), exist_ok=True)

        t0 = time.time()
        print(f"[{i}/{len(found)}] {rel} ... ", end="", flush=True)
        fr, rate = extract_video_gpu(vp, csv_p, model, device, show=args.show)
        print(f"{fr} fr, {rate*100:.1f}%, {time.time()-t0:.0f}s")
        total_fr += fr
        total_det += int(rate * fr)

    overall = total_det / total_fr if total_fr else 0
    print(f"\nDone. {total_fr} frames, {overall*100:.1f}% detection.")


if __name__ == "__main__":
    main()
