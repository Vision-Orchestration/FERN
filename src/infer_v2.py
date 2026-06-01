"""
FERN v2 — Step 6: Live inference from a camera.

Uses a rolling buffer of skeleton frames.  MediaPipe extracts the skeleton
from each camera frame.  Once the buffer fills a full window, the model
makes a prediction.  The window slides by `stride` frames each time,
so predictions are updated frequently without waiting for a full window.

Usage
-----
python src/infer_v2.py \
    --model       models/fern_v2_best.pth \
    --camera_id   0 \
    --window_size 60 \
    --stride      10 \
    --threshold   0.6
"""

import argparse
import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

from extract_skeleton import (
    FEATURE_JOINT_INDICES, FEATURE_JOINT_NAMES, NUM_FEATURES,
    normalize_landmarks,
)
from dataset_v2 import DEFAULT_CLASSES
from model_v2   import FERNv2


# ---------------------------------------------------------------------------
# Rolling skeleton buffer
# ---------------------------------------------------------------------------

class SkeletonBuffer:
    """
    Maintains a deque of the last `window_size` skeleton frames.
    Call `push(frame_features)` for each camera frame.
    Call `ready()` to check if the buffer is full.
    Call `get_window()` to retrieve the current window as (T, F) ndarray.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)

    def push(self, features: np.ndarray):
        self.buffer.append(features)

    def ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        return np.stack(list(self.buffer), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-frame skeleton extraction (reuses extract_skeleton helpers)
# ---------------------------------------------------------------------------

def frame_to_features(results) -> np.ndarray:
    """
    Convert MediaPipe pose results for one frame to a normalized feature
    vector of shape (NUM_FEATURES,) = (30,).
    Returns zeros if pose was not detected.
    """
    if not results.pose_landmarks:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    coords, _ = normalize_landmarks(results.pose_landmarks.landmark)
    return np.array(coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model.
    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    model = FERNv2(
        num_joints=10,
        num_classes=len(DEFAULT_CLASSES),
        cnn_out=saved.get("cnn_out", 64),
        lstm_hidden=saved.get("lstm_hidden", 128),
        lstm_layers=saved.get("lstm_layers", 2),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Camera.
    try:
        src = int(args.camera_id)
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    except ValueError:
        src = args.camera_id
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        return

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    buffer     = SkeletonBuffer(args.window_size)

    # State.
    current_label  = "waiting..."
    current_conf   = 0.0
    frames_since_pred = 0
    fps_times = collections.deque(maxlen=30)

    print("Running.  Press Q to quit.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 1 = faster for real-time
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            t0  = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Extract skeleton.
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            feats   = frame_to_features(results)
            buffer.push(feats)

            # Draw skeleton overlay.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 0), thickness=2
                    ),
                )

            # Make a prediction every `stride` frames once buffer is full.
            frames_since_pred += 1
            if buffer.ready() and frames_since_pred >= args.stride:
                frames_since_pred = 0
                window = buffer.get_window()
                x      = torch.from_numpy(window).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

                best_idx  = int(probs.argmax())
                best_conf = float(probs[best_idx])

                if best_conf >= args.threshold:
                    current_label = DEFAULT_CLASSES[best_idx]
                    current_conf  = best_conf
                else:
                    current_label = "uncertain"
                    current_conf  = best_conf

            # FPS
            fps_times.append(time.time() - t0)
            avg_fps = 1.0 / (sum(fps_times) / len(fps_times))

            # Draw HUD.
            h, w = frame.shape[:2]

            # Background rectangle for readability.
            cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)

            color = (0, 255, 0) if current_conf >= args.threshold else (0, 165, 255)
            cv2.putText(frame, f"Gesture: {current_label}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(frame, f"Confidence: {current_conf*100:.1f}%",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}  Buffer: {len(buffer.buffer)}/{args.window_size}",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (150, 150, 150), 1)
            
            # Scale vertical videos to fit screen
            h, w = frame.shape[:2]
            if h > w:  # vertical
                scale = 800 / h
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            cv2.imshow("FERN v2 — Live Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--camera_id",   type=str,   default="0")
    p.add_argument("--window_size", type=int,   default=60)
    p.add_argument("--stride",      type=int,   default=10)
    p.add_argument("--threshold",   type=float, default=0.6)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
