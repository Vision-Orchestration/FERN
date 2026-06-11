"""
FERN v2 — Live inference from a camera.

Supports both 30-dim (front-only) and 30+n_cameras-dim (camera-flag) models.
Model hyperparameters are read from the checkpoint's saved args.
"""

import argparse
import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

from extract_skeleton import NUM_FEATURES, normalize_landmarks
from dataset_v2 import DEFAULT_CLASSES
from model_v2 import FERNv2


# ---------------------------------------------------------------------------
# Rolling skeleton buffer
# ---------------------------------------------------------------------------

class SkeletonBuffer:
    """
    Maintains a deque of the last `window_size` skeleton frames.

    When a frame has no detection, repeats the last valid frame
    (causal stand-in for training-time interpolation).
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)
        self._last_valid = None

    def push(self, features: np.ndarray, valid: bool):
        if not valid:
            features = self._last_valid if self._last_valid is not None \
                       else np.zeros(NUM_FEATURES, dtype=np.float32)
        else:
            self._last_valid = features
        self.buffer.append(features)

    def ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        return np.stack(list(self.buffer), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-frame extraction
# ---------------------------------------------------------------------------

def frame_to_features(results):
    """
    Convert MediaPipe pose results to (features, valid).

    features — (NUM_FEATURES,) float32, z is zeroed to match training data
    valid    — bool, False if pose not detected
    """
    if not results.pose_landmarks:
        return np.zeros(NUM_FEATURES, dtype=np.float32), False

    coords, _ = normalize_landmarks(results.pose_landmarks.landmark)
    feats = np.array(coords, dtype=np.float32)
    feats[2::3] = 0.0       # zero z — training data has z=0 everywhere
    return feats, True


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    """Load model, auto-detecting n_cameras and input_features from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}) or {}

    n_cameras = int(saved.get("n_cameras", 1))
    input_features = NUM_FEATURES + (n_cameras if n_cameras > 1 else 0)

    model = FERNv2(
        num_joints=10,
        num_classes=len(DEFAULT_CLASSES),
        cnn_out=saved.get("cnn_out", 64),
        lstm_hidden=saved.get("lstm_hidden", 0),
        lstm_layers=saved.get("lstm_layers", 1),
        dropout=0.0,
        input_features=input_features,
    ).to(device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, n_cameras


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_cameras = load_model(args.model, device)
    print(f"Model: n_cameras={n_cameras}")

    # Pre-compute camera-ID flag if needed.
    cam_flag = None
    if n_cameras > 1:
        if args.camera_view is None or not (0 <= args.camera_view < n_cameras):
            raise SystemExit(
                f"Model trained with n_cameras={n_cameras}. "
                f"Requires --camera_view in [0..{n_cameras-1}] "
                f"(0=front/c3, 1=side/c2).")
        one_hot = np.zeros(n_cameras, dtype=np.float32)
        one_hot[args.camera_view] = 1.0
        cam_flag = np.tile(one_hot, (args.window_size, 1))

    # Camera.
    try:
        cap = cv2.VideoCapture(int(args.camera_id), cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    buffer = SkeletonBuffer(args.window_size)

    current_label = "waiting..."
    current_conf = 0.0
    frames_since_pred = 0
    fps_times = collections.deque(maxlen=30)

    print("Running.  Press Q to quit.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feats, valid = frame_to_features(results)
            buffer.push(feats, valid)

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

            frames_since_pred += 1
            if buffer.ready() and frames_since_pred >= args.stride:
                frames_since_pred = 0
                window = buffer.get_window()
                if cam_flag is not None:
                    window = np.concatenate([window, cam_flag], axis=-1)

                x = torch.from_numpy(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                best_idx = int(probs.argmax())
                best_conf = float(probs[best_idx])
                current_conf = best_conf
                current_label = DEFAULT_CLASSES[best_idx] \
                    if best_conf >= args.threshold else "uncertain"

            fps_times.append(time.time() - t0)
            avg_fps = 1.0 / (sum(fps_times) / len(fps_times))

            h, w = frame.shape[:2]
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

            if h > w:
                scale = 800 / h
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            cv2.imshow("FERN v2 — Live Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--camera_id", type=str, default="0")
    p.add_argument("--camera_view", type=int, default=None,
                   help="One-hot index: 0=front(c3), 1=side(c2) — required for multi-camera models")
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.6)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
