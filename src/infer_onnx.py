"""
ONNX Runtime live/streaming inference. No PyTorch dependency.

Feature dim is auto-detected from the ONNX graph input shape.
Supports 30-dim (front-only) and 30+n-dim (camera-flag) models.
"""

import argparse
import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

from extract_skeleton import NUM_FEATURES, normalize_landmarks
from dataset_v2 import DEFAULT_CLASSES


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class StreamingClassifier:
    """Reusable streaming wrapper: push frames, poll predictions."""

    def __init__(self, onnx_path, window_size=60, stride=10,
                 camera_view=None, threshold=0.5):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.feat_dim = int(inp.shape[2])
        self.n_cameras = self.feat_dim - NUM_FEATURES
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold
        self.buffer = collections.deque(maxlen=window_size)
        self._last_valid = None
        self._since_pred = 0

        if self.n_cameras > 0:
            if camera_view is None or not (0 <= camera_view < self.n_cameras):
                raise ValueError(
                    f"Model expects camera flag (n_cameras={self.n_cameras}); "
                    f"pass camera_view in [0..{self.n_cameras-1}].")
            one_hot = np.zeros(self.n_cameras, dtype=np.float32)
            one_hot[camera_view] = 1.0
            self.cam_flag = np.tile(one_hot, (window_size, 1))
        else:
            self.cam_flag = None

    def push(self, feats):
        """
        feats: (30,) float32 or None if pose not detected.
        Returns (label, confidence) when a new prediction fires, else None.
        """
        if feats is None:
            feats = self._last_valid if self._last_valid is not None \
                    else np.zeros(NUM_FEATURES, dtype=np.float32)
        else:
            feats = feats.copy()
            feats[2::3] = 0.0
            self._last_valid = feats

        self.buffer.append(feats)
        self._since_pred += 1

        if len(self.buffer) < self.window_size or self._since_pred < self.stride:
            return None
        self._since_pred = 0

        window = np.stack(list(self.buffer)).astype(np.float32)
        if self.cam_flag is not None:
            window = np.concatenate([window, self.cam_flag], axis=-1)

        logits = self.session.run(None, {self.input_name: window[None]})[0][0]
        probs = softmax(logits)
        idx = int(probs.argmax())
        conf = float(probs[idx])
        label = DEFAULT_CLASSES[idx] if conf >= self.threshold else "uncertain"
        return label, conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path", required=True)
    p.add_argument("--camera_id", type=str, default="0")
    p.add_argument("--camera_view", type=int, default=None,
                   help="One-hot index: 0=front(c3), 1=side(c2) — required for multi-camera models")
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    clf = StreamingClassifier(
        args.onnx_path, args.window_size, args.stride,
        args.camera_view, args.threshold,
    )
    print(f"Model feat_dim={clf.feat_dim}  n_cameras={clf.n_cameras}")

    try:
        cap = cv2.VideoCapture(int(args.camera_id), cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera_id}")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    label, conf = "waiting...", 0.0
    fps_t = collections.deque(maxlen=30)

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
            ok, frame = cap.read()
            if not ok:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                coords, _ = normalize_landmarks(results.pose_landmarks.landmark)
                out = clf.push(np.asarray(coords, dtype=np.float32))
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(255, 255, 0), thickness=2
                    ),
                )
            else:
                out = clf.push(None)

            if out is not None:
                label, conf = out

            fps_t.append(time.time() - t0)
            fps = 1.0 / (sum(fps_t) / len(fps_t))

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
            color = (0, 255, 0) if conf >= args.threshold else (0, 165, 255)
            cv2.putText(frame, f"Gesture: {label}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(frame, f"Conf: {conf*100:.1f}%  FPS: {fps:.1f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 2)
            cv2.imshow("FERN v2 — ONNX Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
