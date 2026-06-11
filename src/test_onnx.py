"""Test ONNX model accuracy on a dataset, with optional camera-ID flag support."""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2 import (
    load_skeleton_csv, load_label_json, DEFAULT_CLASSES, build_class_map
)


def test(
    onnx_path: str,
    skeleton_dir: str,
    label_dir: str,
    n_cameras: int = 1,
    window_size: int = 60,
    stride: int = 15,
):
    class_map = build_class_map(DEFAULT_CLASSES)
    skel_path = Path(skeleton_dir)
    label_path = Path(label_dir)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    expected_feats = session.get_inputs()[0].shape[2]
    if expected_feats == -1:
        expected_feats = 30 + (n_cameras if n_cameras > 1 else 0)

    total = 0
    correct = 0
    per_camera = {}
    confusion = np.zeros((8, 8), dtype=int)

    csv_files = sorted(skel_path.glob("*.csv"))
    for csv_path in csv_files:
        json_path = label_path / csv_path.with_suffix(".json").name
        if not json_path.exists():
            continue

        try:
            skeleton, detected = load_skeleton_csv(str(csv_path))
            with open(json_path) as fh:
                label_data = json.load(fh)
            segments = label_data["segments"]
            cam_id = int(label_data.get("camera_id", 0))
        except Exception as e:
            print(f"  SKIP {csv_path.name}: {e}")
            continue

        if cam_id not in per_camera:
            per_camera[cam_id] = {"correct": 0, "total": 0}

        T = len(skeleton)
        for seg in segments:
            gesture = seg["gesture"]
            if gesture not in class_map:
                continue
            label = class_map[gesture]
            s = seg["start_frame"]
            e = min(seg["end_frame"], T - 1)

            pos = s
            while pos + window_size <= e + 1:
                window = skeleton[pos:pos + window_size]
                det_ratio = detected[pos:pos + window_size].mean()
                if det_ratio < 0.7:
                    pos += stride
                    continue

                if n_cameras > 1:
                    one_hot = np.zeros(n_cameras, dtype=np.float32)
                    one_hot[cam_id] = 1.0
                    flag = np.tile(one_hot, (window_size, 1))
                    window = np.concatenate([window, flag], axis=-1)

                inp = window.astype(np.float32).reshape(1, window_size, -1)
                out = session.run(None, {input_name: inp})[0]
                pred = int(out.argmax())

                total += 1
                per_camera[cam_id]["total"] += 1
                if pred == label:
                    correct += 1
                    per_camera[cam_id]["correct"] += 1
                confusion[label, pred] += 1
                pos += stride

    acc = 100.0 * correct / total if total else 0.0
    print(f"\nTotal windows: {total}  Correct: {correct}  Accuracy: {acc:.2f}%\n")

    for cid in sorted(per_camera):
        c = per_camera[cid]
        ca = 100.0 * c["correct"] / c["total"] if c["total"] else 0.0
        print(f"  Camera {cid}: {c['correct']}/{c['total']} = {ca:.2f}%")
    print()

    print(f"{'':>15}", end="")
    for c in DEFAULT_CLASSES:
        print(f"{c:>15}", end="")
    print()
    for i in range(8):
        print(f"{DEFAULT_CLASSES[i]:>15}", end="")
        for j in range(8):
            print(f"{confusion[i,j]:>15}", end="")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test ONNX model accuracy")
    p.add_argument("--onnx_path", default=r"..\models_final\fern_v2.onnx")
    p.add_argument("--skeleton_dir", default=r"..\data\skeletons\front")
    p.add_argument("--label_dir", default=r"..\data\labels\front")
    p.add_argument("--n_cameras", type=int, default=1)
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=15)
    args = p.parse_args()

    test(
        onnx_path=args.onnx_path,
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        n_cameras=args.n_cameras,
        window_size=args.window_size,
        stride=args.stride,
    )
