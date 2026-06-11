"""
test_transform.py
Evaluate the existing ONNX model on three conditions:
  A) Front camera — baseline
  B) Raw 45° skeleton — no transform
  C) Transformed 45° skeleton — after R_y(-45°)

Prints accuracy per condition and per class, then emits a Phase 1 verdict.

Usage:
    python src/test_transform.py
    python src/test_transform.py --angle 45.0 --model models_final/fern_v2.onnx
"""
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path

from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES


def evaluate(model_path: str, skeleton_dir: str, label_dir: str,
             window_size: int = 60, stride: int = 15) -> dict:
    """Run ONNX model on a full dataset directory. Returns accuracy dict."""
    if not Path(skeleton_dir).exists():
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess       = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    ds = SkeletonWindowDataset(
        skeleton_dir=skeleton_dir,
        label_dir=label_dir,
        window_size=window_size,
        stride=stride,
        split="all",
        augment=False,
    )
    if len(ds) == 0:
        print(f"  WARNING: 0 windows in {skeleton_dir}")
        return None

    nc = len(DEFAULT_CLASSES)
    correct_per_class = np.zeros(nc, dtype=int)
    total_per_class   = np.zeros(nc, dtype=int)

    for x, y in ds:
        inp    = x.numpy().astype(np.float32)[np.newaxis]
        logits = sess.run(None, {input_name: inp})[0]
        pred   = int(np.argmax(logits))
        total_per_class[y]   += 1
        if pred == y:
            correct_per_class[y] += 1

    total   = int(total_per_class.sum())
    correct = int(correct_per_class.sum())
    return {
        "acc":     correct / total,
        "total":   total,
        "correct": correct,
        "per_class_acc":   correct_per_class / np.maximum(total_per_class, 1),
        "per_class_total": total_per_class,
    }


def print_result(label: str, r: dict):
    if r is None:
        print(f"  {label}: SKIPPED (directory not found)")
        return
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Overall accuracy: {r['acc']*100:.2f}%  "
          f"({r['correct']}/{r['total']} windows)")
    print(f"  {'Class':<22} {'Acc':>8} {'Windows':>9}")
    for i, cls in enumerate(DEFAULT_CLASSES):
        print(f"  {cls:<22} {r['per_class_acc'][i]*100:>7.1f}% "
              f"{r['per_class_total'][i]:>9}")


def main(args):
    print(f"\nModel: {args.model}")

    results = {
        "A_front": evaluate(
            args.model,
            args.front_skel, args.front_label,
            args.window_size, args.stride,
        ),
        "B_raw45": evaluate(
            args.model,
            args.raw45_skel, args.raw45_label,
            args.window_size, args.stride,
        ),
        "C_tf45": evaluate(
            args.model,
            args.tf45_skel, args.tf45_label,
            args.window_size, args.stride,
        ),
    }

    print_result("A — Front camera (baseline)", results["A_front"])
    print_result("B — Raw 45deg (no transform)",  results["B_raw45"])
    print_result("C — Transformed 45deg",          results["C_tf45"])

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  {'Condition':<35} {'Accuracy':>10}")
    for key, label in [
        ("A_front", "A — Front (baseline)"),
        ("B_raw45", "B — Raw 45deg, no transform"),
        ("C_tf45",  "C — Transformed 45deg"),
    ]:
        r = results[key]
        if r:
            print(f"  {label:<35} {r['acc']*100:>9.2f}%")

    r_b = results["B_raw45"]
    r_c = results["C_tf45"]
    print(f"\n{'='*60}")
    print("  PHASE 1 VERDICT")
    if r_b and r_c:
        gain = (r_c["acc"] - r_b["acc"]) * 100
        print(f"  Transform gain over raw 45deg: {gain:+.2f}%")
        if r_c["acc"] >= 0.65:
            print("  STRONG PASS (>=65%) — Phase 2 strongly recommended.")
            print("     MediaPipe z has enough signal. Stereo will improve further.")
        elif r_c["acc"] >= 0.50:
            print("  PASS (>=50%) — Proceed to Phase 2.")
            print("     Transform works. Real z from stereo will close the gap.")
        elif r_c["acc"] >= 0.35:
            print("  MARGINAL (35-50%) — MediaPipe z is noisy.")
            print("     Phase 2 still worth attempting. Stereo z may fix it.")
        else:
            print("  FAIL (<35%) — Transform did not help meaningfully.")
            print("     Do NOT proceed to Phase 2 without investigating.")
            print("     Check: is the camera angle correct? Is z truly non-zero?")
    else:
        print("  Could not compare — one or more conditions missing.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="models_final/fern_v2.onnx")
    p.add_argument("--front_skel",   default="data/skeletons/front")
    p.add_argument("--front_label",  default="data/labels/front")
    p.add_argument("--raw45_skel",   default="data/skeletons/raw_45")
    p.add_argument("--raw45_label",  default="data/labels/raw_45")
    p.add_argument("--tf45_skel",    default="data/skeletons/transformed_45")
    p.add_argument("--tf45_label",   default="data/labels/transformed_45")
    p.add_argument("--window_size",  type=int, default=60)
    p.add_argument("--stride",       type=int, default=15)
    p.add_argument("--angle",        type=float, default=45.0)
    main(p.parse_args())
