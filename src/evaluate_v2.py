"""
FERN v2 — Step 5: Evaluate a trained model on the test split.

Computes per-class precision, recall, F1, top-1 accuracy, top-3 accuracy,
and prints a confusion matrix.

Usage
-----
python src/evaluate_v2.py \
    --model        models/fern_v2_best.pth \
    --skeleton_dir data/skeletons \
    --label_dir    data/labels \
    --window_size  60 \
    --stride       15
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES
from model_v2   import FERNv2


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint.
    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    model = FERNv2(
        num_joints=10,
        num_classes=len(DEFAULT_CLASSES),
        cnn_out=saved.get("cnn_out", 64),
        lstm_hidden=saved.get("lstm_hidden", 128),
        lstm_layers=saved.get("lstm_layers", 2),
        dropout=0.0,   # no dropout at eval time
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = SkeletonWindowDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        classes=DEFAULT_CLASSES,
        window_size=args.window_size,
        stride=args.stride,
        split="test",
        augment=False,
    )

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    n      = len(DEFAULT_CLASSES)

    all_preds  = []
    all_labels = []
    top3_correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

            # Top-3
            top3 = logits.topk(min(3, n), dim=1).indices.cpu().numpy()
            for i, label in enumerate(y.numpy()):
                if label in top3[i]:
                    top3_correct += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    total      = len(all_labels)

    top1_acc = (all_preds == all_labels).mean() * 100
    top3_acc = top3_correct / total * 100

    # Confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    print(f"\n{'='*55}")
    print(f"  FERN v2 Evaluation   ({total} windows)")
    print(f"{'='*55}")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print(f"  {'-'*45}")

    for i, cls in enumerate(DEFAULT_CLASSES):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / max(1, tp + fp) * 100
        rec  = tp / max(1, tp + fn) * 100
        f1   = 2 * prec * rec / max(1, prec + rec)
        cnt  = cm[i].sum()
        print(f"  {cls:<20} {prec:>5.1f}% {rec:>5.1f}% {f1:>5.1f}% {cnt:>5}")

    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "".join(f"{c[:4]:>6}" for c in DEFAULT_CLASSES)
    print(f"  {'':20}{header}")
    for i, cls in enumerate(DEFAULT_CLASSES):
        row = "".join(f"{cm[i][j]:>6}" for j in range(n))
        print(f"  {cls:<20}{row}")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--skeleton_dir", default="data/skeletons")
    p.add_argument("--label_dir",    default="data/labels")
    p.add_argument("--window_size",  type=int, default=60)
    p.add_argument("--stride",       type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
