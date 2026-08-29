"""Learning curve: train on N subjects, evaluate on held-out. Proves data scarcity bottleneck."""

import re, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from pathlib import Path

from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES
from model_v2 import FERNv2


def get_subject_groups(dataset):
    groups = {}
    for i, src in enumerate(dataset.sample_sources):
        stem = Path(src).stem.replace("_mirror", "")
        m = re.match(r"^(p\d{2})_c\d", stem)
        key = m.group(1) if m else stem
        groups.setdefault(key, []).append(i)
    return groups


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_correct = total_samples = total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += len(y)
        total_loss += loss.item() * len(y)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = total_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += len(y)
    return total_correct / total_samples


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load held-out test subjects
    held_out = ['p25', 'p30', 'p35', 'p40', 'p45', 'p50']

    # Training dataset (front, all subjects)
    train_ds = SkeletonWindowDataset(
        skeleton_dir=args.skeleton_dir, label_dir=args.label_dir,
        classes=DEFAULT_CLASSES, window_size=args.window_size,
        stride=args.stride, split="all", augment=True,
    )
    train_groups = get_subject_groups(train_ds)
    all_subjects = sorted(train_groups.keys())
    train_subjects = [s for s in all_subjects if s not in held_out]
    print(f"Training subjects: {len(train_subjects)}")

    # Held-out test dataset
    test_ds = SkeletonWindowDataset(
        skeleton_dir=args.test_skeleton_dir, label_dir=args.test_label_dir,
        classes=DEFAULT_CLASSES, window_size=args.window_size,
        stride=args.stride, split="all", augment=False,
    )
    test_groups = get_subject_groups(test_ds)
    test_idx = []
    for indices in test_groups.values():
        test_idx.extend(indices)
    test_loader = DataLoader(test_ds, batch_size=64, sampler=SubsetRandomSampler(test_idx), num_workers=0)
    print(f"Test windows: {len(test_idx)}")

    # Learning curve
    subject_counts = [5, 10, 15, 20, 25, 30, 38]
    results = []

    for n_subj in subject_counts:
        if n_subj > len(train_subjects):
            break
        rng = np.random.RandomState(42)
        selected = sorted(rng.choice(train_subjects, size=n_subj, replace=False).tolist())
        print(f"\n--- {n_subj} subjects: {selected} ---")

        # Get training indices for selected subjects
        train_idx = []
        for s in selected:
            train_idx.extend(train_groups[s])

        train_loader = DataLoader(train_ds, batch_size=32, sampler=SubsetRandomSampler(train_idx), num_workers=0, drop_last=True)

        model = FERNv2(num_joints=10, num_classes=len(DEFAULT_CLASSES), cnn_out=128, dropout=0.3).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

        best_acc = 0
        patience = 30
        no_improve = 0
        warmup = 15

        for epoch in range(100):
            lr = 3e-4 if epoch >= warmup else 3e-4 * (epoch + 1) / warmup
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            if (epoch + 1) % 10 == 0:
                acc = evaluate(model, test_loader, device)
                print(f"  E{epoch+1:>3}  test_acc={acc*100:.1f}%")
                if acc > best_acc:
                    best_acc = acc
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience and epoch >= warmup:
                    break

        results.append({"n_subjects": n_subj, "accuracy": best_acc * 100})
        print(f"  >> {n_subj} subjects: {best_acc*100:.2f}%")

    # Save results
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "learning_curve.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Learning Curve Results:")
    for r in results:
        print(f"  {r['n_subjects']:>3} subjects: {r['accuracy']:.2f}%")
    print(f"{'='*50}")
    print(f"Saved to {out / 'learning_curve.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton_dir", default="data/skeletons/front")
    p.add_argument("--label_dir", default="data/labels/front")
    p.add_argument("--test_skeleton_dir", default="data/skeletons/front_test")
    p.add_argument("--test_label_dir", default="data/labels/front_test")
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--output", default="results_orig/learning_curve")
    run(p.parse_args())
