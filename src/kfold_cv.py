"""
K-fold CV with leak-free fold grouping.

--group_by window  : legacy behavior (LEAKS — adjacent windows share 75% frames)
--group_by video   : folds split at video level (mirror stays with original)
--group_by subject : folds split at subject level (recommended unbiased estimate)
"""

import argparse
import re
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES
from model_v2 import FERNv2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += len(y)
            total_loss += loss.item() * len(y)
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def build_groups(dataset, group_by):
    """Return one group key per sample index."""
    if group_by == "window":
        return [str(i) for i in range(len(dataset))]
    keys = []
    for src in dataset.sample_sources:
        stem = Path(src).stem.replace("_mirror", "")
        if group_by == "video":
            keys.append(stem)
        else:  # subject
            m = re.match(r"^(p\d{2})_c\d", stem)
            keys.append(m.group(1) if m else stem)
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton_dir", default="data/skeletons/front_plus_45")
    p.add_argument("--label_dir", default="data/labels/front_plus_45")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--cnn_out", type=int, default=64)
    p.add_argument("--lstm_hidden", type=int, default=0)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_cameras", type=int, default=1)
    p.add_argument("--group_by", default="subject",
                   choices=["window", "video", "subject"])
    p.add_argument("--label_smoothing", type=float, default=0.1)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  group_by={args.group_by}  n_cameras={args.n_cameras}")

    input_features = 30 + (args.n_cameras if args.n_cameras > 1 else 0)

    common = dict(
        skeleton_dir=args.skeleton_dir, label_dir=args.label_dir,
        classes=DEFAULT_CLASSES, window_size=args.window_size,
        stride=args.stride, split="all", n_cameras=args.n_cameras,
    )
    train_ds = SkeletonWindowDataset(**common, augment=True)
    val_ds = SkeletonWindowDataset(**common, augment=False)

    group_keys = build_groups(train_ds, args.group_by)
    unique_groups = sorted(set(group_keys))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(unique_groups)
    fold_of_group = {g: i % args.k_folds for i, g in enumerate(unique_groups)}
    print(f"Total windows: {len(train_ds)}  Groups: {len(unique_groups)} ({args.group_by})")

    fold_accs = []
    for fold in range(args.k_folds):
        train_idx = [i for i, g in enumerate(group_keys) if fold_of_group[g] != fold]
        val_idx = [i for i, g in enumerate(group_keys) if fold_of_group[g] == fold]
        print(f"\n{'='*50}\nFold {fold+1}/{args.k_folds}  "
              f"Train: {len(train_idx)}  Val: {len(val_idx)}")
        if not val_idx:
            print("  EMPTY fold — too few groups for k. Skipping.")
            continue

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(val_idx),
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )

        model = FERNv2(
            num_joints=10, num_classes=len(DEFAULT_CLASSES),
            cnn_out=args.cnn_out, lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers, dropout=args.dropout,
            input_features=input_features,
        ).to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=(device.type == "cuda"))
        patience = 30
        best_val_acc = 0.0
        no_improve = 0

        for epoch in range(args.epochs):
            lr = args.lr if epoch >= args.warmup_epochs else args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, scaler, device, True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device, False
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == args.epochs - 1 or no_improve == patience:
                print(f"  E{epoch+1:>3}  T:{train_loss:.3f}/{train_acc*100:.1f}%  "
                      f"V:{val_loss:.3f}/{val_acc*100:.1f}%  Best:{best_val_acc*100:.1f}%")

            if no_improve >= patience and epoch >= args.warmup_epochs:
                break

        fold_accs.append(best_val_acc)
        print(f"  >> Fold {fold+1} best: {best_val_acc*100:.2f}%")

    print(f"\n{'='*50}")
    print(f"{args.k_folds}-fold CV (group_by={args.group_by}):")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: {acc*100:.2f}%")
    print(f"  Mean: {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
