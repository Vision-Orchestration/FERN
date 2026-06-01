"""
FERN v2 — Step 4: Training loop.

Features
--------
- Mixed precision (AMP) with GradScaler for RTX 3070 efficiency.
- Cosine LR schedule with linear warmup.
- Gradient clipping (max_norm = 1.0).
- EarlyStopping on validation loss.
- Checkpoint saving (best model + latest epoch).
- TensorBoard logging.
- Two-phase training:
    Phase 1 (warmup epochs): lower LR, no augmentation.
    Phase 2 (remaining):     full LR, augmentation enabled.

Usage
-----
python src/train_v2.py \
    --skeleton_dir  data/skeletons \
    --label_dir     data/labels \
    --output_dir    models \
    --log_dir       logs \
    --epochs        100 \
    --warmup_epochs 10 \
    --batch_size    32 \
    --window_size   60 \
    --stride        15 \
    --lr            3e-4 \
    --device        cuda \
    --seed          42
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES
from model_v2   import FERNv2


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Learning rate scheduler that linearly ramps up over `warmup_steps`
    then follows a cosine decay down to `min_lr`.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Training / validation pass
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler,
              device, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast():
                logits = model(x)
                loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            preds          = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += len(y)
            total_loss    += loss.item() * len(y)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)

    # --- Device ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {vram:.1f} GB")
    print()

    # --- Datasets ---
    train_ds = SkeletonWindowDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        classes=DEFAULT_CLASSES,
        window_size=args.window_size,
        stride=args.stride,
        split="train",
        augment=False,   # augmentation toggled per-epoch in Phase 2
    )
    val_ds = SkeletonWindowDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        classes=DEFAULT_CLASSES,
        window_size=args.window_size,
        stride=args.stride,
        split="val",
        augment=False,
    )

    train_ds.class_distribution()

    train_sampler = train_ds.get_sampler()
    train_loader  = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = FERNv2(
        num_joints=10,
        num_classes=len(DEFAULT_CLASSES),
        cnn_out=args.cnn_out,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Loss, optimiser, scaler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler    = GradScaler()
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
    )
    stopper   = EarlyStopping(patience=args.patience)

    # --- Logging ---
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir)

    best_val_acc  = 0.0
    best_ckpt     = os.path.join(args.output_dir, "fern_v2_best.pth")
    latest_ckpt   = os.path.join(args.output_dir, "fern_v2_latest.pth")

    print(f"\nTraining for up to {args.epochs} epochs ...\n")
    header = f"{'Epoch':>6} {'LR':>8} {'T-Loss':>8} {'T-Acc':>7} {'V-Loss':>8} {'V-Acc':>7} {'Time':>7}"
    print(header)
    print("-" * len(header))

    for epoch in range(args.epochs):
        t0 = time.time()

        # Enable augmentation after warmup.
        train_ds.augment = (epoch >= args.warmup_epochs)

        lr = scheduler.step(epoch)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, False
        )

        elapsed = time.time() - t0

        print(f"{epoch+1:>6} {lr:>8.2e} {train_loss:>8.4f} "
              f"{train_acc*100:>6.1f}% {val_loss:>8.4f} "
              f"{val_acc*100:>6.1f}% {elapsed:>6.1f}s")

        # TensorBoard
        writer.add_scalar("Loss/train",    train_loss, epoch)
        writer.add_scalar("Loss/val",      val_loss,   epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val",   val_acc,   epoch)
        writer.add_scalar("LR",             lr,        epoch)

        # Save checkpoints
        ckpt = {
            "epoch":       epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_acc":     val_acc,
            "args":        vars(args),
        }
        torch.save(ckpt, latest_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt)
            print(f"  -> New best val acc: {best_val_acc*100:.2f}%  "
                  f"(saved to {best_ckpt})")

        stopper.step(val_loss)
        if stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs).")
            break

    writer.close()
    print(f"\nTraining complete.  Best val acc: {best_val_acc*100:.2f}%")
    print(f"Best checkpoint: {best_ckpt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train FERN v2")
    p.add_argument("--skeleton_dir",  default="data/skeletons")
    p.add_argument("--label_dir",     default="data/labels")
    p.add_argument("--output_dir",    default="models")
    p.add_argument("--log_dir",       default="logs")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--window_size",   type=int,   default=60)
    p.add_argument("--stride",        type=int,   default=15)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--cnn_out",       type=int,   default=64)
    p.add_argument("--lstm_hidden",   type=int,   default=128)
    p.add_argument("--lstm_layers",   type=int,   default=2)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
