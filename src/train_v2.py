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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


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
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_metric = 0.0
        self.counter     = 0
        self.should_stop = False

    def step(self, val_acc: float):
        if val_acc > self.best_metric + self.min_delta:
            self.best_metric = val_acc
            self.counter     = 0
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

            with autocast("cuda", enabled=device.type == "cuda"):
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

    # --- CUDA optimisations (RTX 30xx Tensor Cores) ---
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32      = True
        torch.backends.cudnn.allow_tf32             = True
        torch.set_float32_matmul_precision("high")

    # --- Device ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {vram:.1f} GB")
    print()

    # --- Input feature dimension ---
    base_features = 30  # 10 joints × 3 (x, y, z)
    input_features = base_features + (args.n_cameras if args.n_cameras > 1 else 0)
    dataset_kwargs = dict(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        classes=DEFAULT_CLASSES,
        window_size=args.window_size,
        stride=args.stride,
        split_mode=args.split_mode,
        n_cameras=args.n_cameras,
    )

    # --- Datasets ---
    if args.train_all:
        full_ds = SkeletonWindowDataset(
            **dataset_kwargs, split="all", augment=True
        )
        full_ds.class_distribution()
        full_sampler = full_ds.get_sampler()
        train_loader = DataLoader(
            full_ds,
            batch_size=args.batch_size,
            sampler=full_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = None
    else:
        train_ds = SkeletonWindowDataset(
            **dataset_kwargs, split="train", augment=True
        )
        val_ds = SkeletonWindowDataset(
            **dataset_kwargs, split="val", augment=False
        )
        train_ds.class_distribution()
        train_sampler = train_ds.get_sampler()
        train_loader = DataLoader(
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

    # --- Model (optionally resume from checkpoint) ---
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt["epoch"]
        args.epochs = max(args.epochs, start_epoch)  # ensure we don't stop early

    model = FERNv2(
        num_joints=10,
        num_classes=len(DEFAULT_CLASSES),
        cnn_out=args.cnn_out,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        input_features=input_features,
    ).to(device)

    if args.resume:
        model.load_state_dict(ckpt["model_state"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Loss, optimiser, scaler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
    )
    stopper   = EarlyStopping(patience=args.patience)

    if args.resume:
        optimizer.load_state_dict(ckpt["optim_state"])
        scaler.load_state_dict(ckpt.get("scaler_state", scaler.state_dict()))
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        stopper.best_metric = best_val_acc

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

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        lr = scheduler.step(epoch)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, True
        )

        if val_loader is not None:
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device, False
            )
        else:
            val_loss, val_acc = 0.0, 0.0

        elapsed = time.time() - t0

        if val_loader is not None:
            print(f"{epoch+1:>6} {lr:>8.2e} {train_loss:>8.4f} "
                  f"{train_acc*100:>6.1f}% {val_loss:>8.4f} "
                  f"{val_acc*100:>6.1f}% {elapsed:>6.1f}s")
        else:
            print(f"{epoch+1:>6} {lr:>8.2e} {train_loss:>8.4f} "
                  f"{train_acc*100:>6.1f}% {elapsed:>6.1f}s")

        # TensorBoard
        writer.add_scalar("Loss/train",    train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("LR",             lr,        epoch)

        # Save checkpoints
        ckpt = {
            "epoch":        epoch + 1,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_acc":      0.0,
            "best_val_acc": best_val_acc,
            "args":         vars(args),
        }
        torch.save(ckpt, latest_ckpt)

        if args.train_all:
            # Just save latest — no validation to track
            continue

        writer.add_scalar("Loss/val",    val_loss,   epoch)
        writer.add_scalar("Accuracy/val", val_acc,   epoch)
        ckpt["val_acc"] = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt)
            print(f"  -> New best val acc: {best_val_acc*100:.2f}%  "
                  f"(saved to {best_ckpt})")

        if epoch >= args.warmup_epochs:
            stopper.step(val_acc)
        if stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs).")
            break

    writer.close()
    if args.train_all:
        print(f"\nTraining complete.  Final model saved to {latest_ckpt}")
    else:
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
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--cnn_out",       type=int,   default=32)
    p.add_argument("--lstm_hidden",   type=int,   default=64)
    p.add_argument("--lstm_layers",   type=int,   default=2)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--split_mode",    default="random", choices=["random", "subject"])
    p.add_argument("--train_all",     action="store_true",
                   help="Train on ALL data (no validation split)")
    p.add_argument("--n_cameras",     type=int,   default=1,
                   help="1=no flag, 2=c3+c2, 3=c3+c4+c2")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to .pth checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
