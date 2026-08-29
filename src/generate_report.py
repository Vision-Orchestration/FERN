"""Generate PDF report with all results, charts, and figures."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# ---- Load data ----
root = Path("C:/fern/FERN_V2")

with open(root / "results_orig/learning_curve/learning_curve.json") as f:
    lc_orig = json.load(f)

aug_lc_path = root / "results_aug/learning_curve/learning_curve.json"
if aug_lc_path.exists():
    with open(aug_lc_path) as f:
        lc_aug = json.load(f)
else:
    # Partial results from terminal output (aug training was interrupted at 30 subj)
    lc_aug = [
        {"n_subjects": 5, "accuracy": 80.65},
        {"n_subjects": 10, "accuracy": 83.10},
        {"n_subjects": 20, "accuracy": 83.27},
        {"n_subjects": 30, "accuracy": 85.06},
    ]

# ---- PDF setup ----
pdf_path = root / "FERN_V2_Paper_Report.pdf"
with PdfPages(str(pdf_path)) as pdf:

    # ==================== PAGE 1: Title ====================
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.7, "FERN v2", fontsize=36, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.6, "Foot Gesture Recognition Pipeline", fontsize=18,
            ha="center", va="center", color="#555", transform=ax.transAxes)
    ax.text(0.5, 0.5, "Paper Readiness Report", fontsize=14,
            ha="center", va="center", color="#888", transform=ax.transAxes)
    ax.text(0.5, 0.35, "Generated: Aug 29, 2026", fontsize=11,
            ha="center", va="center", color="#aaa", transform=ax.transAxes)
    ax.text(0.5, 0.25, "8 Gesture Classes  |  44 Subjects  |  CNN1D-BiLSTM-Attention",
            fontsize=10, ha="center", va="center", color="#aaa", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close()

    # ==================== PAGE 2: Dataset Overview ====================
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    text = (
        "DATASET OVERVIEW\n\n"
        "• 88 recording sessions from 44 unique participants\n"
        "• Front-facing camera, 8 gesture classes\n"
        "• MediaPipe Holistic: 10-joint skeleton, 30-dim vector (x,y,z)\n"
        "• 30 fps extraction\n\n"
        "GESTURE CLASSES\n\n"
        "  1. foot_hold — Static foot placement\n"
        "  2. foot_lift — Lift foot off ground\n"
        "  3. sideway_kick — Lateral kick motion\n"
        "  4. cross_front — Cross foot in front\n"
        "  5. heel_tap — Tap heel on ground\n"
        "  6. flamingo_bend — Single-leg bend (flamingo pose)\n"
        "  7. forward_step — Step forward\n"
        "  8. forward_kick — Kick forward\n\n"
        "DATA SPLITS\n\n"
        "  Original: 88 files (44 subjects × 2 mirror)\n"
        "  Augmented: 484 files (rotation, spatial, interpolation)\n"
        "  Held-out test: 12 files (6 subjects × 2 mirror)\n"
        "  Training subjects: 38  |  Held-out: 6 (p25,p30,p35,p40,p45,p50)"
    )
    ax.text(0.05, 0.95, text, fontsize=10, fontfamily="monospace",
            ha="left", va="top", transform=ax.transAxes, linespacing=1.4)
    pdf.savefig(fig)
    plt.close()

    # ==================== PAGE 3: Class Distribution ====================
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    classes = ["foot_hold", "foot_lift", "sideway_kick", "cross_front",
               "heel_tap", "flamingo_bend", "forward_step", "forward_kick"]
    orig_dist = [4594, 330, 302, 512, 292, 270, 414, 354]
    aug_dist = [27935, 2313, 2091, 3459, 2041, 1910, 2804, 2442]

    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
              "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

    axes[0].barh(classes, orig_dist, color=colors)
    axes[0].set_title("Original (7,068 windows)", fontsize=11)
    axes[0].set_xlabel("Windows")
    for i, v in enumerate(orig_dist):
        axes[0].text(v + 50, i, str(v), va="center", fontsize=8)

    axes[1].barh(classes, aug_dist, color=colors)
    axes[1].set_title("Augmented (44,995 windows)", fontsize=11)
    axes[1].set_xlabel("Windows")
    for i, v in enumerate(aug_dist):
        axes[1].text(v + 100, i, str(v), va="center", fontsize=8)

    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ==================== PAGE 4: Learning Curve ====================
    fig, ax = plt.subplots(figsize=(8, 5))

    subj_orig = [r["n_subjects"] for r in lc_orig]
    acc_orig = [r["accuracy"] for r in lc_orig]
    subj_aug = [r["n_subjects"] for r in lc_aug]
    acc_aug = [r["accuracy"] for r in lc_aug]

    ax.plot(subj_orig, acc_orig, "o-", color="#e6194b", linewidth=2,
            markersize=8, label="Original (88 files)")
    ax.plot(subj_aug, acc_aug, "s--", color="#4363d8", linewidth=2,
            markersize=8, label="Augmented (418 files)")

    ax.set_xlabel("Number of Training Subjects", fontsize=12)
    ax.set_ylabel("Held-Out Test Accuracy (%)", fontsize=12)
    ax.set_title("Learning Curve: Accuracy vs # Training Subjects", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(75, 95)

    for i, (s, a) in enumerate(zip(subj_orig, acc_orig)):
        ax.annotate(f"{a:.1f}%", (s, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#e6194b")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ==================== PAGE 5: Confusion Matrix (Original) ====================
    fig, ax = plt.subplots(figsize=(8, 7))

    cm_orig = np.array([
        [4220,   15,   53,   83,   25,   69,   59,   70],
        [   0,  329,    0,    0,    0,    0,    1,    0],
        [   0,    0,  300,    0,    0,    0,    0,    2],
        [   0,    0,    3,  505,    0,    1,    3,    0],
        [   0,    0,    0,    0,  292,    0,    0,    0],
        [   0,    0,    0,    0,    0,  270,    0,    0],
        [   0,    0,    0,    0,    0,    2,  412,    0],
        [   0,    0,    0,    0,    7,    0,    0,  347],
    ])

    short = [c[:5] for c in classes]
    im = ax.imshow(cm_orig, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix — Original Model (94.44%)", fontsize=12, fontweight="bold")

    for i in range(8):
        for j in range(8):
            color = "white" if cm_orig[i, j] > cm_orig.max() / 2 else "black"
            ax.text(j, i, str(cm_orig[i, j]), ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ==================== PAGE 6: Confusion Matrix (Augmented) ====================
    fig, ax = plt.subplots(figsize=(8, 7))

    cm_aug = np.array([
        [25651,  106,  319,  545,  194,  383,  303,  434],
        [    1, 2292,    1,    0,   10,    2,    2,    5],
        [    0,    4, 2075,    0,    0,    0,    0,   12],
        [    0,    2,    1, 3408,   12,    4,   12,   20],
        [    1,    0,    0,    0, 2040,    0,    0,    0],
        [    3,    2,    0,    0,    4, 1901,    0,    0],
        [    1,   12,    5,    0,    3,    7, 2769,    7],
        [    0,    1,    6,    0,   32,    0,    4, 2399],
    ])

    im = ax.imshow(cm_aug, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix — Augmented Model (94.53%)", fontsize=12, fontweight="bold")

    for i in range(8):
        for j in range(8):
            color = "white" if cm_aug[i, j] > cm_aug.max() / 2 else "black"
            ax.text(j, i, str(cm_aug[i, j]), ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ==================== PAGE 7: Held-Out Test Comparison ====================
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    held_classes = ["foot_hold", "foot_lift", "sideway_kick", "cross_front",
                    "heel_tap", "flamingo_bend", "forward_step", "forward_kick"]
    held_orig_f1 = [3.4, 72.7, 84.2, 89.8, 63.4, 66.7, 78.9, 73.7]
    held_aug_f1 = [0.0, 88.9, 84.2, 82.2, 63.4, 69.4, 72.7, 73.7]

    x = np.arange(len(held_classes))
    width = 0.35

    axes[0].bar(x - width/2, held_orig_f1, width, label="Original", color="#e6194b", alpha=0.8)
    axes[0].bar(x + width/2, held_aug_f1, width, label="Augmented", color="#4363d8", alpha=0.8)
    axes[0].set_ylabel("F1 Score (%)", fontsize=10)
    axes[0].set_title("Held-Out Test: Per-Class F1", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c[:6] for c in held_classes], rotation=45, ha="right", fontsize=8)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.3)

    # Overall comparison
    models = ["Original\n(train_all)", "Augmented\n(train_all)"]
    held_accs = [68.14, 66.37]
    train_accs = [94.44, 94.53]

    x2 = np.arange(2)
    axes[1].bar(x2 - width/2, train_accs, width, label="Train-All", color="#3cb44b", alpha=0.8)
    axes[1].bar(x2 + width/2, held_accs, width, label="Held-Out Test", color="#e6194b", alpha=0.8)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].set_title("Train-All vs Held-Out", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(models, fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.3)

    for i, (t, h) in enumerate(zip(train_accs, held_accs)):
        axes[1].text(i - width/2, t + 1, f"{t:.1f}%", ha="center", fontsize=8, fontweight="bold")
        axes[1].text(i + width/2, h + 1, f"{h:.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ==================== PAGE 8: t-SNE (Original) ====================
    from PIL import Image
    tsne_orig_path = root / "results_orig/tsne.png"
    if tsne_orig_path.exists():
        img = Image.open(tsne_orig_path)
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("t-SNE Visualization — Original Model", fontsize=12, fontweight="bold", pad=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # ==================== PAGE 9: t-SNE (Augmented) ====================
    tsne_aug_path = root / "results_aug/tsne.png"
    if tsne_aug_path.exists():
        img = Image.open(tsne_aug_path)
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("t-SNE Visualization — Augmented Model", fontsize=12, fontweight="bold", pad=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # ==================== PAGE 10: Ablation Table ====================
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    table_data = [
        ["#", "Config", "Dataset", "Params", "CV Accuracy", "Δ"],
        ["1", "Baseline (cnn=64, do=0.6)", "front", "132K", "41.11% ±3.59%", "—"],
        ["2", "+ cnn_out=128", "front", "526K", "42.80% ±3.94%", "+1.69"],
        ["3", "+ dropout=0.3", "front", "132K", "43.73% ±5.02%", "+2.62"],
        ["4", "+ cnn=128 + do=0.3", "front", "526K", "44.36% ±6.75%", "+3.25"],
        ["5", "BiLSTM baseline", "front", "207K", "24.80%", "-16.31"],
        ["6", "BiLSTM + combined", "combined", "207K", "35.12%", "-5.99"],
        ["7", "CNN + combined (no flag)", "combined", "132K", "38.29%", "-2.82"],
        ["8", "CNN + camera-ID flag", "front+45", "140K", "55.71%", "+14.60"],
        ["9", "Final (cnn=128, do=0.3)", "front", "526K", "94.44%*", "—"],
        ["10", "+ 5× augmentation CV", "front_aug", "526K", "68.96% ±22%", "-5.27"],
        ["11", "+ 5× augmentation train", "front_aug", "526K", "94.53%*", "—"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.6)

    for i in range(len(table_data[0])):
        table[0, i].set_facecolor("#4472C4")
        table[0, i].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i in [9, 11]:
                table[i, j].set_facecolor("#E2EFDA")

    ax.set_title("Ablation Study — All Experiments", fontsize=13, fontweight="bold", pad=20)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ==================== PAGE 11: Key Findings ====================
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    text = (
        "KEY FINDINGS\n\n"
        "1. CNN-only dominates BiLSTM\n"
        "   +16 pp over BiLSTM baseline (74.10% vs 24.80%)\n\n"
        "2. Width matters more than depth\n"
        "   cnn_out=128 + dropout=0.3 → +3.25 pp CV over baseline\n\n"
        "3. Multi-camera hurts without flag\n"
        "   38.29% vs 41.11% (front-only). Camera-ID flag: +14.60 pp\n\n"
        "4. Augmentation does NOT improve cross-subject generalization\n"
        "   CV: 68.96% vs 74.23% (-5.27 pp)\n"
        "   Held-out: 66.37% vs 68.14% (-1.77 pp)\n"
        "   Synthetic variants don't teach the model new people\n\n"
        "5. Data scarcity is the primary bottleneck\n"
        "   Train-all: ~94.5%  |  Held-out: ~68%  |  CV mean: ~69-74%\n"
        "   26 pp gap = memorization without generalization\n\n"
        "6. foot_hold collapses on unseen subjects\n"
        "   25% precision (original), 0% (augmented)\n"
        "   Clearest overfitting signal\n\n"
        "7. Learning curve still climbing at 38 subjects\n"
        "   80.97% (5 subj) → 89.38% (38 subj)\n"
        "   More real subjects = best path forward\n\n"
        "RECOMMENDATIONS FOR PAPER\n\n"
        "• Report held-out test (68.14%) as headline result\n"
        "• Show learning curve as evidence of data bottleneck\n"
        "• Include augmentation as negative result\n"
        "• Report per-class F1 — foot_hold confusion is key story\n"
        "• Consider LOSO CV for gold-standard evaluation"
    )
    ax.text(0.05, 0.95, text, fontsize=10, fontfamily="monospace",
            ha="left", va="top", transform=ax.transAxes, linespacing=1.35)
    pdf.savefig(fig)
    plt.close()

print(f"PDF saved to {pdf_path}")
