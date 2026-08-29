"""t-SNE visualization of CNN window embeddings, colored by gesture class."""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_v2 import SkeletonWindowDataset, DEFAULT_CLASSES

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def extract_embeddings(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    from model_v2 import FERNv2
    model = FERNv2(
        num_joints=10, num_classes=len(DEFAULT_CLASSES),
        cnn_out=saved.get("cnn_out", 64),
        lstm_hidden=saved.get("lstm_hidden", 0),
        lstm_layers=saved.get("lstm_layers", 1),
        dropout=0.0,
        input_features=30,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Hook to capture CNN embeddings (before classifier head)
    embeddings = []
    labels = []

    def hook_fn(module, input, output):
        embeddings.append(output.detach().cpu().numpy())

    # Register on the global pooling layer (after CNN)
    hook = model.gap.register_forward_hook(hook_fn)

    ds = SkeletonWindowDataset(
        skeleton_dir=args.skeleton_dir, label_dir=args.label_dir,
        classes=DEFAULT_CLASSES, window_size=args.window_size,
        stride=args.stride, split="all", augment=False,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            model(x)
            labels.extend(y.numpy())

    hook.remove()

    all_emb = np.concatenate(embeddings, axis=0)
    all_labels = np.array(labels)
    print(f"Extracted {len(all_emb)} embeddings, shape={all_emb.shape}")

    return all_emb, all_labels


def plot_tsne(embeddings, labels, output_path, title="t-SNE of CNN Embeddings"):
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(DEFAULT_CLASSES):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[i], label=cls, alpha=0.6, s=10, edgecolors="none")

    ax.legend(fontsize=9, markerscale=2, framealpha=0.9)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--skeleton_dir", default="data/skeletons/front")
    p.add_argument("--label_dir", default="data/labels/front")
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--output", default="results_orig/tsne.png")
    p.add_argument("--title", default="t-SNE of CNN Embeddings (Original)")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    emb, labels = extract_embeddings(args)
    plot_tsne(emb, labels, args.output, args.title)
