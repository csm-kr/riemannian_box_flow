"""Draw actual boxes on a 224×224 canvas: GT (dataset) vs init priors.

Saves outputs/init_viz/init_boxes_canvas.png.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.trajectory import sample_init_box


CANVAS = 224


def draw_boxes(ax, boxes: np.ndarray, color: str, alpha: float = 0.4, lw: float = 0.6):
    for cx, cy, w, h in boxes:
        x0 = (cx - w / 2) * CANVAS
        y0 = (cy - h / 2) * CANVAS
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), w * CANVAS, h * CANVAS,
            fill=False, edgecolor=color, linewidth=lw, alpha=alpha,
        ))


def collect_gt(N_samples: int = 100, seed: int = 0):
    """Sample N images from dataset → each gives 10 GT boxes → flatten."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = MNISTBoxDataset(split="train", root="./data")
    boxes = []
    idx = np.random.randint(0, len(ds), size=N_samples)
    for i in idx:
        d = ds[i]
        boxes.append(d["gt_boxes"].numpy())
    return np.concatenate(boxes, axis=0)   # (N_samples * 10, 4)


def collect_init(prior: str, N: int = 1000, seed: int = 0):
    torch.manual_seed(seed)
    ref = torch.empty(N, 4)
    return sample_init_box(ref, prior=prior).numpy()   # (N, 4)


def plot_overlays(gt, init_def, init_sm, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("GT (dataset, 100 images × 10 = 1000)", gt, "tab:green"),
        ("init default (1000 boxes)", init_def, "tab:blue"),
        ("init small_size (1000 boxes)", init_sm, "tab:orange"),
    ]
    for ax, (title, boxes, color) in zip(axes, panels):
        ax.set_xlim(0, CANVAS)
        ax.set_ylim(CANVAS, 0)
        ax.set_aspect("equal")
        ax.set_facecolor("#f6f6f6")
        ax.set_title(title, fontsize=11)
        ax.add_patch(mpatches.Rectangle(
            (0, 0), CANVAS, CANVAS, fill=False, edgecolor="black", linewidth=1.2,
        ))
        draw_boxes(ax, boxes, color=color, alpha=0.3, lw=0.4)
        wh = boxes[:, 2:].mean(axis=0) * CANVAS
        ax.text(
            0.02, 0.98,
            f"N={len(boxes)}\n"
            f"⟨w⟩={wh[0]:.1f}px  ⟨h⟩={wh[1]:.1f}px\n"
            f"size range: [{boxes[:,2:].min():.3f}, {boxes[:,2:].max():.3f}]",
            transform=ax.transAxes, fontsize=8, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    fig.suptitle("Boxes drawn on a 224×224 canvas", fontsize=13)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")


def plot_size_compare(gt, init_def, init_sm, save_path: Path):
    """w-h scatter to see where init prior falls relative to GT."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, boxes, color, marker, alpha in [
        ("GT",          gt,       "tab:green",  "o", 0.35),
        ("default",     init_def, "tab:blue",   "x", 0.35),
        ("small_size",  init_sm,  "tab:orange", "+", 0.5),
    ]:
        ax.scatter(boxes[:, 2], boxes[:, 3], s=12, color=color,
                   marker=marker, alpha=alpha, label=f"{name} (n={len(boxes)})")
    ax.set_xlabel("w  (normalized)")
    ax.set_ylabel("h  (normalized)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Box size: GT vs init priors", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    out_dir = Path("outputs/init_viz")

    gt = collect_gt(N_samples=100)
    init_def = collect_init("default", N=1000)
    init_sm = collect_init("small_size", N=1000)

    plot_overlays(gt, init_def, init_sm, out_dir / "init_boxes_canvas.png")
    plot_size_compare(gt, init_def, init_sm, out_dir / "init_size_vs_gt.png")
