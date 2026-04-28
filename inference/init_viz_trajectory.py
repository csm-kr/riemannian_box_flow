"""Box-space distribution of intermediate b_t per space.

For each of 4 spaces (S-E / S-R / C-R / Logit), compute b_t along the
training trajectory:
  b_t = decode( (1−t) encode(b_0) + t encode(b_1) )

Same (b_0, b_1) pairs across spaces; only the encode/decode map changes.
b_0 ~ default prior, b_1 ~ GT (dataset).

Saves outputs/init_viz/box_trajectory_t05.png  — t = 0.5 snapshot
       outputs/init_viz/box_trajectory_sweep.png — single pair, 11 t-steps
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.trajectory import (
    chart_decode, chart_encode, logit_decode, logit_encode,
    sample_init_box, signal_decode, signal_encode,
)


CANVAS = 224


# -- 4 trajectories share (b_0, b_1); only encode/decode differs --

def traj_signal(b0, b1, t):
    """S-E: linear in box (signal_decode(linear in signal) ⇔ linear in box)."""
    return (1 - t) * b0 + t * b1


def traj_chart(b0, b1, t):
    """psi-trajectory (S-R = C-R): pos linear, size multiplicative."""
    y0, y1 = chart_encode(b0), chart_encode(b1)
    return chart_decode((1 - t) * y0 + t * y1)


def traj_logit(b0, b1, t):
    """Logit: 4-dim multiplicative-symmetric (sigmoid of linear logit)."""
    y0, y1 = logit_encode(b0), logit_encode(b1)
    return logit_decode((1 - t) * y0 + t * y1)


SPACES = [
    ("S-E (Signal Eucl)\nb_t = (1−t)b_0 + t b_1\n[linear in box]",          traj_signal, "tab:blue"),
    ("S-R (psi + signal model)\nb_t = psi_inv(linear y)\n[pos lin, size mult]",
                                                                            traj_chart,  "tab:purple"),
    ("C-R (chart_native)\nsame b_t as S-R\n[only model space differs]",     traj_chart,  "tab:olive"),
    ("Logit (013)\nb_t = sigmoid(linear y)\n[4-dim multiplicative]",        traj_logit,  "tab:orange"),
]


# ============================================================
# View 1: many (b_0, b_1) pairs, t = 0.5 snapshot
# ============================================================

def collect_pairs(N: int = 500, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ref = torch.empty(N, 4)
    b_0 = sample_init_box(ref, prior="default")
    ds = MNISTBoxDataset(split="train", root="./data")
    idx = np.random.randint(0, len(ds), size=N // 10 + 1)
    gt_list = []
    for i in idx:
        gt_list.append(ds[int(i)]["gt_boxes"].numpy())
    gt = np.concatenate(gt_list, axis=0)[:N]
    b_1 = torch.from_numpy(gt).float()
    return b_0, b_1


def draw_boxes(ax, boxes_np: np.ndarray, color: str, alpha: float = 0.3, lw: float = 0.4):
    for cx, cy, w, h in boxes_np:
        x0 = (cx - w / 2) * CANVAS
        y0 = (cy - h / 2) * CANVAS
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), w * CANVAS, h * CANVAS,
            fill=False, edgecolor=color, linewidth=lw, alpha=alpha,
        ))


def plot_t_snapshot(b_0, b_1, t_val: float, save_path: Path):
    t = torch.full((b_0.shape[0], 1), t_val)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, (title, fn, color) in zip(axes, SPACES):
        b_t = fn(b_0, b_1, t).numpy()
        ax.set_xlim(0, CANVAS); ax.set_ylim(CANVAS, 0); ax.set_aspect("equal")
        ax.set_facecolor("#f6f6f6")
        ax.set_title(title, fontsize=10)
        ax.add_patch(mpatches.Rectangle(
            (0, 0), CANVAS, CANVAS, fill=False, edgecolor="black", linewidth=1.0,
        ))
        draw_boxes(ax, b_t, color=color, alpha=0.30, lw=0.4)
        m = b_t.mean(axis=0)
        ax.text(
            0.02, 0.98,
            f"N={len(b_t)}, t={t_val}\n"
            f"⟨cx,cy⟩ = ({m[0]:.2f}, {m[1]:.2f})\n"
            f"⟨w,h⟩  = ({m[2]:.2f}, {m[3]:.2f})\n"
            f"size range [{b_t[:,2:].min():.2f}, {b_t[:,2:].max():.2f}]",
            transform=ax.transAxes, fontsize=8, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    fig.suptitle(
        f"Intermediate box b_t at t={t_val} — 4 spaces, same (b_0, b_1) pairs",
        fontsize=12,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")


# ============================================================
# View 2: a single (b_0, b_1) pair, 11 t-steps overlaid
# ============================================================

def plot_single_pair_sweep(save_path: Path):
    torch.manual_seed(7)
    # construct one pair: large central init → small corner GT
    b_0 = torch.tensor([[0.5, 0.5, 0.65, 0.65]])
    b_1 = torch.tensor([[0.78, 0.22, 0.10, 0.16]])
    ts = torch.linspace(0, 1, 11)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    cmap = cm.get_cmap("coolwarm")

    for ax, (title, fn, _) in zip(axes, SPACES):
        ax.set_xlim(0, CANVAS); ax.set_ylim(CANVAS, 0); ax.set_aspect("equal")
        ax.set_facecolor("#f6f6f6")
        ax.set_title(title, fontsize=10)
        ax.add_patch(mpatches.Rectangle(
            (0, 0), CANVAS, CANVAS, fill=False, edgecolor="black", linewidth=1.0,
        ))
        for i, t in enumerate(ts):
            t_b = torch.full((1, 1), float(t))
            b = fn(b_0, b_1, t_b)[0].numpy()
            cx, cy, w, h = b
            x0 = (cx - w / 2) * CANVAS
            y0 = (cy - h / 2) * CANVAS
            color = cmap(i / (len(ts) - 1))
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), w * CANVAS, h * CANVAS,
                fill=False, edgecolor=color, linewidth=1.5,
            ))
            ax.plot(cx * CANVAS, cy * CANVAS, "o", color=color, markersize=3.5)

        ax.text(0.02, 0.98,
                "blue: t=0 (init)\nred:  t=1 (GT)",
                transform=ax.transAxes, fontsize=8, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle(
        "Single (b_0, b_1) pair · 11 t-steps  —  same endpoints, different paths",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    out_dir = Path("outputs/init_viz")
    b_0, b_1 = collect_pairs(N=500)
    plot_t_snapshot(b_0, b_1, t_val=0.5, save_path=out_dir / "box_trajectory_t05.png")
    plot_single_pair_sweep(out_dir / "box_trajectory_sweep.png")
