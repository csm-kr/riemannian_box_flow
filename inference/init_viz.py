"""Visualize init distribution across spaces (signal / chart / logit) and
priors (default / small_size).

Saves outputs/init_viz/init_distributions.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.trajectory import (
    chart_encode, logit_encode, sample_init_box, signal_encode,
)


SPACES = {
    "box": lambda b: b,
    "signal\n(s = 6b − 3)": signal_encode,
    "chart psi\n(pos raw, size log)": chart_encode,
    "logit\n(4 dims symmetric)": logit_encode,
}

PRIORS = ["default", "small_size"]
COLORS = {"default": "#1f77b4", "small_size": "#ff7f0e"}


def collect(N: int = 50_000, seed: int = 0):
    torch.manual_seed(seed)
    ref = torch.empty(N, 4)
    out = {}
    for prior in PRIORS:
        b = sample_init_box(ref, prior=prior)
        for sp_name, sp_fn in SPACES.items():
            y = sp_fn(b).numpy()                 # (N, 4)
            out[(prior, sp_name)] = y
    return out


def plot(data, save_path: Path):
    fig, axes = plt.subplots(2, len(SPACES), figsize=(5 * len(SPACES), 7))
    for col, sp_name in enumerate(SPACES.keys()):
        for row, (kind, slc) in enumerate([
            ("pos (cx, cy)", slice(0, 2)),
            ("size (w, h)", slice(2, 4)),
        ]):
            ax = axes[row, col]
            for prior in PRIORS:
                vals = data[(prior, sp_name)][:, slc].flatten()
                ax.hist(
                    vals, bins=80, alpha=0.55, density=True,
                    color=COLORS[prior], label=prior,
                )
            ax.set_title(f"{sp_name}\n{kind}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            # annotate min/max for default prior
            v_def = data[("default", sp_name)][:, slc].flatten()
            ax.text(
                0.02, 0.97,
                f"default: [{v_def.min():.2f}, {v_def.max():.2f}]\n"
                f"          μ={v_def.mean():.2f}, σ={v_def.std():.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
    fig.suptitle(
        "Init b_0 distribution across spaces & priors  (N=50000)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")


def plot_2d_scatter(data, save_path: Path):
    """2D scatter (cx, cy) and (w, h) per space — see joint distribution."""
    fig, axes = plt.subplots(2, len(SPACES), figsize=(5 * len(SPACES), 9))
    for col, sp_name in enumerate(SPACES.keys()):
        for row, (kind, slc) in enumerate([
            ("pos: (cx, cy)", slice(0, 2)),
            ("size: (w, h)", slice(2, 4)),
        ]):
            ax = axes[row, col]
            for prior in PRIORS:
                v = data[(prior, sp_name)][:, slc]
                ax.scatter(v[:2000, 0], v[:2000, 1], s=2, alpha=0.25,
                           color=COLORS[prior], label=prior)
            ax.set_title(f"{sp_name}\n{kind}", fontsize=10)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, markerscale=3)
    fig.suptitle(
        "Init joint scatter (2000 samples shown per prior)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    data = collect(N=50_000)
    out_dir = Path("outputs/init_viz")
    plot(data, out_dir / "init_distributions.png")
    plot_2d_scatter(data, out_dir / "init_scatter.png")
