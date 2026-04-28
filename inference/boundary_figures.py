"""Phase 3.0 H1 — figures from boundary_audit output.

Reads outputs/logit_strengths/boundary_audit/{boundary_metrics.json, excess_l1.pt}
and writes 3 figures.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _load(in_dir):
    with open(in_dir / "boundary_metrics.json") as f:
        summary = json.load(f)
    excess_path = in_dir / "excess_l1.pt"
    excess = torch.load(excess_path, weights_only=False) if excess_path.exists() else None
    return summary, excess


def _split_rows(rows):
    """rows: list of dicts {K, model, ...}. Returns dict[model] = sorted list by K."""
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for k in by_model:
        by_model[k].sort(key=lambda r: r["K"])
    return by_model


def fig_iou_vs_K(summary, out_path):
    by = _split_rows(summary["rows"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for model, rows in by.items():
        Ks = [r["K"] for r in rows]
        ax[0].plot(Ks, [r["iou_clamp_mean"] for r in rows],
                   marker="o", label=model)
        ax[1].plot(Ks, [r["iou_no_clamp_mean"] for r in rows],
                   marker="o", label=model)
    for a, title in zip(ax, ("IoU (clamp(0,1) post-decode)", "IoU (raw, no clamp)")):
        a.set_xscale("log", base=2)
        a.set_xlabel("ODE steps K")
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend()
    ax[0].set_ylabel("mean IoU")
    fig.suptitle("H1 — IoU vs K  (signal vs logit_native, paired init)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_ooc_vs_K(summary, out_path):
    by = _split_rows(summary["rows"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for model, rows in by.items():
        Ks = [r["K"] for r in rows]
        ax[0].plot(Ks, [r["coord_oob_rate"] for r in rows],
                   marker="o", label=model)
        ax[1].plot(Ks, [r["canvas_oob_rate"] for r in rows],
                   marker="o", label=model)
    ax[0].set_title("coord-OOB rate  (any of cx,cy,w,h ∉ [0,1])")
    ax[1].set_title("canvas-OOB rate (cx±w/2 or cy±h/2 ∉ [0,1])")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("ODE steps K")
        a.set_ylabel("rate")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("H1 — boundary violation rate vs K")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_excess_hist(excess, out_path, k_main):
    if excess is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 40
    for model, vals in excess.items():
        v = vals.numpy()
        ax.hist(v, bins=bins, alpha=0.55, label=f"{model}  (mean={v.mean():.4f})",
                log=True)
    ax.set_xlabel("excess_l1 = Σ max(0, -val) + max(0, val-1)  per box")
    ax.set_ylabel("count (log)")
    ax.set_title(f"H1 — boundary excess distribution at K={k_main}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(in_dir):
    in_dir = Path(in_dir)
    summary, excess = _load(in_dir)
    fig_iou_vs_K(summary, in_dir / "iou_vs_K.png")
    fig_ooc_vs_K(summary, in_dir / "ooc_vs_K.png")
    k_main = max(summary["k_values"])
    fig_excess_hist(excess, in_dir / "excess_hist.png", k_main)
    print(f"figures written → {in_dir}")
    print(" - iou_vs_K.png")
    print(" - ooc_vs_K.png")
    print(" - excess_hist.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",
                   default="outputs/logit_strengths/boundary_audit")
    args = p.parse_args()
    main(args.in_dir)
