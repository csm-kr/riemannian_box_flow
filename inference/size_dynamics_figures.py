"""Phase 3.0 H2 — figures from size_dynamics output.

Reads outputs/logit_strengths/size_dynamics/summary.json and writes 4 figures.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(in_dir):
    with open(in_dir / "summary.json") as f:
        return json.load(f)


def _grouped_bar(rows, names, title, ylabel, out_path,
                  label_a="signal", label_b="logit_native",
                  highlight_diff=True):
    """rows = list[dict] with keys mean_a, mean_b, n_boxes, p_value."""
    a = [r["mean_a"] if r["mean_a"] is not None else 0.0 for r in rows]
    b = [r["mean_b"] if r["mean_b"] is not None else 0.0 for r in rows]
    n = [r["n_boxes"] for r in rows]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - w/2, a, w, label=label_a, color="#1f77b4")
    ax.bar(x + w/2, b, w, label=label_b, color="#ff7f0e")
    if highlight_diff:
        for xi, ra, rb, ni in zip(x, a, b, n):
            if ni == 0:
                continue
            top = max(ra, rb)
            ax.text(xi, top + 0.005, f"Δ={rb-ra:+.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\n(n={ni})" for nm, ni in zip(names, n)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_iou_by_fine_size(summary, out_path):
    rows = summary["size_strat"]["iou"]
    _grouped_bar(rows, summary["size_names"],
                 "H2a — mean IoU by GT-size bucket  (5000 sample paired)",
                 "mean IoU", out_path)


def fig_size_err_by_fine_size(summary, out_path):
    rows = summary["size_strat"]["size_err"]
    _grouped_bar(rows, summary["size_names"],
                 "H2c — size_err by GT-size bucket  (lower better)",
                 "mean size_err  (|Δw|+|Δh|)", out_path)


def fig_iou_by_ratio(summary, out_path):
    rows = summary["ratio_strat"]["iou"]
    _grouped_bar(rows, summary["ratio_names"],
                 "H2b — mean IoU by size-change ratio  (init→GT)",
                 "mean IoU", out_path)


def fig_iou_diff_heatmap(summary, out_path):
    """Cell value = IoU(logit) − IoU(signal) per (size, ratio) bucket."""
    rows  = summary["heatmap"]["rows"]
    cols  = summary["heatmap"]["cols"]
    diff_signal_minus_logit = np.array(summary["heatmap"]["diff"], dtype=float)
    counts = np.array(summary["heatmap"]["counts"], dtype=int)
    diff = -diff_signal_minus_logit            # logit − signal (positive ⇒ logit better)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.RdBu_r
    vmax = float(np.nanmax(np.abs(diff))) if np.any(~np.isnan(diff)) else 0.05
    vmin = -vmax
    im = ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="IoU(logit) − IoU(signal)")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("size_change_ratio (init → GT)")
    ax.set_ylabel("GT-size bucket")

    for i in range(len(rows)):
        for j in range(len(cols)):
            n = counts[i, j]
            d = diff[i, j]
            if n == 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888")
            else:
                txt = f"{d:+.3f}\nn={n}"
                color = "white" if abs(d) > 0.5 * vmax else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    ax.set_title("H2 — IoU(logit) − IoU(signal) per (size × ratio) cell")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(in_dir):
    in_dir = Path(in_dir)
    summary = _load(in_dir)
    fig_iou_by_fine_size(summary,    in_dir / "iou_by_fine_size.png")
    fig_size_err_by_fine_size(summary, in_dir / "size_err_by_fine_size.png")
    fig_iou_by_ratio(summary,        in_dir / "iou_by_ratio.png")
    fig_iou_diff_heatmap(summary,    in_dir / "iou_diff_heatmap.png")
    print(f"figures written → {in_dir}")
    for name in ("iou_by_fine_size.png", "size_err_by_fine_size.png",
                  "iou_by_ratio.png", "iou_diff_heatmap.png"):
        print(f" - {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",
                   default="outputs/logit_strengths/size_dynamics")
    args = p.parse_args()
    main(args.in_dir)
