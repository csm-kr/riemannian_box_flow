"""Phase 2 — figures from comparison JSON outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_size_iou(summary_json, out_path):
    s = json.loads(Path(summary_json).read_text())
    strat = s["axis_2_size_stratified"]
    rstrat = s.get("axis_2b_ratio_stratified", {})
    name_a, name_b = s["model_a"], s["model_b"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    buckets = list(strat.keys())
    iou_a = [strat[b]["mean_iou_a"] for b in buckets]
    iou_b = [strat[b]["mean_iou_b"] for b in buckets]
    n = [strat[b]["n_boxes"] for b in buckets]
    x = range(len(buckets))
    axes[0].bar([i - 0.18 for i in x], iou_a, width=0.35, label=name_a, color="tab:blue")
    axes[0].bar([i + 0.18 for i in x], iou_b, width=0.35, label=name_b, color="tab:orange")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f"{b}\n(n={n[i]})" for i, b in enumerate(buckets)])
    axes[0].set_ylabel("mean IoU")
    axes[0].set_title("axis 2 — by GT box size")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    if rstrat:
        rbuckets = list(rstrat.keys())
        ra = [rstrat[b]["mean_iou_a"] for b in rbuckets]
        rb = [rstrat[b]["mean_iou_b"] for b in rbuckets]
        rn = [rstrat[b]["n_boxes"] for b in rbuckets]
        rx = range(len(rbuckets))
        axes[1].bar([i - 0.18 for i in rx], ra, width=0.35, label=name_a, color="tab:blue")
        axes[1].bar([i + 0.18 for i in rx], rb, width=0.35, label=name_b, color="tab:orange")
        axes[1].set_xticks(list(rx))
        axes[1].set_xticklabels([f"{b}\n(n={rn[i]})" for i, b in enumerate(rbuckets)])
        axes[1].set_ylabel("mean IoU")
        axes[1].set_title("axis 2b — by size-change ratio (init→GT)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("saved:", out_path)


def plot_k_sweep(k_json, out_path):
    s = json.loads(Path(k_json).read_text())
    rows = s["rows"]
    name_a, name_b = s["model_a"], s["model_b"]
    Ks = [r["K"] for r in rows]
    a = [r["mean_iou_a"] for r in rows]
    b = [r["mean_iou_b"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ks, a, "o-", label=name_a, color="tab:blue")
    ax.plot(Ks, b, "o-", label=name_b, color="tab:orange")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("K (ODE Euler steps)")
    ax.set_ylabel("mean IoU (val 5000)")
    ax.set_title("axis 3 — K sensitivity")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("saved:", out_path)


def plot_seed_variance(sv_json, out_path):
    s = json.loads(Path(sv_json).read_text())
    name_a, name_b = s["model_a"], s["model_b"]
    rows = s["per_image"]
    std_a = [r["std_iou_a"] for r in rows]
    std_b = [r["std_iou_b"] for r in rows]
    mean_a = [r["mean_iou_a"] for r in rows]
    mean_b = [r["mean_iou_b"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].boxplot([std_a, std_b], labels=[name_a, name_b])
    axes[0].set_ylabel("per-image IoU std (across seeds)")
    axes[0].set_title(f"axis 4 — IoU stability (lower = more robust to init)\n"
                      f"n_seeds={s['n_seeds']}, n_images={s['n_images']}")
    axes[0].grid(alpha=0.3, axis='y')

    axes[1].boxplot([mean_a, mean_b], labels=[name_a, name_b])
    axes[1].set_ylabel("per-image mean IoU")
    axes[1].set_title("axis 4 — IoU mean across seeds")
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("saved:", out_path)


def main(comparison_dir):
    cdir = Path(comparison_dir)
    fdir = cdir / "figures"; fdir.mkdir(exist_ok=True)
    if (cdir / "summary.json").exists():
        plot_size_iou(cdir / "summary.json", fdir / "size_iou.png")
    if (cdir / "k_sweep.json").exists():
        plot_k_sweep(cdir / "k_sweep.json", fdir / "k_sweep.png")
    if (cdir / "seed_variance.json").exists():
        plot_seed_variance(cdir / "seed_variance.json", fdir / "seed_variance.png")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-dir", default="outputs/comparison")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    main(args.comparison_dir)
