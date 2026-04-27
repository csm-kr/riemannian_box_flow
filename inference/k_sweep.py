"""Phase 2 axis 3 — K (ODE step) sensitivity.

Hypothesis: Riemannian, being chart-straight, may be more robust at low K
(fewer Euler steps) than Euclidean.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.mnist_box_dataset import MNISTBoxDataset

from .compare import cache_val_batches, collect_predictions, load_model
from .metrics import iou_xywh, paired_wilcoxon


K_VALUES = (2, 4, 8, 16, 32)


def run(ckpt_a, ckpt_b, out_dir, *, k_values=K_VALUES, seed=0,
        batch_size=64, val_root="./data", max_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model_a, name_a = load_model(ckpt_a, device)
    model_b, name_b = load_model(ckpt_b, device)
    val_ds = MNISTBoxDataset(split="val", root=val_root)
    if max_batches is not None:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, list(range(max_batches * batch_size)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print("caching val batches ...")
    batches = cache_val_batches(val_loader, device)
    print(f"  {len(batches)} batches × {batches[0]['image'].shape[0]}")

    rows = []
    for K in k_values:
        print(f"K={K} ...")
        pa, _, gt = collect_predictions(model_a, batches, K=K, seed=seed)
        pb, _, _  = collect_predictions(model_b, batches, K=K, seed=seed)
        iou_a = iou_xywh(pa, gt).mean(dim=1)   # (N,) per-sample mean
        iou_b = iou_xywh(pb, gt).mean(dim=1)
        w = paired_wilcoxon(iou_a, iou_b)
        rows.append({
            "K": K,
            "mean_iou_a": float(iou_a.mean()),
            "mean_iou_b": float(iou_b.mean()),
            "median_diff_a_minus_b": float(w.median_diff),
            "p_value": float(w.p_value),
        })
        print(f"  iou_a={rows[-1]['mean_iou_a']:.4f}  iou_b={rows[-1]['mean_iou_b']:.4f}  "
              f"diff(a-b)={w.median_diff:+.4f}  p={w.p_value:.2g}")

    summary = {
        "ckpt_a": str(ckpt_a), "ckpt_b": str(ckpt_b),
        "model_a": name_a, "model_b": name_b,
        "seed": seed, "k_values": list(k_values),
        "n_total_boxes": int(gt.numel() // 4),
        "rows": rows,
    }
    with open(out_dir / "k_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "k_sweep.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print("k_sweep saved:", out_dir)
    return summary


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--out-dir", default="outputs/comparison")
    p.add_argument("--k-values", default="2,4,8,16,32",
                   help="comma-separated K values")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = _parse()
        ks = tuple(int(x) for x in args.k_values.split(","))
        run(args.ckpt_a, args.ckpt_b, args.out_dir,
            k_values=ks, seed=args.seed, batch_size=args.batch_size,
            max_batches=args.max_batches)
    else:
        # smoke
        ckpt = "outputs/001_fullrun/ckpt/final.pt"
        if not Path(ckpt).exists():
            print(f"smoke skipped — {ckpt} not found")
            raise SystemExit
        run(ckpt, ckpt, "/tmp/k_sweep_smoke", k_values=(2, 4),
            batch_size=8, max_batches=2)
        print("k_sweep smoke 통과")
