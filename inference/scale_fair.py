"""Exp 006 — Scale-fair metric evaluation across baselines.

Loads multiple ckpts, runs paired inference (shared b_0), computes:
- mean IoU (existing — biased to populous bucket)
- per-bucket-weighted IoU (equal weight per size bucket)
- log-size error (scale-invariant size accuracy)
- scale-relative center error (small/large box equally weighted)

For Riem-positive scenarios, scale-fair metrics often shift the ranking.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.mnist_box_dataset import MNISTBoxDataset

from .compare import cache_val_batches, collect_predictions, load_model
from .metrics import (
    SIZE_BUCKETS,
    iou_xywh,
    log_size_error,
    per_bucket_weighted_iou,
    scale_relative_center_err,
    stratify_by_size,
)


def run(ckpts, names, out_dir, *, K=10, seed=0, batch_size=64, val_root="./data",
        wide=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    val_ds = MNISTBoxDataset(split="val", root=val_root, wide=wide)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"caching val (wide={wide}) ...")
    batches = cache_val_batches(val_loader, device)

    rows = []
    for name, ckpt in zip(names, ckpts):
        print(f"running {name} ...")
        model, _ = load_model(ckpt, device)
        pred, init_box, gt = collect_predictions(model, batches, K=K, seed=seed)
        del model
        torch.cuda.empty_cache()

        iou = iou_xywh(pred, gt)                            # (N, 10)
        bucket = stratify_by_size(gt, thresholds=SIZE_BUCKETS)
        log_size = log_size_error(pred, gt)
        sr_center = scale_relative_center_err(pred, gt)

        rows.append({
            "name":                          name,
            "mean_iou":                      float(iou.mean()),
            "median_iou":                    float(iou.median()),
            "per_bucket_weighted_iou":       float(per_bucket_weighted_iou(iou, bucket)),
            "log_size_err_median":           float(log_size.median()),
            "log_size_err_mean":             float(log_size.mean()),
            "scale_relative_center_median":  float(sr_center.median()),
            "scale_relative_center_mean":    float(sr_center.mean()),
            "iou_small_bucket":  float(iou[bucket == 0].mean()) if (bucket == 0).any() else None,
            "iou_medium_bucket": float(iou[bucket == 1].mean()) if (bucket == 1).any() else None,
            "iou_large_bucket":  float(iou[bucket == 2].mean()) if (bucket == 2).any() else None,
        })

    with open(out_dir / "scale_fair.json", "w") as f:
        json.dump({"K": K, "seed": seed, "wide": wide,
                   "ckpts": dict(zip(names, [str(c) for c in ckpts])),
                   "rows": rows}, f, indent=2)
    with open(out_dir / "scale_fair.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # Pretty table
    print("\n=== scale-fair metric table ===")
    print(f"{'name':<14} {'iou':>7} {'iou-bw':>8} {'log_size':>9} "
          f"{'rel_ctr':>9} {'iou_S':>7} {'iou_M':>7} {'iou_L':>7}")
    for r in rows:
        s = r['iou_small_bucket'];  s = f"{s:.3f}" if s is not None else "  -"
        m = r['iou_medium_bucket']; m = f"{m:.3f}" if m is not None else "  -"
        l = r['iou_large_bucket'];  l = f"{l:.3f}" if l is not None else "  -"
        print(f"{r['name']:<14} {r['mean_iou']:>7.4f} "
              f"{r['per_bucket_weighted_iou']:>8.4f} "
              f"{r['log_size_err_median']:>9.5f} "
              f"{r['scale_relative_center_median']:>9.5f} "
              f"{s:>7} {m:>7} {l:>7}")
    print("legend: iou=mean IoU, iou-bw=bucket-weighted IoU, "
          "log_size=median log-size error, rel_ctr=median scale-relative center err")
    print(f"\nsaved: {out_dir / 'scale_fair.json'}, {out_dir / 'scale_fair.csv'}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out-dir", default="outputs/comparison_riem_strength")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--wide", action="store_true",
                   help="Use wide-scale val dataset (for 005, 006 ckpts).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    assert len(args.ckpts) == len(args.names)
    run(args.ckpts, args.names, args.out_dir, K=args.K, seed=args.seed,
        batch_size=args.batch_size, wide=args.wide)
