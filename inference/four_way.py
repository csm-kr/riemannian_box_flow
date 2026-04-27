"""4-baseline comparison: load all 4 ckpts, compute IoU on shared b_0."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.trajectory import signal_decode

from .compare import cache_val_batches, load_model
from .metrics import (
    RATIO_BUCKETS,
    SIZE_BUCKETS,
    chart_mse,
    iou_xywh,
    paired_wilcoxon,
    signal_mse,
    stratify_by_ratio,
    stratify_by_size,
)


@torch.no_grad()
def collect(model, batches, K, seed, n_queries=10):
    preds, gts, inits = [], [], []
    for batch_idx, batch in enumerate(batches):
        image = batch["image"]; gt = batch["gt_boxes"]
        B = image.shape[0]
        torch.manual_seed(seed + batch_idx)
        s0 = torch.randn(B, n_queries, 4, device=image.device).clamp_(-3, 3)
        init_box = signal_decode(s0).clamp(0, 1)
        pred, _ = model.sample(image, K=K, init_box=init_box)
        preds.append(pred.cpu()); gts.append(gt.cpu()); inits.append(init_box.cpu())
    return torch.cat(preds, 0), torch.cat(inits, 0), torch.cat(gts, 0)


def run(ckpts, names, out_dir, K=10, seed=0, batch_size=64, val_root="./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_model(c, device)[0] for c in ckpts]
    val_loader = DataLoader(MNISTBoxDataset(split="val", root=val_root),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    print("caching val ...")
    batches = cache_val_batches(val_loader, device)

    preds = []; init_box = None; gt = None
    for name, model in zip(names, models):
        print(f"running {name} ...")
        p, ib, g = collect(model, batches, K, seed)
        preds.append(p)
        if init_box is None: init_box, gt = ib, g
        else: assert torch.allclose(init_box, ib), f"{name} init mismatch"
    del models  # free GPU

    # Per-model summary metrics
    rows = []
    for name, p in zip(names, preds):
        iou = iou_xywh(p, gt)
        cmse = chart_mse(p, gt)
        smse = signal_mse(p, gt)
        rows.append({
            "name":            name,
            "mean_iou":        float(iou.mean()),
            "median_iou":      float(iou.median()),
            "std_iou_per_sample": float(iou.mean(dim=1).std()),
            "mean_chart_mse":  float(cmse.mean()),
            "median_chart_mse": float(cmse.median()),
            "mean_signal_mse": float(smse.mean()),
            "median_signal_mse": float(smse.median()),
        })

    # Pairwise IoU diff (Wilcoxon, all pairs)
    pair_iou = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j: continue
            ia = iou_xywh(preds[i], gt).mean(dim=1)
            ib = iou_xywh(preds[j], gt).mean(dim=1)
            w = paired_wilcoxon(ia, ib)
            pair_iou[f"{ni}_minus_{nj}"] = {
                "median_diff": float(w.median_diff),
                "p_value":     float(w.p_value),
            }

    # Size-stratified per-model
    bucket = stratify_by_size(gt)
    bucket_names = ("small", "medium", "large")
    size_strat = {n: {} for n in names}
    for name, p in zip(names, preds):
        iou = iou_xywh(p, gt)
        for k, bn in enumerate(bucket_names):
            mask = bucket == k
            if mask.sum() == 0: continue
            size_strat[name][bn] = {
                "n_boxes":    int(mask.sum()),
                "mean_iou":   float(iou[mask].mean()),
                "median_iou": float(iou[mask].median()),
            }

    # Ratio-stratified
    rbucket = stratify_by_ratio(init_box, gt)
    ratio_strat = {n: {} for n in names}
    for name, p in zip(names, preds):
        iou = iou_xywh(p, gt)
        for k, bn in enumerate(bucket_names):
            mask = rbucket == k
            if mask.sum() == 0: continue
            ratio_strat[name][bn] = {
                "n_boxes":    int(mask.sum()),
                "mean_iou":   float(iou[mask].mean()),
                "median_iou": float(iou[mask].median()),
            }

    summary = {
        "ckpts": dict(zip(names, [str(c) for c in ckpts])),
        "K": K, "seed": seed, "n_total_boxes": int(gt.numel() // 4),
        "n_val_samples": int(gt.shape[0]),
        "size_buckets":  list(SIZE_BUCKETS),
        "ratio_buckets": list(RATIO_BUCKETS),
        "per_model":         rows,
        "pairwise_iou_diff": pair_iou,
        "size_stratified":   size_strat,
        "ratio_stratified":  ratio_strat,
    }
    with open(out_dir / "four_way.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty print table
    print("\n=== 4-way per-model summary ===")
    print(f"{'name':<14} {'mean_iou':>10} {'median_iou':>11} {'chart_mse':>11} {'signal_mse':>11}")
    for r in rows:
        print(f"{r['name']:<14} {r['mean_iou']:>10.4f} {r['median_iou']:>11.4f} "
              f"{r['median_chart_mse']:>11.6f} {r['median_signal_mse']:>11.6f}")
    print("\n=== pairwise mean IoU paired Wilcoxon ===")
    for k, v in pair_iou.items():
        print(f"  {k:<30}  median_diff={v['median_diff']:+.4f}  p={v['p_value']:.4g}")

    print(f"\nsaved: {out_dir / 'four_way.json'}")
    return summary


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="ckpt paths in order matching --names")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out-dir", default="outputs/comparison_4way")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    assert len(args.ckpts) == len(args.names), "ckpts and names lengths must match"
    run(args.ckpts, args.names, args.out_dir, K=args.K, seed=args.seed,
        batch_size=args.batch_size)
