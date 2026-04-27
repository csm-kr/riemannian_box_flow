"""Phase 2 axis 4 — init seed variance.

Hypothesis: Riemannian's chart geometry is more robust to s_0 init noise
→ smaller IoU variance across seeds for the same image.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.mnist_box_dataset import MNISTBoxDataset

from .compare import cache_val_batches, load_model
from .metrics import iou_xywh, paired_wilcoxon


@torch.no_grad()
def collect_per_seed(model, batch, n_seeds, K, base_seed=10000):
    """For one batch, run sample() with n_seeds different seeds.
    Returns iou per (seed, B*10) shape (n_seeds, B, 10)."""
    image = batch["image"]
    gt = batch["gt_boxes"]
    out = []
    for s in range(n_seeds):
        torch.manual_seed(base_seed + s)
        pred, _ = model.sample(image, K=K)
        out.append(iou_xywh(pred, gt).cpu())
    return torch.stack(out, dim=0)            # (n_seeds, B, 10)


def run(ckpt_a, ckpt_b, out_dir, *, n_seeds=100, n_images=20, K=10,
        batch_size=20, val_root="./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    assert n_images <= batch_size, "n_images must fit one batch"

    model_a, name_a = load_model(ckpt_a, device)
    model_b, name_b = load_model(ckpt_b, device)

    val_ds = MNISTBoxDataset(split="val", root=val_root)
    from torch.utils.data import Subset
    val_ds = Subset(val_ds, list(range(batch_size)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = cache_val_batches(val_loader, device)
    batch = batches[0]
    print(f"seed_var: {n_images} images × {n_seeds} seeds × 10 queries; K={K}")

    iou_a = collect_per_seed(model_a, batch, n_seeds, K)[:, :n_images]   # (S, I, 10)
    iou_b = collect_per_seed(model_b, batch, n_seeds, K)[:, :n_images]

    # mean & std across seeds for each (image, query)
    mean_a = iou_a.mean(dim=0)        # (I, 10)
    mean_b = iou_b.mean(dim=0)
    std_a  = iou_a.std(dim=0)
    std_b  = iou_b.std(dim=0)

    # Aggregate: per-image mean and std (avg over 10 queries)
    img_mean_a = mean_a.mean(dim=1)   # (I,)
    img_mean_b = mean_b.mean(dim=1)
    img_std_a  = std_a.mean(dim=1)    # avg per-query std
    img_std_b  = std_b.mean(dim=1)

    summary = {
        "ckpt_a": str(ckpt_a), "ckpt_b": str(ckpt_b),
        "model_a": name_a, "model_b": name_b,
        "n_images": int(n_images), "n_seeds": int(n_seeds), "K": K,
        "overall": {
            "mean_iou_a":     float(mean_a.mean()),
            "mean_iou_b":     float(mean_b.mean()),
            "mean_std_a":     float(std_a.mean()),
            "mean_std_b":     float(std_b.mean()),
            "wilcoxon_std":   asdict_or_none(paired_wilcoxon(img_std_a, img_std_b)),
            "wilcoxon_mean":  asdict_or_none(paired_wilcoxon(img_mean_a, img_mean_b)),
        },
        "per_image": [
            {"image_idx": i,
             "mean_iou_a": float(img_mean_a[i]), "mean_iou_b": float(img_mean_b[i]),
             "std_iou_a":  float(img_std_a[i]),  "std_iou_b":  float(img_std_b[i])}
            for i in range(n_images)
        ],
    }
    with open(out_dir / "seed_variance.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "seed_variance.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary["per_image"][0].keys()))
        wr.writeheader(); wr.writerows(summary["per_image"])

    print(f"overall mean_iou: a={summary['overall']['mean_iou_a']:.4f} "
          f"b={summary['overall']['mean_iou_b']:.4f}")
    print(f"overall mean_std (across seeds, avg over images & queries): "
          f"a={summary['overall']['mean_std_a']:.4f} "
          f"b={summary['overall']['mean_std_b']:.4f}")
    print(f"per-image-std Wilcoxon p={summary['overall']['wilcoxon_std']['p_value']:.4g} "
          f"(median diff a-b = {summary['overall']['wilcoxon_std']['median_diff']:+.4f})")
    print("seed_var saved:", out_dir)
    return summary


def asdict_or_none(res):
    from dataclasses import asdict
    return asdict(res)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--out-dir", default="outputs/comparison")
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = _parse()
        run(args.ckpt_a, args.ckpt_b, args.out_dir,
            n_seeds=args.n_seeds, n_images=args.n_images, K=args.K,
            batch_size=args.batch_size)
    else:
        ckpt = "outputs/001_fullrun/ckpt/final.pt"
        if not Path(ckpt).exists():
            print(f"smoke skipped — {ckpt} not found")
            raise SystemExit
        run(ckpt, ckpt, "/tmp/seed_var_smoke",
            n_seeds=4, n_images=2, K=4, batch_size=2)
        print("seed_var smoke 통과")
