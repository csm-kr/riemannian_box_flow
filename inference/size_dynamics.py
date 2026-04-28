"""Phase 3.0 H2 — scale-sensitive w, h dynamics 검증.

비교: 001 signal (final.pt) vs 013 logit_native (final.pt)

가설:
  H2a tiny / small bucket 에서 logit 우위 가장 큼, huge bucket 에선 격차 줄어듦
  H2b size_change_ratio ≥ 4 인 케이스에서 logit 우위가 ratio 작은 케이스보다 큼
  H2c size-only error 만 봐도 H2a / H2b 가 유지 (center 영향 아님)

메커니즘:
  signal: w 변화는 단순 affine — small box / large box 가 같은 magnitude 의 학습 신호
  logit:  d logit / d w = 1 / w (1−w) — 작은 w 일수록 큰 gradient → small box 학습 신호 강화

Buckets:
  fine size (per-box, GT box area = w·h):
    tiny=[0, 0.0025)  small=[0.0025, 0.01)  mid=[0.01, 0.04)  large=[0.04, 0.16)  huge=[0.16, 1]
    (side-length thresholds [0, 0.05, 0.10, 0.20, 0.40, 1.0])
  size_change_ratio (per-box, init→gt): 5 buckets [1, 1.5, 2, 4, 8, ∞]
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.mnist_box_dataset import MNISTBoxDataset

from .compare import cache_val_batches, collect_predictions, load_model
from .metrics import (
    center_error, iou_xywh, paired_wilcoxon, size_change_ratio, size_error,
)


# Fine size buckets — area = w * h, side-length thresholds [0.05, 0.10, 0.20, 0.40].
# tiny < 0.0025 ≤ small < 0.01 ≤ mid < 0.04 ≤ large < 0.16 ≤ huge
FINE_SIZE_THRESHOLDS = (0.0025, 0.01, 0.04, 0.16)
FINE_SIZE_NAMES      = ("tiny", "small", "mid", "large", "huge")

# Ratio buckets — geometric mean of max(w_1/w_0, w_0/w_1).
RATIO_FINE_THRESHOLDS = (1.5, 2.0, 4.0, 8.0)
RATIO_FINE_NAMES      = ("≤1.5", "1.5–2", "2–4", "4–8", "≥8")


def stratify(values, thresholds):
    """Bucket each value by the supplied thresholds. Returns long indices."""
    bucket = torch.zeros_like(values, dtype=torch.long)
    for k, t in enumerate(thresholds, start=1):
        bucket[values >= t] = k
    return bucket


def stratify_size_fine(boxes):
    """Bucket boxes by area = w * h with FINE_SIZE_THRESHOLDS."""
    return stratify(boxes[..., 2] * boxes[..., 3], FINE_SIZE_THRESHOLDS)


def stratify_ratio_fine(b_init, b_gt):
    return stratify(size_change_ratio(b_init, b_gt), RATIO_FINE_THRESHOLDS)


def _bucket_summary(metric_a, metric_b, bucket, names):
    """Per-bucket mean + paired Wilcoxon (a − b). Returns list[dict]."""
    rows = []
    for k, name in enumerate(names):
        mask = bucket == k
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": name, "n_boxes": 0, "mean_a": None,
                         "mean_b": None, "median_diff_a_minus_b": None,
                         "p_value": None})
            continue
        a = metric_a[mask]; b = metric_b[mask]
        w = paired_wilcoxon(a, b)
        rows.append({
            "bucket": name,
            "n_boxes": n,
            "mean_a":  float(a.mean()),
            "mean_b":  float(b.mean()),
            "median_diff_a_minus_b": float(w.median_diff),
            "p_value":               float(w.p_value),
        })
    return rows


def _heatmap_iou_diff(iou_a, iou_b, size_bucket, ratio_bucket):
    """Mean (iou_a − iou_b) per (size, ratio) cell. Returns (5, 5) tensor + counts."""
    n_s = len(FINE_SIZE_NAMES)
    n_r = len(RATIO_FINE_NAMES)
    diff = torch.full((n_s, n_r), float("nan"))
    counts = torch.zeros((n_s, n_r), dtype=torch.long)
    for i in range(n_s):
        for j in range(n_r):
            mask = (size_bucket == i) & (ratio_bucket == j)
            n = int(mask.sum())
            counts[i, j] = n
            if n == 0:
                continue
            diff[i, j] = (iou_a[mask] - iou_b[mask]).mean()
    return diff, counts


def run(ckpt_signal, ckpt_logit, out_dir, *, K=10, seed=0,
        batch_size=64, val_root="./data", max_batches=None,
        same_ckpt_smoke=False):
    """Run H2 size_dynamics. Returns summary dict.

    same_ckpt_smoke: when True, skips the second model load and reuses
                     the first to confirm all paired metrics are zero.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model_a, name_a = load_model(ckpt_signal, device)
    if same_ckpt_smoke:
        model_b, name_b = model_a, name_a
    else:
        model_b, name_b = load_model(ckpt_logit, device)

    val_ds = MNISTBoxDataset(split="val", root=val_root)
    if max_batches is not None:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, list(range(max_batches * batch_size)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print("caching val batches ...")
    batches = cache_val_batches(val_loader, device)
    n_total = sum(b["image"].shape[0] for b in batches)
    print(f"  {len(batches)} batches × {batches[0]['image'].shape[0]} = {n_total}")

    print(f"collecting predictions  K={K}  seed={seed}")
    pred_a, init_a, gt = collect_predictions(model_a, batches, K, seed)
    pred_b, init_b, _  = collect_predictions(model_b, batches, K, seed)
    assert torch.allclose(init_a, init_b), "paired init mismatch"

    # Per-box metrics (shape (N, 10))
    iou_a   = iou_xywh(pred_a, gt);   iou_b   = iou_xywh(pred_b, gt)
    serr_a  = size_error(pred_a, gt); serr_b  = size_error(pred_b, gt)
    cerr_a  = center_error(pred_a, gt); cerr_b = center_error(pred_b, gt)

    # Bucket assignments (per box)
    size_bucket  = stratify_size_fine(gt)            # (N, 10)
    ratio_bucket = stratify_ratio_fine(init_a, gt)   # (N, 10)

    # Flatten to (N*10,) for bucket aggregation
    flat_iou_a, flat_iou_b   = iou_a.flatten(),   iou_b.flatten()
    flat_serr_a, flat_serr_b = serr_a.flatten(),  serr_b.flatten()
    flat_cerr_a, flat_cerr_b = cerr_a.flatten(),  cerr_b.flatten()
    flat_size  = size_bucket.flatten()
    flat_ratio = ratio_bucket.flatten()

    # H2a: fine size bucket × {iou, size_err, center_err}
    size_strat = {
        "iou":        _bucket_summary(flat_iou_a,  flat_iou_b,  flat_size, FINE_SIZE_NAMES),
        "size_err":   _bucket_summary(flat_serr_a, flat_serr_b, flat_size, FINE_SIZE_NAMES),
        "center_err": _bucket_summary(flat_cerr_a, flat_cerr_b, flat_size, FINE_SIZE_NAMES),
    }
    # H2b: ratio bucket × {iou, size_err, center_err}
    ratio_strat = {
        "iou":        _bucket_summary(flat_iou_a,  flat_iou_b,  flat_ratio, RATIO_FINE_NAMES),
        "size_err":   _bucket_summary(flat_serr_a, flat_serr_b, flat_ratio, RATIO_FINE_NAMES),
        "center_err": _bucket_summary(flat_cerr_a, flat_cerr_b, flat_ratio, RATIO_FINE_NAMES),
    }
    # 2D heatmap: (iou_a − iou_b) per (size, ratio); a=signal, b=logit so signal − logit.
    diff, counts = _heatmap_iou_diff(flat_iou_a, flat_iou_b, flat_size, flat_ratio)

    # Overall (paired Wilcoxon on per-sample mean IoU)
    sample_a = iou_a.mean(dim=1)
    sample_b = iou_b.mean(dim=1)
    overall_w = paired_wilcoxon(sample_a, sample_b)
    overall = {
        "model_a":       name_a,
        "model_b":       name_b,
        "mean_iou_a":    float(sample_a.mean()),
        "mean_iou_b":    float(sample_b.mean()),
        "wilcoxon":      asdict(overall_w),
        "n_val_samples": int(sample_a.shape[0]),
        "n_boxes":       int(iou_a.numel()),
    }

    summary = {
        "ckpt_a":      str(ckpt_signal),
        "ckpt_b":      str(ckpt_logit) if not same_ckpt_smoke else str(ckpt_signal),
        "K":           K, "seed": seed,
        "size_thresholds":  list(FINE_SIZE_THRESHOLDS),
        "size_names":       list(FINE_SIZE_NAMES),
        "ratio_thresholds": list(RATIO_FINE_THRESHOLDS),
        "ratio_names":      list(RATIO_FINE_NAMES),
        "overall":          overall,
        "size_strat":       size_strat,
        "ratio_strat":      ratio_strat,
        "heatmap": {
            "rows":   list(FINE_SIZE_NAMES),
            "cols":   list(RATIO_FINE_NAMES),
            "diff":   diff.tolist(),       # a − b (signal − logit)
            "counts": counts.tolist(),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_strat_csv(out_dir / "stratified.csv",
                     size_strat, ratio_strat,
                     FINE_SIZE_NAMES, RATIO_FINE_NAMES,
                     name_a, name_b)
    print(f"saved → {out_dir}")
    return summary


def _write_strat_csv(path, size_strat, ratio_strat, size_names, ratio_names,
                     name_a, name_b):
    fields = ["axis", "bucket", "metric", "n_boxes",
              f"mean_{name_a}", f"mean_{name_b}",
              "median_diff_a_minus_b", "p_value"]
    rows = []
    for metric, sub in size_strat.items():
        for r in sub:
            rows.append({"axis": "size", "bucket": r["bucket"], "metric": metric,
                         "n_boxes": r["n_boxes"],
                         f"mean_{name_a}": r["mean_a"],
                         f"mean_{name_b}": r["mean_b"],
                         "median_diff_a_minus_b": r["median_diff_a_minus_b"],
                         "p_value": r["p_value"]})
    for metric, sub in ratio_strat.items():
        for r in sub:
            rows.append({"axis": "ratio", "bucket": r["bucket"], "metric": metric,
                         "n_boxes": r["n_boxes"],
                         f"mean_{name_a}": r["mean_a"],
                         f"mean_{name_b}": r["mean_b"],
                         "median_diff_a_minus_b": r["median_diff_a_minus_b"],
                         "p_value": r["p_value"]})
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)


def _parse():
    p = argparse.ArgumentParser(description="Phase 3.0 H2 size_dynamics")
    p.add_argument("--ckpt-signal", required=True)
    p.add_argument("--ckpt-logit",  required=True)
    p.add_argument("--out-dir", default="outputs/logit_strengths/size_dynamics")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def _smoke():
    # 1) stratify correctness
    boxes = torch.tensor([
        [0.5, 0.5, 0.04, 0.04],   # area 0.0016 → tiny  (<0.0025)
        [0.5, 0.5, 0.07, 0.07],   # area 0.0049 → small
        [0.5, 0.5, 0.15, 0.15],   # area 0.0225 → mid
        [0.5, 0.5, 0.30, 0.30],   # area 0.0900 → large
        [0.5, 0.5, 0.50, 0.50],   # area 0.2500 → huge
    ])
    bucket = stratify_size_fine(boxes)
    assert bucket.tolist() == [0, 1, 2, 3, 4], bucket.tolist()
    print("  [1] stratify_size_fine 5-bucket correctness ✓")

    # 2) ratio bucket correctness
    b0 = torch.tensor([[0.5, 0.5, 0.10, 0.10]] * 5)
    b1 = torch.tensor([
        [0.5, 0.5, 0.10, 0.10],   # ratio = 1
        [0.5, 0.5, 0.18, 0.18],   # ratio = 1.8
        [0.5, 0.5, 0.30, 0.30],   # ratio = 3
        [0.5, 0.5, 0.60, 0.60],   # ratio = 6
        [0.5, 0.5, 0.90, 0.90],   # ratio = 9
    ])
    rbucket = stratify_ratio_fine(b0, b1)
    assert rbucket.tolist() == [0, 1, 2, 3, 4], rbucket.tolist()
    print("  [2] stratify_ratio_fine 5-bucket correctness ✓")

    # 3) _bucket_summary on synthetic — same metric → diff=0, p=1
    a = torch.tensor([0.7, 0.6, 0.8, 0.5, 0.9])
    b = a.clone()
    bk = torch.tensor([0, 0, 1, 1, 2])
    rows = _bucket_summary(a, b, bk, ("A", "B", "C"))
    for r in rows:
        if r["n_boxes"] > 0:
            assert r["median_diff_a_minus_b"] == 0.0, r
            assert r["p_value"] == 1.0, r
    print("  [3] _bucket_summary same-data: diff=0, p=1 ✓")

    # 4) heatmap shape
    diff, counts = _heatmap_iou_diff(a, b, bk, torch.tensor([0, 1, 0, 1, 2]))
    assert diff.shape == (5, 5) and counts.shape == (5, 5)
    print("  [4] heatmap shape (5, 5) ✓")

    # 5) Same-ckpt run if available — diff=0 across all buckets/metrics
    ckpt_s = Path("outputs/001_fullrun/ckpt/final.pt")
    if not ckpt_s.exists():
        print("  [skip] full smoke — ckpt missing")
        print("inference/size_dynamics smoke 통과 (synthetic only)")
        return

    out = run(ckpt_s, ckpt_s, "/tmp/size_dynamics_smoke",
              batch_size=8, max_batches=2, same_ckpt_smoke=True)
    # All bucket diffs should be 0 (or None for empty buckets)
    for axis_name, axis in (("size", out["size_strat"]),
                             ("ratio", out["ratio_strat"])):
        for metric, rows in axis.items():
            for r in rows:
                if r["median_diff_a_minus_b"] is None:
                    continue
                assert r["median_diff_a_minus_b"] == 0.0, (axis_name, metric, r)
                assert r["p_value"] == 1.0, (axis_name, metric, r)
    print("  [5] same-ckpt: all bucket diffs = 0, p = 1 ✓")
    print("inference/size_dynamics smoke 통과")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = _parse()
        run(args.ckpt_signal, args.ckpt_logit, args.out_dir,
            K=args.K, seed=args.seed, batch_size=args.batch_size,
            max_batches=args.max_batches)
    else:
        _smoke()
