"""Phase 2 axis 1 + 2 — mean IoU + size-stratified IoU between two checkpoints.

Same val_loader, **same init seed** for both models → paired comparison.
Output: summary.json, stratified_iou.csv. (visualize.py later turns these into figures.)
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.flow_chart import ChartFlowModel
from model.flow_chart_native import ChartNativeFlowModel
from model.flow_signal import SignalFlowModel

from .metrics import (
    RATIO_BUCKETS,
    SIZE_BUCKETS,
    center_error,
    chart_mse,
    iou_xywh,
    paired_wilcoxon,
    signal_mse,
    size_change_ratio,
    size_error,
    stratify_by_ratio,
    stratify_by_size,
)
from model.trajectory import sample_init_box, signal_decode


from model.flow_chart_boxloss import ChartBoxLossFlowModel  # noqa: E402
from model.flow_chart_linear import ChartLinearFlowModel  # noqa: E402
from model.flow_hybrid import HybridFlowModel  # noqa: E402
from model.flow_local import LocalChartFlowModel  # noqa: E402
from model.flow_logit_native import LogitNativeFlowModel  # noqa: E402

_MODEL_CLS = {
    "signal":         SignalFlowModel,
    "chart":          ChartFlowModel,
    "chart_native":   ChartNativeFlowModel,
    "chart_linear":   ChartLinearFlowModel,
    "hybrid":         HybridFlowModel,
    "chart_boxloss":  ChartBoxLossFlowModel,
    "local":          LocalChartFlowModel,
    "logit_native":   LogitNativeFlowModel,
}
_BUCKET_NAMES = ("small", "medium", "large")


def load_model(ckpt_path, device):
    """Load a flow model from a final.pt / step_NNNN.pt ckpt."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model_name = cfg.get("model", "signal")
    cls = _MODEL_CLS[model_name]
    model = cls(
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        n_queries=cfg["n_queries"],
        encoder_pretrained=cfg["encoder_pretrained"],
        encoder_freeze=cfg["encoder_freeze"],
    ).to(device).eval()
    model.load_state_dict(state["model"])
    return model, model_name


def cache_val_batches(val_loader, device, sample_seed=12345):
    """Materialize val_loader once into a fixed list of batches.

    MNISTBoxDataset.__getitem__ is non-deterministic per call, so paired
    comparison REQUIRES the same (image, gt) pairs for both models.
    sample_seed pins the global RNG before iteration so the cache is
    reproducible across runs.
    """
    g_state = torch.get_rng_state()
    torch.manual_seed(sample_seed)
    batches = []
    for batch in val_loader:
        batches.append({
            "image":    batch["image"].to(device, non_blocking=True),
            "gt_boxes": batch["gt_boxes"].to(device, non_blocking=True),
        })
    torch.set_rng_state(g_state)
    return batches


@torch.no_grad()
def collect_predictions(model, batches, K, seed, n_queries=10,
                         init_prior: str = "default"):
    """Returns (pred_boxes, init_boxes, gt_boxes) each of shape (N, 10, 4).

    Init box is sampled in **box space** (per init_prior) and passed
    to model.sample(init_box=...) — guarantees both models share the SAME b_0
    when called with the same (seed + batch_idx).
    """
    preds, inits, gts = [], [], []
    for batch_idx, batch in enumerate(batches):
        image = batch["image"]
        gt = batch["gt_boxes"]
        B = image.shape[0]

        torch.manual_seed(seed + batch_idx)
        ref = torch.empty(B, n_queries, 4, device=image.device)
        init_box = sample_init_box(ref, prior=init_prior).clamp(0, 1)

        pred, _ = model.sample(image, K=K, init_box=init_box)

        preds.append(pred.cpu())
        inits.append(init_box.cpu())
        gts.append(gt.cpu())
    return torch.cat(preds, 0), torch.cat(inits, 0), torch.cat(gts, 0)


def collect_iou(model, batches, K, seed):
    """Backwards-compatible: returns (iou (N, 10), gt (N, 10, 4))."""
    pred, _, gt = collect_predictions(model, batches, K, seed)
    return iou_xywh(pred, gt), gt


def compare(ckpt_a, ckpt_b, out_dir, *, K=10, seed=0, batch_size=32,
            val_root="./data", max_batches=None, device=None, wide=False,
            init_prior: str = "default"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_a, name_a = load_model(ckpt_a, device)
    model_b, name_b = load_model(ckpt_b, device)

    val_ds = MNISTBoxDataset(split="val", root=val_root, wide=wide)
    if max_batches is not None:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, list(range(max_batches * batch_size)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"compare: A={name_a} ({ckpt_a}) | B={name_b} ({ckpt_b}) | K={K} "
          f"seed={seed} init_prior={init_prior}")
    print("caching val batches (paired) ...")
    batches = cache_val_batches(val_loader, device)
    print(f"  cached {len(batches)} batches × {batches[0]['image'].shape[0]} = "
          f"{sum(b['image'].shape[0] for b in batches)} samples")
    pred_a, init_a, gt = collect_predictions(model_a, batches, K, seed,
                                             init_prior=init_prior)
    pred_b, init_b, _  = collect_predictions(model_b, batches, K, seed,
                                             init_prior=init_prior)
    assert torch.allclose(init_a, init_b), "paired init mismatch"

    iou_a = iou_xywh(pred_a, gt)
    iou_b = iou_xywh(pred_b, gt)
    cerr_a = center_error(pred_a, gt); cerr_b = center_error(pred_b, gt)
    serr_a = size_error(pred_a, gt);   serr_b = size_error(pred_b, gt)

    # axis 1 — per-sample mean (across 10 queries)
    sample_a = iou_a.mean(dim=1)
    sample_b = iou_b.mean(dim=1)
    overall_w = paired_wilcoxon(sample_a, sample_b)
    overall = {
        "mean_iou_a": float(sample_a.mean()),
        "mean_iou_b": float(sample_b.mean()),
        "std_iou_a":  float(sample_a.std()),
        "std_iou_b":  float(sample_b.std()),
        "wilcoxon":   asdict(overall_w),
    }

    # axis 1b — center / size error decomposition (per-box)
    decomposition = {
        "center_err": {
            "mean_a": float(cerr_a.mean()), "mean_b": float(cerr_b.mean()),
            "median_a": float(cerr_a.median()), "median_b": float(cerr_b.median()),
            "wilcoxon": asdict(paired_wilcoxon(cerr_a.flatten(), cerr_b.flatten())),
        },
        "size_err": {
            "mean_a": float(serr_a.mean()), "mean_b": float(serr_b.mean()),
            "median_a": float(serr_a.median()), "median_b": float(serr_b.median()),
            "wilcoxon": asdict(paired_wilcoxon(serr_a.flatten(), serr_b.flatten())),
        },
    }

    # axis 1c — alternative metrics (chart-MSE / signal-MSE)
    #   Tests whether each model wins on its NATIVE training metric:
    #     chart_native learns to minimize chart-MSE → should win that
    #     signal model  learns to minimize signal-MSE → should win that
    cmse_a = chart_mse(pred_a, gt); cmse_b = chart_mse(pred_b, gt)
    smse_a = signal_mse(pred_a, gt); smse_b = signal_mse(pred_b, gt)
    alt_metrics = {
        "chart_mse_(log-space)": {
            "mean_a": float(cmse_a.mean()), "mean_b": float(cmse_b.mean()),
            "median_a": float(cmse_a.median()), "median_b": float(cmse_b.median()),
            "wilcoxon": asdict(paired_wilcoxon(cmse_a.flatten(), cmse_b.flatten())),
        },
        "signal_mse_(affine-box)": {
            "mean_a": float(smse_a.mean()), "mean_b": float(smse_b.mean()),
            "median_a": float(smse_a.median()), "median_b": float(smse_b.median()),
            "wilcoxon": asdict(paired_wilcoxon(smse_a.flatten(), smse_b.flatten())),
        },
    }

    # axis 2 — GT-size-stratified (per-box)
    bucket = stratify_by_size(gt, thresholds=SIZE_BUCKETS)
    stratified = _stratify_compare(bucket, _BUCKET_NAMES, iou_a, iou_b)

    # axis 2b — size-CHANGE-ratio stratified (per-box)
    rbucket = stratify_by_ratio(init_a, gt, thresholds=RATIO_BUCKETS)
    ratio_strat = _stratify_compare(rbucket, _BUCKET_NAMES, iou_a, iou_b)
    ratio_vals = size_change_ratio(init_a, gt).flatten()

    summary = {
        "ckpt_a": str(ckpt_a), "ckpt_b": str(ckpt_b),
        "model_a": name_a, "model_b": name_b,
        "K": K, "seed": seed,
        "n_val_samples": int(sample_a.shape[0]),
        "n_total_boxes": int(iou_a.numel()),
        "size_buckets": list(SIZE_BUCKETS),
        "ratio_buckets": list(RATIO_BUCKETS),
        "axis_1_overall": overall,
        "axis_1b_decomposition": decomposition,
        "axis_1c_alt_metrics": alt_metrics,
        "axis_2_size_stratified": stratified,
        "axis_2b_ratio_stratified": ratio_strat,
        "ratio_distribution": {
            "median": float(ratio_vals.median()),
            "p90":    float(ratio_vals.quantile(0.9)),
            "p99":    float(ratio_vals.quantile(0.99)),
            "max":    float(ratio_vals.max()),
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "stratified_iou.csv", "w") as f:
        f.write("bucket,n_boxes,mean_iou_a,mean_iou_b,median_diff,p_value\n")
        for name in _BUCKET_NAMES:
            if name not in stratified:
                continue
            s = stratified[name]
            w = s["wilcoxon"]
            f.write(f"{name},{s['n_boxes']},{s['mean_iou_a']:.4f},"
                    f"{s['mean_iou_b']:.4f},{w['median_diff']:.4f},"
                    f"{w['p_value']:.4g}\n")

    print("=== axis 1 overall ===")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print("=== axis 1b decomposition (center / size error) ===")
    print(json.dumps(decomposition, indent=2, ensure_ascii=False))
    print("=== axis 1c alt metrics (chart-MSE / signal-MSE) ===")
    print(json.dumps(alt_metrics, indent=2, ensure_ascii=False))
    print("=== axis 2b ratio-stratified ===")
    print(json.dumps(ratio_strat, indent=2, ensure_ascii=False))
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"stratified: {out_dir / 'stratified_iou.csv'}")
    return summary


def _stratify_compare(bucket, names, iou_a, iou_b):
    out = {}
    for k, name in enumerate(names):
        mask = bucket == k
        n = int(mask.sum())
        if n == 0:
            continue
        ia = iou_a[mask]; ib = iou_b[mask]
        out[name] = {
            "n_boxes":    n,
            "mean_iou_a": float(ia.mean()),
            "mean_iou_b": float(ib.mean()),
            "wilcoxon":   asdict(paired_wilcoxon(ia, ib)),
        }
    return out


def _parse():
    p = argparse.ArgumentParser(description="Phase 2 axis 1+2 comparison")
    p.add_argument("--ckpt-a", required=True, help="Euclidean ckpt path")
    p.add_argument("--ckpt-b", required=True, help="Riemannian ckpt path")
    p.add_argument("--out-dir", default="outputs/comparison")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap val batches (smoke); None = full val split")
    p.add_argument("--wide", action="store_true",
                   help="Use wide-scale val dataset (for 005, 006 ckpts)")
    p.add_argument("--init-prior", choices=["default", "small_size"],
                   default="default",
                   help="b_0 prior for paired comparison (must match training)")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = _parse()
        compare(args.ckpt_a, args.ckpt_b, args.out_dir,
                K=args.K, seed=args.seed, batch_size=args.batch_size,
                max_batches=args.max_batches, wide=args.wide,
                init_prior=args.init_prior)
    else:
        # Smoke: same ckpt twice → IoU diff = 0 → Wilcoxon p=1, both means equal.
        from pathlib import Path
        ckpt = "outputs/001_fullrun/ckpt/final.pt"
        if not Path(ckpt).exists():
            print(f"sanity skipped — {ckpt} not found")
            raise SystemExit
        s = compare(ckpt, ckpt, "/tmp/compare_smoke",
                    K=4, batch_size=8, max_batches=2)
        ov = s["axis_1_overall"]
        # same ckpt + same seed + deterministic CUDA → bit-identical
        assert ov["mean_iou_a"] == ov["mean_iou_b"], \
            f"same ckpt: {ov['mean_iou_a']} vs {ov['mean_iou_b']}"
        assert ov["wilcoxon"]["p_value"] == 1.0, ov["wilcoxon"]
        print("inference/compare smoke 통과 (same-ckpt = identical)")
