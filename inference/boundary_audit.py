"""Phase 3.0 H1 — boundary 자연 만족 검증.

비교: 001 signal (final.pt) vs 013 logit_native (final.pt)

가설:
  H1a coord_oob_rate(signal) > 0,  coord_oob_rate(logit) = 0  (sigmoid 보장)
  H1b 작은 K 일수록 signal 의 boundary 위반 / IoU 격차가 더 큼
  H1c excess_l1(logit) ≡ 0   (sigmoid 출력은 항상 (0, 1))

Metric (per-box, raw prediction = clamp(0,1) 우회):
  coord_oob   — {cx, cy, w, h} 중 어느 하나라도 [0, 1] 밖
  canvas_oob  — cx ± w/2 ∨ cy ± h/2 가 [0, 1] 밖 (박스 자체가 캔버스 밖)
  excess_l1   — Σ max(0, -val) + max(0, val-1)  (clamp 거리)
  iou_clamp   — clamp(0,1) 후 IoU
  iou_no_clamp — clamp 없이 IoU (참고)

K sweep: K ∈ {2, 4, 8, 16, 32}
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.trajectory import (
    logit_decode, logit_encode, sample_init_box,
    signal_decode, signal_encode,
)

from .compare import cache_val_batches, load_model
from .metrics import iou_xywh, paired_wilcoxon


K_VALUES = (2, 4, 8, 16, 32)

_ENCODERS = {
    "signal":       (signal_encode, signal_decode),
    "logit_native": (logit_encode,  logit_decode),
}


@torch.no_grad()
def ode_raw(model, image, init_box, K, model_name):
    """Forward ODE Euler in the model's native space without final clamp.

    Returns raw decoded box (B, n_q, 4). For signal this can fall outside
    [0, 1]^4 if s_final escapes [-3, 3]; for logit_native sigmoid(z) is
    always in (0, 1) so coord-wise OOB is impossible by construction.
    """
    if model_name not in _ENCODERS:
        raise ValueError(f"boundary_audit only supports signal/logit_native, "
                         f"got {model_name!r}")
    encode, decode = _ENCODERS[model_name]
    z = encode(init_box.to(image.device))
    B = z.shape[0]
    dt = 1.0 / K
    for k in range(K):
        t = torch.full((B,), k * dt, device=z.device)
        v = model.forward(z, t, image)
        z = z + dt * v
    return decode(z)              # NO clamp(0, 1)


@torch.no_grad()
def collect_predictions_raw(model, model_name, batches, K, seed,
                             n_queries=10, init_prior="default"):
    """Mirror compare.collect_predictions but returns *raw* predictions.

    The (init seed + batch_idx) pairing matches compare.py exactly — so
    H1 results are paired with the existing axis-1 paired comparison.
    """
    preds, inits, gts = [], [], []
    for batch_idx, batch in enumerate(batches):
        image = batch["image"]
        gt = batch["gt_boxes"]
        B = image.shape[0]

        torch.manual_seed(seed + batch_idx)
        ref = torch.empty(B, n_queries, 4, device=image.device)
        init_box = sample_init_box(ref, prior=init_prior).clamp(0, 1)

        pred_raw = ode_raw(model, image, init_box, K, model_name)
        preds.append(pred_raw.cpu())
        inits.append(init_box.cpu())
        gts.append(gt.cpu())
    return torch.cat(preds, 0), torch.cat(inits, 0), torch.cat(gts, 0)


def boundary_metrics(pred_raw: torch.Tensor, gt: torch.Tensor) -> dict:
    """All metrics per-box (..., shape) so they pair across models."""
    cx, cy, w, h = pred_raw.unbind(dim=-1)
    coord_oob = ((pred_raw < 0) | (pred_raw > 1)).any(dim=-1).float()
    left  = cx - w / 2
    right = cx + w / 2
    top   = cy - h / 2
    bot   = cy + h / 2
    canvas_oob = ((left < 0) | (right > 1) | (top < 0) | (bot > 1)).float()

    excess_lo = (-pred_raw).clamp_min(0)            # how far below 0
    excess_hi = (pred_raw - 1).clamp_min(0)         # how far above 1
    excess_l1 = (excess_lo + excess_hi).sum(dim=-1)

    iou_no_clamp = iou_xywh(pred_raw, gt)
    iou_clamp    = iou_xywh(pred_raw.clamp(0, 1), gt)

    return {
        "coord_oob":    coord_oob,
        "canvas_oob":   canvas_oob,
        "excess_l1":    excess_l1,
        "iou_no_clamp": iou_no_clamp,
        "iou_clamp":    iou_clamp,
    }


def _summarize(metrics: dict) -> dict:
    """Per-K row: summarize each metric tensor to mean (and std for IoU)."""
    return {
        "coord_oob_rate":   float(metrics["coord_oob"].mean()),
        "canvas_oob_rate":  float(metrics["canvas_oob"].mean()),
        "excess_l1_mean":   float(metrics["excess_l1"].mean()),
        "excess_l1_p99":    float(metrics["excess_l1"].quantile(0.99)),
        "iou_clamp_mean":   float(metrics["iou_clamp"].mean()),
        "iou_no_clamp_mean": float(metrics["iou_no_clamp"].mean()),
    }


def run(ckpt_signal, ckpt_logit, out_dir, *, k_values=K_VALUES, seed=0,
        batch_size=64, val_root="./data", max_batches=None, save_excess=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model_S, name_S = load_model(ckpt_signal, device)
    model_L, name_L = load_model(ckpt_logit, device)
    assert name_S == "signal", f"ckpt_signal must be signal model, got {name_S}"
    assert name_L == "logit_native", \
        f"ckpt_logit must be logit_native, got {name_L}"

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

    rows = []
    excess_dump = {}                # per-model, K=k_main only (for histogram)
    k_main = max(k_values)          # use highest K (closest to "real" inference)
    for K in k_values:
        for model, name in [(model_S, "signal"), (model_L, "logit_native")]:
            print(f"K={K}  model={name} ...")
            pred_raw, _, gt = collect_predictions_raw(
                model, name, batches, K=K, seed=seed,
            )
            m = boundary_metrics(pred_raw, gt)
            row = {"K": K, "model": name, **_summarize(m)}
            rows.append(row)
            print(f"  coord_oob={row['coord_oob_rate']:.4f}  "
                  f"canvas_oob={row['canvas_oob_rate']:.4f}  "
                  f"excess_l1={row['excess_l1_mean']:.5f}  "
                  f"iou_clamp={row['iou_clamp_mean']:.4f}")
            if save_excess and K == k_main:
                excess_dump[name] = m["excess_l1"].flatten().clone()

    # Wilcoxon paired (clamp IoU) per K — uses per-sample mean
    wilcoxon_rows = []
    for K in k_values:
        pred_S, _, gt = collect_predictions_raw(model_S, "signal", batches, K, seed)
        pred_L, _, _  = collect_predictions_raw(model_L, "logit_native", batches, K, seed)
        ic_S = iou_xywh(pred_S.clamp(0, 1), gt).mean(dim=1)
        ic_L = iou_xywh(pred_L.clamp(0, 1), gt).mean(dim=1)
        w = paired_wilcoxon(ic_L, ic_S)         # logit - signal
        wilcoxon_rows.append({
            "K": K,
            "iou_signal":      float(ic_S.mean()),
            "iou_logit":       float(ic_L.mean()),
            "diff_logit_minus_signal": float(w.median_diff),
            "p_value":         float(w.p_value),
        })
        print(f"K={K}  iou_diff={w.median_diff:+.4f}  p={w.p_value:.2g}")

    summary = {
        "ckpt_signal":  str(ckpt_signal),
        "ckpt_logit":   str(ckpt_logit),
        "seed":         seed,
        "k_values":     list(k_values),
        "n_total_boxes": int(n_total * 10),     # 10 queries per sample
        "rows":         rows,
        "wilcoxon":     wilcoxon_rows,
    }
    with open(out_dir / "boundary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "boundary_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    if save_excess:
        torch.save(excess_dump, out_dir / "excess_l1.pt")
    print(f"saved → {out_dir}")
    return summary


def _parse():
    p = argparse.ArgumentParser(description="Phase 3.0 H1 boundary audit")
    p.add_argument("--ckpt-signal", required=True)
    p.add_argument("--ckpt-logit",  required=True)
    p.add_argument("--out-dir", default="outputs/logit_strengths/boundary_audit")
    p.add_argument("--k-values", default="2,4,8,16,32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sanity / smoke
# ---------------------------------------------------------------------------

def _smoke():
    """Same-ckpt smoke + logit invariant + metric correctness."""
    # 1) boundary_metrics: shape + values on synthetic boxes
    pred = torch.tensor([
        [0.5, 0.5, 0.4, 0.4],     # in-bounds, in-canvas
        [0.5, 0.5, 0.8, 0.8],     # in-bounds, OUT of canvas (cx + 0.4 = 0.9 ok, cx - 0.4 = 0.1 ok ... actually in canvas)
        [0.5, 0.5, 1.2, 0.4],     # OUT of bounds (w > 1) AND out of canvas
        [-0.1, 0.5, 0.4, 0.4],    # OUT of bounds (cx < 0) AND out of canvas (left = -0.3)
        [0.7, 0.5, 0.8, 0.4],     # in-bounds, OUT of canvas (right = 1.1)
    ]).unsqueeze(0)               # (1, 5, 4)
    gt = torch.tensor([[0.5, 0.5, 0.4, 0.4]] * 5).unsqueeze(0)
    m = boundary_metrics(pred, gt)
    coord_expected = [0., 0., 1., 1., 0.]
    canvas_expected = [0., 0., 1., 1., 1.]
    assert m["coord_oob"].squeeze(0).tolist() == coord_expected, m["coord_oob"]
    assert m["canvas_oob"].squeeze(0).tolist() == canvas_expected, m["canvas_oob"]
    # excess_l1 of [-0.1, 0.5, 0.4, 0.4] = 0.1; of [0.5, 0.5, 1.2, 0.4] = 0.2
    ex = m["excess_l1"].squeeze(0)
    assert torch.allclose(ex, torch.tensor([0.0, 0.0, 0.2, 0.1, 0.0]), atol=1e-6), ex
    # iou_no_clamp non-NaN
    assert torch.isfinite(m["iou_no_clamp"]).all()
    print("  [1] synthetic metric correctness ✓")

    # 2) ode_raw rejects unsupported model_name
    try:
        ode_raw(None, None, None, 1, "chart_native")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported model")
    print("  [2] ode_raw guards unsupported model ✓")

    # 3) Same-ckpt run if available — coord_oob/canvas_oob/excess identical paired
    ckpt_s = Path("outputs/001_fullrun/ckpt/final.pt")
    ckpt_l = Path("outputs/013_logit_native_default/ckpt/final.pt")
    if not (ckpt_s.exists() and ckpt_l.exists()):
        print(f"  [skip] full smoke — ckpts missing")
        print("inference/boundary_audit smoke 통과 (synthetic only)")
        return

    out = run(ckpt_s, ckpt_l, "/tmp/boundary_audit_smoke",
              k_values=(2, 4), batch_size=8, max_batches=2,
              save_excess=True)
    # H1c: logit_native must have coord_oob = 0 and excess_l1 = 0
    for row in out["rows"]:
        if row["model"] == "logit_native":
            assert row["coord_oob_rate"] == 0.0, row
            assert row["excess_l1_mean"] == 0.0, row
    print("  [3] logit_native: coord_oob = excess_l1 = 0 ✓ (sigmoid 보장)")

    # logit canvas_oob CAN be > 0 (cx + w/2 may exceed 1 even when cx, w ∈ (0,1))
    print("inference/boundary_audit smoke 통과")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = _parse()
        ks = tuple(int(x) for x in args.k_values.split(","))
        run(args.ckpt_signal, args.ckpt_logit, args.out_dir,
            k_values=ks, seed=args.seed, batch_size=args.batch_size,
            max_batches=args.max_batches)
    else:
        _smoke()
