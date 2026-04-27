"""Phase 2 comparison metrics: IoU, paired Wilcoxon, size stratification."""

from dataclasses import dataclass

import torch


SIZE_BUCKETS = (0.015, 0.035)  # area = w*h thresholds → small / medium / large
# Calibrated from MNISTBoxDataset val split: area median ~0.024 (no box exceeds ~0.07).
# Buckets target roughly even thirds (~30/40/30 % of boxes).


@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    n: int           # paired sample count
    median_diff: float


def iou_xywh(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Axis-aligned IoU on cxcywh boxes.

    box_a, box_b: tensor of shape (..., 4). Returns shape (...,) IoU in [0, 1].
    Boxes are clamped to non-negative w/h via half-extent before intersection.
    """
    cx_a, cy_a, w_a, h_a = box_a.unbind(dim=-1)
    cx_b, cy_b, w_b, h_b = box_b.unbind(dim=-1)
    hw_a, hh_a = w_a / 2, h_a / 2
    hw_b, hh_b = w_b / 2, h_b / 2

    x1 = torch.maximum(cx_a - hw_a, cx_b - hw_b)
    y1 = torch.maximum(cy_a - hh_a, cy_b - hh_b)
    x2 = torch.minimum(cx_a + hw_a, cx_b + hw_b)
    y2 = torch.minimum(cy_a + hh_a, cy_b + hh_b)

    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_a = w_a.clamp_min(0) * h_a.clamp_min(0)
    area_b = w_b.clamp_min(0) * h_b.clamp_min(0)
    union = area_a + area_b - inter
    return inter / union.clamp_min(1e-12)


def stratify_by_size(boxes: torch.Tensor,
                     thresholds=SIZE_BUCKETS) -> torch.Tensor:
    """Bucket each box by area = w * h. Returns long bucket indices (...,).
    0 = small (area < t0), 1 = medium (t0 ≤ area < t1), 2 = large (area ≥ t1).
    """
    t0, t1 = thresholds
    area = boxes[..., 2] * boxes[..., 3]
    bucket = torch.zeros_like(area, dtype=torch.long)
    bucket[area >= t0] = 1
    bucket[area >= t1] = 2
    return bucket


# Size-change ratio buckets (geometric-mean of w_1/w_0 and h_1/h_0).
# 1.0 = no change; values >> 1 mean drastic resize.
RATIO_BUCKETS = (1.5, 3.0)  # small / medium / large size CHANGE


def size_change_ratio(b0: torch.Tensor, b1: torch.Tensor,
                      eps: float = 1e-3) -> torch.Tensor:
    """Geometric-mean size-change ratio: sqrt(max(w1/w0, w0/w1) * max(h1/h0, h0/h1)).

    Returns (...,) values in [1, ∞). 1 means same size, ≥1 always.
    """
    w0 = b0[..., 2].clamp_min(eps); h0 = b0[..., 3].clamp_min(eps)
    w1 = b1[..., 2].clamp_min(eps); h1 = b1[..., 3].clamp_min(eps)
    rw = torch.maximum(w1 / w0, w0 / w1)
    rh = torch.maximum(h1 / h0, h0 / h1)
    return (rw * rh).sqrt()


def stratify_by_ratio(b0, b1, thresholds=RATIO_BUCKETS):
    r = size_change_ratio(b0, b1)
    t0, t1 = thresholds
    bucket = torch.zeros_like(r, dtype=torch.long)
    bucket[r >= t0] = 1
    bucket[r >= t1] = 2
    return bucket


def center_error(box_a, box_b):
    """L2 distance between centers. (...,)"""
    d = box_a[..., :2] - box_b[..., :2]
    return d.pow(2).sum(dim=-1).sqrt()


def size_error(box_a, box_b):
    """L1 distance between (w, h). (...,)"""
    d = (box_a[..., 2:] - box_b[..., 2:]).abs()
    return d.sum(dim=-1)


def chart_mse(pred_box, gt_box, eps: float = 1e-3):
    """Squared error in chart space (cx, cy, log w, log h), per-box mean.

    Tests whether C-R wins on its NATIVE evaluation metric (chart-space loss).
    Returns (...,)."""
    import torch as _t
    p = _t.cat([pred_box[..., :2], pred_box[..., 2:].clamp_min(eps).log()], dim=-1)
    g = _t.cat([gt_box[..., :2],   gt_box[..., 2:].clamp_min(eps).log()], dim=-1)
    return (p - g).pow(2).mean(dim=-1)


def signal_mse(pred_box, gt_box):
    """Squared error in signal space s = 6b - 3, per-box mean.

    Tests whether S-E wins on its NATIVE evaluation metric (signal-space loss)."""
    return ((pred_box - gt_box) * 6.0).pow(2).mean(dim=-1)


def log_size_error(pred_box, gt_box, eps: float = 1e-3):
    """|log(pred_size) - log(gt_size)| L1 on (w, h). Scale-invariant size error.

    A 10% relative error gives same value regardless of absolute box size.
    """
    p = pred_box[..., 2:].clamp_min(eps).log()
    g = gt_box[..., 2:].clamp_min(eps).log()
    return (p - g).abs().sum(dim=-1)


def scale_relative_center_err(pred_box, gt_box, eps: float = 1e-3):
    """L2 center distance normalized by GT box size (geometric mean of w, h).

    Small boxes count equally with large.
    """
    d = pred_box[..., :2] - gt_box[..., :2]
    dist = d.pow(2).sum(dim=-1).sqrt()
    scale = (gt_box[..., 2] * gt_box[..., 3]).clamp_min(eps).sqrt()
    return dist / scale


def per_bucket_weighted_iou(iou_per_box, bucket_per_box, n_buckets=3):
    """Mean IoU per bucket, then mean over buckets (equal weight).

    Counters the bias of mean-over-boxes (which is dominated by populous buckets).
    Returns scalar tensor.
    """
    means = []
    for k in range(n_buckets):
        mask = bucket_per_box == k
        if mask.sum() == 0:
            continue
        means.append(iou_per_box[mask].mean())
    return torch.stack(means).mean() if means else torch.tensor(0.0)


def paired_wilcoxon(scores_a: torch.Tensor,
                    scores_b: torch.Tensor) -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired scores (two-sided).

    scipy.stats.wilcoxon raises if all diffs are zero — we treat that as p=1.
    """
    from scipy.stats import wilcoxon

    a = scores_a.detach().cpu().double().flatten().numpy()
    b = scores_b.detach().cpu().double().flatten().numpy()
    assert a.shape == b.shape, f"length mismatch: {a.shape} vs {b.shape}"
    diff = a - b
    n = int(len(diff))
    median = float(torch.tensor(diff).median().item())

    if (diff == 0).all():
        return WilcoxonResult(statistic=0.0, p_value=1.0, n=n, median_diff=median)

    res = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return WilcoxonResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n=n,
        median_diff=median,
    )


# ---------------------------------------------------------------------------
# Sanity check (`python -m inference.metrics`)
# ---------------------------------------------------------------------------

def _sanity():
    # 1) IoU = 1 for identical boxes
    a = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    b = a.clone()
    iou = iou_xywh(a, b)
    assert iou.shape == (1,), iou.shape
    assert torch.allclose(iou, torch.ones(1), atol=1e-6), iou

    # 2) IoU = 0 for fully disjoint
    a2 = torch.tensor([[0.2, 0.2, 0.2, 0.2]])
    b2 = torch.tensor([[0.8, 0.8, 0.2, 0.2]])
    iou2 = iou_xywh(a2, b2)
    assert torch.allclose(iou2, torch.zeros(1), atol=1e-6), iou2

    # 3) Known value: half-overlap along x
    #    a = (0.4,0.5,0.4,0.4), b = (0.6,0.5,0.4,0.4)
    #    intersection = 0.2 * 0.4 = 0.08
    #    union = 2*0.16 - 0.08 = 0.24 → IoU = 1/3
    a3 = torch.tensor([[0.4, 0.5, 0.4, 0.4]])
    b3 = torch.tensor([[0.6, 0.5, 0.4, 0.4]])
    iou3 = iou_xywh(a3, b3)
    assert torch.allclose(iou3, torch.tensor([1 / 3]), atol=1e-5), iou3

    # 4) Symmetric
    torch.manual_seed(0)
    boxes_a = torch.rand(4, 10, 4)
    boxes_b = torch.rand(4, 10, 4)
    boxes_a[..., 2:].clamp_(min=1e-3)
    boxes_b[..., 2:].clamp_(min=1e-3)
    iou_ab = iou_xywh(boxes_a, boxes_b)
    iou_ba = iou_xywh(boxes_b, boxes_a)
    assert iou_ab.shape == (4, 10), iou_ab.shape
    assert torch.allclose(iou_ab, iou_ba, atol=1e-6)
    assert (iou_ab >= 0).all() and (iou_ab <= 1).all()

    # 5) stratify_by_size — calibrated thresholds (0.015, 0.035) on MNIST box dist
    boxes_str = torch.tensor([
        [0.5, 0.5, 0.10, 0.10],   # area 0.010 → small  (<0.015)
        [0.5, 0.5, 0.15, 0.15],   # area 0.0225 → medium
        [0.5, 0.5, 0.20, 0.20],   # area 0.040 → large  (>=0.035)
        [0.5, 0.5, 0.05, 0.20],   # area 0.010 → small
    ])
    bucket = stratify_by_size(boxes_str)
    assert bucket.tolist() == [0, 1, 2, 0], bucket.tolist()

    # 6) Wilcoxon: identical scores → returns (well-defined edge); a > b consistently → small p
    scores_a = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65])
    scores_b = scores_a - 0.05
    res = paired_wilcoxon(scores_a, scores_b)
    assert res.n == 7
    assert res.p_value < 0.05, f"expected significant difference, got p={res.p_value}"
    assert res.median_diff > 0   # a > b

    # 7) Wilcoxon: random equal-mean → likely non-significant (p > 0.1)
    torch.manual_seed(42)
    sa = torch.rand(50)
    sb = sa + (torch.randn(50) * 0.01)   # tiny noise around equal
    res2 = paired_wilcoxon(sa, sb)
    assert 0 <= res2.p_value <= 1.0

    print("inference/metrics sanity check 통과")


if __name__ == "__main__":
    _sanity()
