import numpy as np

try:
    from .box_utils import boxes_overlap, box_in_canvas
except ImportError:
    from box_utils import boxes_overlap, box_in_canvas

_SIZE_MIN = 14 / 224
_SIZE_MAX = 56 / 224

# Wide-scale variant for plans/riem_strength.md exp 005:
# 7..120 px on 224 px canvas → side ratio 17×, area ratio ~290×.
_WIDE_SIZE_MIN = 7 / 224
_WIDE_SIZE_MAX = 120 / 224


def sample_gt_boxes(max_tries: int = 200, wide: bool = False) -> np.ndarray:
    """10개의 non-overlap gt box 샘플링. Returns (10, 4) normalized cx,cy,w,h.

    wide=True: size 분포 확장 (7~120 px / 224, 17× side variation).
               Riemannian의 multiplicative dynamics 강점 노출용.
    """
    smin = _WIDE_SIZE_MIN if wide else _SIZE_MIN
    smax = _WIDE_SIZE_MAX if wide else _SIZE_MAX
    # Log-uniform sampling so small/medium/large 모두 적절히 분포 (wide일 때 특히)
    if wide:
        log_sizes = np.random.uniform(np.log(smin), np.log(smax), size=10)
        sizes = np.exp(log_sizes)
    else:
        sizes = np.random.uniform(smin, smax, size=10)
    for _ in range(max_tries):
        # 큰 사이즈부터 배치하면 packing 성공률 ↑
        order = np.argsort(-sizes)
        boxes = [None] * 10
        success = True
        for i in order:
            s = sizes[i]
            placed = False
            for _ in range(400):
                cx = np.random.uniform(s / 2, 1 - s / 2)
                cy = np.random.uniform(s / 2, 1 - s / 2)
                b = np.array([cx, cy, s, s])
                if not box_in_canvas(b):
                    continue
                if all(boxes[j] is None or not boxes_overlap(b, boxes[j])
                       for j in range(10)):
                    boxes[i] = b
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return np.stack(boxes)  # (10, 4)
        # re-sample sizes for next try (avoid stuck on impossible config)
        if wide:
            log_sizes = np.random.uniform(np.log(smin), np.log(smax), size=10)
            sizes = np.exp(log_sizes)
        else:
            sizes = np.random.uniform(smin, smax, size=10)
    raise RuntimeError(f"gt box 샘플링 실패 (wide={wide}): max_tries 초과")


def sample_init_signal() -> np.ndarray:
    """z ~ N(0, I_4), clip to [-3, 3]. Returns (10, 4)."""
    return np.clip(np.random.randn(10, 4), -3, 3)


if __name__ == "__main__":
    np.random.seed(0)
    # Default
    gt_boxes = sample_gt_boxes()
    assert gt_boxes.shape == (10, 4)
    for i in range(10):
        assert box_in_canvas(gt_boxes[i])
        for j in range(i + 1, 10):
            assert not boxes_overlap(gt_boxes[i], gt_boxes[j])

    # Wide variant
    np.random.seed(1)
    gt_wide = sample_gt_boxes(wide=True)
    assert gt_wide.shape == (10, 4)
    for i in range(10):
        assert box_in_canvas(gt_wide[i]), f"wide box {i} canvas 밖"
        for j in range(i + 1, 10):
            assert not boxes_overlap(gt_wide[i], gt_wide[j]), f"wide box {i},{j} 겹침"
    print(f"default size range: w [{gt_boxes[:,2].min():.3f}, {gt_boxes[:,2].max():.3f}]")
    print(f"wide size range:    w [{gt_wide[:,2].min():.3f}, {gt_wide[:,2].max():.3f}]")
    print(f"wide size ratio:    max/min = {gt_wide[:,2].max() / gt_wide[:,2].min():.1f}×")

    init_signals = sample_init_signal()
    assert init_signals.shape == (10, 4)
    assert init_signals.min() >= -3 and init_signals.max() <= 3
    print("sampler sanity check 통과")
