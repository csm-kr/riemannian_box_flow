import numpy as np

try:
    from .box_utils import boxes_overlap, box_in_canvas
except ImportError:
    from box_utils import boxes_overlap, box_in_canvas

_SIZE_MIN = 14 / 224
_SIZE_MAX = 56 / 224


def sample_gt_boxes(max_tries: int = 200) -> np.ndarray:
    """10개의 non-overlap gt box 샘플링. Returns (10, 4) normalized cx,cy,w,h."""
    for _ in range(max_tries):
        boxes = []
        sizes = np.random.uniform(_SIZE_MIN, _SIZE_MAX, size=10)
        success = True
        for i in range(10):
            s = sizes[i]
            placed = False
            for _ in range(200):
                cx = np.random.uniform(s / 2, 1 - s / 2)
                cy = np.random.uniform(s / 2, 1 - s / 2)
                b = np.array([cx, cy, s, s])
                if not box_in_canvas(b):
                    continue
                if all(not boxes_overlap(b, existing) for existing in boxes):
                    boxes.append(b)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return np.stack(boxes)  # (10, 4)
    raise RuntimeError("gt box 샘플링 실패: max_tries 초과")


def sample_init_signal() -> np.ndarray:
    """z ~ N(0, I_4), clip to [-3, 3]. Returns (10, 4)."""
    return np.clip(np.random.randn(10, 4), -3, 3)


if __name__ == "__main__":
    gt_boxes = sample_gt_boxes()
    assert gt_boxes.shape == (10, 4), f"shape 오류: {gt_boxes.shape}"

    for i in range(10):
        assert box_in_canvas(gt_boxes[i]), f"box {i} canvas 밖"
        for j in range(i + 1, 10):
            assert not boxes_overlap(gt_boxes[i], gt_boxes[j]), f"box {i},{j} 겹침"

    init_signals = sample_init_signal()
    assert init_signals.shape == (10, 4)
    assert init_signals.min() >= -3 and init_signals.max() <= 3

    print(f"gt_boxes shape: {gt_boxes.shape}")
    print(f"gt_boxes range: cx [{gt_boxes[:,0].min():.3f}, {gt_boxes[:,0].max():.3f}]")
    print(f"init_signals range: [{init_signals.min():.3f}, {init_signals.max():.3f}]")
    print("sampler sanity check 통과")
