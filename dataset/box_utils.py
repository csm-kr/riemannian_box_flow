import numpy as np


def signal_to_box(s: np.ndarray) -> np.ndarray:
    """[-3, 3]^4 -> [0, 1]^4"""
    return (s / 3 + 1) / 2


def box_to_signal(b: np.ndarray) -> np.ndarray:
    """[0, 1]^4 -> [-3, 3]^4"""
    return (2 * b - 1) * 3


def boxes_overlap(b1: np.ndarray, b2: np.ndarray) -> bool:
    """두 box (cx, cy, w, h) 가 겹치는지 확인."""
    def to_xyxy(b):
        cx, cy, w, h = b
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    x1, y1, x2, y2 = to_xyxy(b1)
    x3, y3, x4, y4 = to_xyxy(b2)
    return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)


def box_in_canvas(b: np.ndarray) -> bool:
    """box (cx, cy, w, h) 가 [0,1] canvas 안에 완전히 포함되는지 확인."""
    cx, cy, w, h = b
    return cx - w / 2 >= 0 and cx + w / 2 <= 1 and cy - h / 2 >= 0 and cy + h / 2 <= 1


def norm_to_pixel(b: np.ndarray, H: int = 224, W: int = 224) -> np.ndarray:
    """Normalized (cx, cy, w, h) -> pixel (cx, cy, w, h)."""
    cx, cy, w, h = b
    return np.array([cx * W, cy * H, w * W, h * H])


if __name__ == "__main__":
    # round-trip 변환 확인
    b = np.array([0.5, 0.5, 0.2, 0.2])
    s = box_to_signal(b)
    b2 = signal_to_box(s)
    assert np.allclose(b, b2), f"round-trip 실패: {b} -> {s} -> {b2}"
    print(f"round-trip OK: {b} -> {s} -> {b2}")

    # overlap 체크
    b1 = np.array([0.3, 0.3, 0.2, 0.2])
    b2 = np.array([0.7, 0.7, 0.2, 0.2])
    b3 = np.array([0.35, 0.35, 0.2, 0.2])
    assert not boxes_overlap(b1, b2), "겹치지 않아야 함"
    assert boxes_overlap(b1, b3), "겹쳐야 함"
    print("overlap check OK")

    # bound 체크
    assert box_in_canvas(np.array([0.5, 0.5, 0.2, 0.2]))
    assert not box_in_canvas(np.array([0.05, 0.5, 0.2, 0.2]))
    print("bound check OK")

    print("box_utils sanity check 통과")
