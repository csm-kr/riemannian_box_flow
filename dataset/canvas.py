import numpy as np
import cv2


def compose_canvas(digits: list[np.ndarray], gt_boxes: np.ndarray, H: int = 224, W: int = 224) -> np.ndarray:
    """
    10개의 digit을 gt_boxes 위치에 paste한 canvas 반환.
    digits: list of (h, w) uint8
    gt_boxes: (10, 4) normalized cx, cy, w, h
    Returns: (H, W, 3) uint8
    """
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for digit, box in zip(digits, gt_boxes):
        cx, cy, w, h = box
        pw = max(1, int(round(w * W)))
        ph = max(1, int(round(h * H)))
        pcx = int(round(cx * W))
        pcy = int(round(cy * H))

        resized = cv2.resize(digit, (pw, ph), interpolation=cv2.INTER_LINEAR)
        resized_3ch = np.stack([resized] * 3, axis=-1)

        x1, y1 = pcx - pw // 2, pcy - ph // 2
        x2, y2 = x1 + pw, y1 + ph

        # canvas 경계 클램프
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(W, x2), min(H, y2)
        dx1, dy1 = cx1 - x1, cy1 - y1
        dx2, dy2 = dx1 + (cx2 - cx1), dy1 + (cy2 - cy1)

        canvas[cy1:cy2, cx1:cx2] = resized_3ch[dy1:dy2, dx1:dx2]

    return canvas


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset.sampler import sample_gt_boxes

    gt_boxes = sample_gt_boxes()
    digits = [np.random.randint(0, 255, (28, 28), dtype=np.uint8) for _ in range(10)]
    canvas = compose_canvas(digits, gt_boxes)

    assert canvas.shape == (224, 224, 3), f"shape 오류: {canvas.shape}"
    print(f"canvas shape: {canvas.shape}")

    cv2.imshow("canvas test", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("canvas sanity check 통과")
