import numpy as np
import cv2
import torch

_GT_COLOR = (0, 255, 0)    # 녹색 — GT box
_INIT_COLOR = (0, 0, 255)  # 빨강 — init box


def draw_sample(sample: dict, H: int = 224, W: int = 224) -> np.ndarray:
    """sample dict를 받아 gt/init box overlay 이미지 반환. Returns (H, W, 3) uint8."""
    image = sample["image"]
    gt_boxes = sample["gt_boxes"]
    init_boxes = sample["init_boxes"]

    if isinstance(image, torch.Tensor):
        img = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    else:
        img = (image.transpose(1, 2, 0) * 255).astype(np.uint8).copy()

    def draw_box(img, box, color, text=""):
        if isinstance(box, torch.Tensor):
            box = box.numpy()
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if text:
            cv2.putText(img, text, (x1 + 1, max(y1 - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    for i in range(10):
        draw_box(img, gt_boxes[i], _GT_COLOR, str(i))
        draw_box(img, init_boxes[i], _INIT_COLOR)

    return img


def show_sample(sample: dict, title: str = "GT(green) Init(red)"):
    img = draw_sample(sample)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset.mnist_box_dataset import MNISTBoxDataset

    ds = MNISTBoxDataset(split="train", root="./data")
    sample = ds[0]
    show_sample(sample)
    print("visualize sanity check 통과")
