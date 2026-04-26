"""ODE trajectory → frames → GIF."""

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import imageio
import numpy as np
import torch


# 10 distinct BGR colors per box index (cv2 uses BGR)
BOX_COLORS = [
    (64,  64, 255),   # 0  red
    (64, 128, 255),   # 1  orange
    (64, 192, 255),   # 2  amber
    (64, 255, 192),   # 3  teal
    (64, 255,  64),   # 4  green
    (192, 255, 64),   # 5  lime
    (255, 192, 64),   # 6  sky
    (255,  64, 64),   # 7  blue
    (255,  64, 192),  # 8  magenta
    (192,  64, 255),  # 9  pink
]


def _to_bgr_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """(3, H, W) tensor [0,1] or (H, W, 3) ndarray → (H, W, 3) BGR uint8 copy."""
    if isinstance(image, torch.Tensor):
        arr = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return arr.copy()
    if image.ndim == 3 and image.shape[0] == 3:  # CHW
        if image.dtype == np.uint8:
            return image.transpose(1, 2, 0).copy()
        return (image.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
    return image.copy()


def _draw_box(
    canvas: np.ndarray,
    box: torch.Tensor | np.ndarray,
    color: tuple[int, int, int],
    *,
    dashed: bool = False,
    label: str | None = None,
    H: int = 224,
    W: int = 224,
) -> None:
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    cx, cy, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    x1 = int(round((cx - w / 2) * W))
    y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W))
    y2 = int(round((cy + h / 2) * H))

    if dashed:
        step, on = 6, 3
        for x in range(x1, x2, step):
            cv2.line(canvas, (x, y1), (min(x + on, x2), y1), color, 1)
            cv2.line(canvas, (x, y2), (min(x + on, x2), y2), color, 1)
        for y in range(y1, y2, step):
            cv2.line(canvas, (x1, y), (x1, min(y + on, y2)), color, 1)
            cv2.line(canvas, (x2, y), (x2, min(y + on, y2)), color, 1)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

    if label is not None:
        cv2.putText(
            canvas, label, (x1 + 1, max(y1 - 2, 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )


def draw_trajectory_frames(
    image: torch.Tensor | np.ndarray,
    traj_boxes: Sequence[torch.Tensor | np.ndarray],
    gt_boxes: torch.Tensor | np.ndarray | None = None,
    H: int = 224,
    W: int = 224,
    step_label: bool = True,
) -> list[np.ndarray]:
    """Build per-step BGR uint8 frames for one sample.

    image:      (3, H, W) tensor [0,1] or (H, W, 3) ndarray
    traj_boxes: K+1 boxes-per-step, each (10, 4) in [0,1]
    gt_boxes:   optional (10, 4) overlay (dashed, same color as predicted)
    """
    bg = _to_bgr_uint8(image)
    K1 = len(traj_boxes)
    frames: list[np.ndarray] = []
    for k, boxes in enumerate(traj_boxes):
        canvas = bg.copy()
        if gt_boxes is not None:
            for i in range(10):
                _draw_box(canvas, gt_boxes[i], BOX_COLORS[i], dashed=True, H=H, W=W)
        for i in range(10):
            _draw_box(canvas, boxes[i], BOX_COLORS[i], label=str(i), H=H, W=W)
        if step_label:
            cv2.putText(
                canvas, f"t={k}/{K1 - 1}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        frames.append(canvas)
    return frames


def save_gif(frames: Iterable[np.ndarray], path: str | Path, fps: int = 8) -> Path:
    """Write BGR uint8 frames as a looping GIF (BGR→RGB conversion inside)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(path, rgb_frames, duration=1.0 / fps, loop=0)
    return path


if __name__ == "__main__":
    # End-to-end sanity: real MNIST + untrained SignalFlowModel → GIF
    from dataset.mnist_box_dataset import MNISTBoxDataset
    from model.flow_signal import SignalFlowModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MNISTBoxDataset(split="val", root="./data")
    sample = ds[0]
    image = sample["image"].unsqueeze(0).to(device)   # (1, 3, 224, 224)
    gt_boxes = sample["gt_boxes"]                      # (10, 4)

    model = SignalFlowModel(
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    ).to(device).eval()

    K = 16
    _, traj_boxes_b = model.sample(image, K=K)
    assert len(traj_boxes_b) == K + 1
    traj_boxes = [b.squeeze(0).cpu() for b in traj_boxes_b]   # K+1 of (10, 4)

    frames = draw_trajectory_frames(sample["image"], traj_boxes, gt_boxes=gt_boxes)
    assert len(frames) == K + 1, f"frame 수: {len(frames)}"
    assert frames[0].shape == (224, 224, 3), f"frame shape: {frames[0].shape}"
    assert frames[0].dtype == np.uint8, f"frame dtype: {frames[0].dtype}"

    out = Path("outputs/figures/sanity_traj.gif")
    save_gif(frames, out, fps=6)
    assert out.exists() and out.stat().st_size > 0, f"GIF 저장 실패: {out}"
    print(f"GIF 저장: {out}  (frames={len(frames)}, size={out.stat().st_size}B)")
    print("training/visualize sanity check 통과")
