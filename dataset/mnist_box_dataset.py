import torch
from torch.utils.data import Dataset

try:
    from .mnist_source import MNISTSource
    from .sampler import sample_gt_boxes, sample_init_signal
    from .canvas import compose_canvas
    from .box_utils import box_to_signal, signal_to_box
except ImportError:
    from mnist_source import MNISTSource
    from sampler import sample_gt_boxes, sample_init_signal
    from canvas import compose_canvas
    from box_utils import box_to_signal, signal_to_box

_SPLIT_SIZES = {"train": 50000, "val": 5000, "test": 5000}


class MNISTBoxDataset(Dataset):
    def __init__(self, split: str = "train", root: str = "./data",
                 wide: bool = False):
        assert split in _SPLIT_SIZES, f"split은 {list(_SPLIT_SIZES)}중 하나여야 함"
        self.length = _SPLIT_SIZES[split]
        self.source = MNISTSource(root=root, train=(split == "train"))
        self.wide = wide

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        digits = [self.source.get_digit(i) for i in range(10)]
        gt_boxes = sample_gt_boxes(wide=self.wide)      # (10, 4) normalized cx,cy,w,h
        canvas = compose_canvas(digits, gt_boxes)       # (H, W, 3) uint8

        init_signals = sample_init_signal()             # (10, 4) in [-3, 3]
        gt_signals = box_to_signal(gt_boxes)            # (10, 4) in [-3, 3]
        init_boxes = signal_to_box(init_signals)        # (10, 4) in [0, 1]

        image = torch.from_numpy(canvas.transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "gt_boxes": torch.from_numpy(gt_boxes).float(),
            "init_boxes": torch.from_numpy(init_boxes).float(),
            "gt_signals": torch.from_numpy(gt_signals).float(),
            "init_signals": torch.from_numpy(init_signals).float(),
            "labels": torch.arange(10),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset.visualize import show_sample

    dataset = MNISTBoxDataset(split="train", root="./data")
    print(f"dataset size: {len(dataset)}")

    for data in dataset:
        show_sample(data)

