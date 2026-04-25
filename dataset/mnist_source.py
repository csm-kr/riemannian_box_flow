import numpy as np
from torchvision.datasets import MNIST


class MNISTSource:
    def __init__(self, root: str = "./data", train: bool = True):
        self.dataset = MNIST(root=root, train=train, download=True)
        self._index: dict[int, list[int]] = {i: [] for i in range(10)}
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            self._index[label].append(idx)

    def get_digit(self, label: int) -> np.ndarray:
        """label에 해당하는 random digit 이미지 반환. Shape: (H, W) uint8."""
        idx = np.random.choice(self._index[label])
        img, _ = self.dataset[idx]
        return np.array(img)  # PIL -> numpy (28, 28) uint8


if __name__ == "__main__":
    src = MNISTSource(root="./data", train=True)
    for label in range(10):
        digit = src.get_digit(label)
        assert digit.shape == (28, 28), f"shape 오류: {digit.shape}"
        assert digit.dtype == np.uint8
        print(f"digit {label}: shape={digit.shape}, max={digit.max()}")
    print("mnist_source sanity check 통과")
