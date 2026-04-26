"""DINOv2 ViT-S/14 wrapper. 224 input → (B, 256, 384) patch tokens."""

import warnings

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=pretrained
            )
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self._freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]


if __name__ == "__main__":
    enc = ImageEncoder(pretrained=False, freeze=True)
    assert enc.embed_dim == 384
    assert enc.patch_size == 14

    # 1) shape: 224 input → 16*16=256 patches
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 256, 384), f"shape: {out.shape}"

    # 2) frozen — no grad
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert n_trainable == 0, f"frozen 안 됨: {n_trainable} trainable"

    # 3) train() mode 후에도 internal model은 eval
    enc.train(True)
    assert not enc.model.training, "frozen model이 .train() 후 train mode가 됨"

    print("image_encoder sanity check 통과")
