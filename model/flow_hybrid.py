"""Hybrid baseline (exp 007) — position uses signal, size uses chart.

State (B, Q, 4) = [s_cx, s_cy, log w, log h]
- pos (cx, cy): signal-space affine (s = 6b - 3) → linear interp + constant u
- size (w, h):  chart-space log → linear interp + constant u

Best of both: signal's affine box-IoU friendliness for position +
chart's multiplicative dynamics for size. Both targets are constant in t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiTBackbone
from .trajectory import EPS, signal_decode


class HybridFlowModel(nn.Module):
    """Position-signal + size-chart hybrid."""

    def __init__(self, **backbone_kwargs):
        super().__init__()
        self.backbone = DiTBackbone(**backbone_kwargs)

    def forward(self, x_t, t, image):
        return self.backbone(x_t, t, image)

    @staticmethod
    def sample_init_box(reference: torch.Tensor) -> torch.Tensor:
        s = torch.randn_like(reference).clamp_(-3, 3)
        return signal_decode(s)

    @staticmethod
    def box_to_hybrid(b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        """[cx, cy, w, h] → [6cx-3, 6cy-3, log w, log h]."""
        pos = 6.0 * b[..., :2] - 3.0
        siz = b[..., 2:].clamp_min(eps).log()
        return torch.cat([pos, siz], dim=-1)

    @staticmethod
    def hybrid_to_box(x: torch.Tensor) -> torch.Tensor:
        pos = (x[..., :2] + 3.0) / 6.0
        siz = x[..., 2:].exp()
        return torch.cat([pos, siz], dim=-1)

    def fm_loss(self, b_1, image):
        b_0 = self.sample_init_box(b_1)
        x_0 = self.box_to_hybrid(b_0)
        x_1 = self.box_to_hybrid(b_1)
        B = b_1.shape[0]
        t = torch.rand(B, device=b_1.device)
        t_b = t.view(B, 1, 1)
        x_t = (1 - t_b) * x_0 + t_b * x_1
        u_target = (x_1 - x_0).expand_as(x_t)   # constant
        u_pred = self.forward(x_t, t, image)
        loss = F.mse_loss(u_pred, u_target)
        return loss, {"u_pred": u_pred, "u_target": u_target, "x_t": x_t, "t": t}

    @torch.no_grad()
    def sample(self, image, K=10, n_queries=10, init_box=None):
        device = image.device
        B = image.shape[0]
        if init_box is None:
            s = torch.randn(B, n_queries, 4, device=device).clamp_(-3, 3)
            init_box = signal_decode(s)
        x = self.box_to_hybrid(init_box.to(device))
        traj = [x.clone()]
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            v = self.forward(x, t, image)
            x = x + dt * v
            traj.append(x.clone())
        boxes = self.hybrid_to_box(x).clamp(0, 1)
        traj_boxes = [self.hybrid_to_box(z).clamp(0, 1) for z in traj]
        return boxes, traj_boxes


if __name__ == "__main__":
    torch.manual_seed(0)
    model = HybridFlowModel(hidden_size=64, depth=2, num_heads=4, n_queries=10,
                             encoder_pretrained=False)
    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4); b_1[..., 2:].clamp_(min=1e-3)

    loss, info = model.fm_loss(b_1, image)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert info["u_pred"].shape == (B, 10, 4)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)

    K = 8
    boxes, traj = model.sample(image, K=K)
    assert boxes.shape == (B, 10, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all()
    assert len(traj) == K + 1

    b0 = torch.rand(B, 10, 4); b0[..., 2:].clamp_(min=1e-3)
    p1, _ = model.sample(image, K=K, init_box=b0)
    p2, _ = model.sample(image, K=K, init_box=b0)
    assert torch.allclose(p1, p2)
    print("flow_hybrid sanity check 통과")
