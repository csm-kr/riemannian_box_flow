"""Phase 1 — Euclidean (signal) flow matching model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiTBackbone
from .charts.signal import box_to_signal, signal_to_box


class SignalFlowModel(nn.Module):
    def __init__(self, **backbone_kwargs):
        super().__init__()
        self.backbone = DiTBackbone(**backbone_kwargs)

    def forward(
        self,
        s_t: torch.Tensor,    # (B, n_q, 4)
        t: torch.Tensor,      # (B,)
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> torch.Tensor:
        return self.backbone(s_t, t, image)

    @staticmethod
    def sample_init(reference: torch.Tensor) -> torch.Tensor:
        """s_0 ~ clip(N(0, I), -3, 3) with same shape/device as reference."""
        return torch.randn_like(reference).clamp_(-3, 3)

    def fm_loss(
        self,
        b_1: torch.Tensor,    # (B, n_q, 4) box space
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> tuple[torch.Tensor, dict]:
        B = b_1.shape[0]
        s_1 = box_to_signal(b_1)
        s_0 = self.sample_init(s_1)
        t = torch.rand(B, device=s_1.device)
        t_b = t.view(B, 1, 1)
        s_t = (1 - t_b) * s_0 + t_b * s_1
        u_target = s_1 - s_0
        u_pred = self.forward(s_t, t, image)
        loss = F.mse_loss(u_pred, u_target)
        return loss, {"u_pred": u_pred, "u_target": u_target, "s_t": s_t, "t": t}

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        K: int = 16,
        n_queries: int = 10,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """ODE Euler K-step in signal space. Returns (boxes, traj_boxes[K+1])."""
        device = image.device
        B = image.shape[0]
        s = torch.randn(B, n_queries, 4, device=device).clamp_(-3, 3)
        traj = [s.clone()]
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            v = self.forward(s, t, image)
            s = s + dt * v
            traj.append(s.clone())
        boxes = signal_to_box(s).clamp(0, 1)
        traj_boxes = [signal_to_box(z).clamp(0, 1) for z in traj]
        return boxes, traj_boxes


if __name__ == "__main__":
    model = SignalFlowModel(
        hidden_size=64,
        depth=2,
        num_heads=4,
        n_queries=10,
        encoder_pretrained=False,
    )

    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4)  # GT boxes in [0, 1]

    # 1) fm_loss is finite scalar
    loss, info = model.fm_loss(b_1, image)
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss: {loss}"
    assert info["u_pred"].shape == (B, 10, 4)

    # 2) backward 통과
    loss.backward()
    trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(trainable_grads) > 0
    assert all(torch.isfinite(g).all() for g in trainable_grads), "non-finite grad"

    # 3) sample — boxes in [0,1], trajectory K+1 frames
    K = 8
    boxes, traj = model.sample(image, K=K)
    assert boxes.shape == (B, 10, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all()
    assert len(traj) == K + 1
    assert traj[0].shape == (B, 10, 4)

    # 4) sample은 forward와 다른 boxes를 매번 (s_0 random)
    boxes2, _ = model.sample(image, K=K)
    assert not torch.allclose(boxes, boxes2), "sampling이 deterministic"

    print("flow_signal sanity check 통과")
