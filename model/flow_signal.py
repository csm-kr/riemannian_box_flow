"""Phase 1 — Euclidean (signal) flow matching model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiTBackbone
from .trajectory import (
    euclidean_trajectory, sample_init_box, signal_decode, signal_encode,
)


class SignalFlowModel(nn.Module):
    def __init__(self, init_prior: str = "default", **backbone_kwargs):
        super().__init__()
        self.backbone = DiTBackbone(**backbone_kwargs)
        self.init_prior = init_prior

    def forward(
        self,
        s_t: torch.Tensor,    # (B, n_q, 4)
        t: torch.Tensor,      # (B,)
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> torch.Tensor:
        return self.backbone(s_t, t, image)

    def sample_init(self, reference: torch.Tensor) -> torch.Tensor:
        """b_0 from configured prior, then encoded to signal space."""
        b_0 = sample_init_box(reference, prior=self.init_prior)
        return signal_encode(b_0)

    def fm_loss(
        self,
        b_1: torch.Tensor,    # (B, n_q, 4) box space
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> tuple[torch.Tensor, dict]:
        b_0 = sample_init_box(b_1, prior=self.init_prior)
        B = b_1.shape[0]
        t = torch.rand(B, device=b_1.device)
        s_t, u_target = euclidean_trajectory(b_0, b_1, t)
        u_pred = self.forward(s_t, t, image)
        loss = F.mse_loss(u_pred, u_target)
        return loss, {"u_pred": u_pred, "u_target": u_target, "s_t": s_t, "t": t}

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        K: int = 16,
        n_queries: int = 10,
        init_box: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """ODE Euler K-step in signal space. Returns (boxes, traj_boxes[K+1]).

        init_box: optional (B, n_queries, 4) shared init in box space.
                  If None, samples internally as `clip(N(0,I), -3, 3)` in signal.
                  Sharing init_box across baselines enables fair paired comparison.
        """
        device = image.device
        B = image.shape[0]
        if init_box is None:
            ref = torch.empty(B, n_queries, 4, device=device)
            b = sample_init_box(ref, prior=self.init_prior)
            s = signal_encode(b)
        else:
            s = signal_encode(init_box.to(device))
        traj = [s.clone()]
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            v = self.forward(s, t, image)
            s = s + dt * v
            traj.append(s.clone())
        boxes = signal_decode(s).clamp(0, 1)
        traj_boxes = [signal_decode(z).clamp(0, 1) for z in traj]
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

    # 5) init_prior="small_size" propagates to fm_loss + sample
    model_sm = SignalFlowModel(
        init_prior="small_size",
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    _, traj_sm = model_sm.sample(image, K=K)
    init_b = traj_sm[0]   # signal-decoded box at t=0
    assert (init_b[..., 2:] >= 0.01 - 1e-6).all(), init_b[..., 2:].min()
    assert (init_b[..., 2:] <= 0.05 + 1e-6).all(), init_b[..., 2:].max()

    print("flow_signal sanity check 통과")
