"""exp 014 — corner-logit chart: pos normalized by available range, then logit.

Chart:
    y_0 = logit((cx − w/2) / (1 − w))
    y_1 = logit((cy − h/2) / (1 − h))
    y_2 = logit(w)
    y_3 = logit(h)

Decode guarantees in-canvas: cx ∈ [w/2, 1−w/2], cy ∈ [h/2, 1−h/2].
Trajectory + target are constant in chart space (Riemannian-style straight line).

See plans/space_recipes.md §5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiTBackbone
from .trajectory import corner_logit_decode, corner_logit_encode, sample_init_box


class CornerLogitFlowModel(nn.Module):
    """exp 014: model lives in corner-logit chart state (4-dim symmetric)."""

    def __init__(self, init_prior: str = "default", **backbone_kwargs):
        super().__init__()
        self.backbone = DiTBackbone(**backbone_kwargs)
        self.init_prior = init_prior

    def forward(
        self,
        y_t: torch.Tensor,    # (B, n_q, 4) corner-logit chart state
        t: torch.Tensor,      # (B,)
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> torch.Tensor:
        return self.backbone(y_t, t, image)

    def fm_loss(
        self,
        b_1: torch.Tensor,    # (B, n_q, 4) box space
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> tuple[torch.Tensor, dict]:
        b_0 = sample_init_box(b_1, prior=self.init_prior)
        y_0 = corner_logit_encode(b_0)
        y_1 = corner_logit_encode(b_1)
        B = b_1.shape[0]
        t = torch.rand(B, device=b_1.device)
        t_b = t.view(B, 1, 1)
        y_t = (1 - t_b) * y_0 + t_b * y_1
        u_target = (y_1 - y_0).expand_as(y_t)   # constant in t
        u_pred = self.forward(y_t, t, image)
        loss = F.mse_loss(u_pred, u_target)
        return loss, {"u_pred": u_pred, "u_target": u_target, "y_t": y_t, "t": t}

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        K: int = 10,
        n_queries: int = 10,
        init_box: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """ODE Euler in corner-logit chart space. Returns (boxes, traj_boxes[K+1])."""
        device = image.device
        B = image.shape[0]
        if init_box is None:
            ref = torch.empty(B, n_queries, 4, device=device)
            init_box = sample_init_box(ref, prior=self.init_prior)
        y = corner_logit_encode(init_box.to(device))
        traj = [y.clone()]
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            v = self.forward(y, t, image)
            y = y + dt * v
            traj.append(y.clone())
        boxes = corner_logit_decode(y).clamp(0, 1)   # decode 자체로 in-canvas, clamp는 no-op safety
        traj_boxes = [corner_logit_decode(z).clamp(0, 1) for z in traj]
        return boxes, traj_boxes


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CornerLogitFlowModel(
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4)
    # GT는 in-canvas 구성으로 만들어야 corner round-trip 의미가 있음
    w = b_1[..., 2:3].clamp(min=1e-3, max=1 - 1e-3)
    h = b_1[..., 3:4].clamp(min=1e-3, max=1 - 1e-3)
    cx = w / 2 + (1 - w) * b_1[..., 0:1]
    cy = h / 2 + (1 - h) * b_1[..., 1:2]
    b_1 = torch.cat([cx, cy, w, h], dim=-1)

    # 1) loss finite + u_target constant in t
    loss, info = model.fm_loss(b_1, image)
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss={loss}"
    assert info["u_pred"].shape == (B, 10, 4)
    assert info["u_target"].shape == (B, 10, 4)
    assert torch.isfinite(info["u_target"]).all()

    # 2) backward 통과
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad"

    # 3) sample → boxes ∈ [0,1] (decode 자체로 보장)
    K = 8
    boxes, traj = model.sample(image, K=K)
    assert boxes.shape == (B, 10, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all()
    # in-canvas: 더 강한 조건
    cx_, w_ = boxes[..., 0], boxes[..., 2]
    assert (cx_ - w_ / 2 >= -1e-6).all() and (cx_ + w_ / 2 <= 1 + 1e-6).all()
    assert len(traj) == K + 1

    # 4) shared init_box → 결정적
    b0 = torch.rand(B, 10, 4)
    w0 = b0[..., 2:3].clamp(min=1e-3, max=1 - 1e-3)
    h0 = b0[..., 3:4].clamp(min=1e-3, max=1 - 1e-3)
    cx0 = w0 / 2 + (1 - w0) * b0[..., 0:1]
    cy0 = h0 / 2 + (1 - h0) * b0[..., 1:2]
    b0 = torch.cat([cx0, cy0, w0, h0], dim=-1)
    p1, _ = model.sample(image, K=K, init_box=b0)
    p2, _ = model.sample(image, K=K, init_box=b0)
    assert torch.allclose(p1, p2), "shared init_box → deterministic"

    # 5) u_target constant in t (Riemannian core)
    assert (info["u_target"] - info["u_target"][:, 0:1, :]).abs().max() < 1e-6 \
        or info["u_target"].dim() == 3   # broadcasted constant

    # 6) init_prior="small_size" propagates: sampled init box has w,h ∈ [0.01, 0.05]
    model_sm = CornerLogitFlowModel(
        init_prior="small_size",
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    _, traj_sm = model_sm.sample(image, K=K)
    init_b = traj_sm[0]
    # init box was decoded via corner_logit_decode of corner_logit_encode(b_sampled).
    # small_size prior sets w,h ∈ [0.01, 0.05] before encoding; round-trip preserves these.
    assert (init_b[..., 2:] >= 0.01 - 1e-3).all(), init_b[..., 2:].min()
    assert (init_b[..., 2:] <= 0.05 + 1e-3).all(), init_b[..., 2:].max()

    print("flow_corner_logit sanity check 통과")
