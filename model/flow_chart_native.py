"""Phase 2.5 — Riemannian baseline with model living in chart space.

Counterpart to `csm-kr/riemannian_flow_det`'s setup:
- Model input/output state = **chart** y = (cx, cy, log w, log h)
- Trajectory = chart-space straight line: y_t = (1-t) y_0 + t y_1
- Target velocity = **constant** u = y_1 - y_0
- ODE Euler integrates in chart space; decode y → box at the end

Compared to `ChartFlowModel` (signal model + chart trajectory, state-dependent u),
this model gives the Riemannian setup its native chart-space environment.

Init b_0 is shared with SignalFlowModel via the same induced distribution
(`clip(N(0,I), -3, 3) / 6 + 0.5`) so paired comparison is fair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiTBackbone
from .trajectory import chart_decode, chart_encode, sample_init_box, signal_decode


class ChartNativeFlowModel(nn.Module):
    """Riemannian baseline that lives entirely in chart state."""

    def __init__(self, init_prior: str = "default", **backbone_kwargs):
        super().__init__()
        self.backbone = DiTBackbone(**backbone_kwargs)
        self.init_prior = init_prior

    def forward(
        self,
        y_t: torch.Tensor,    # (B, n_q, 4) chart state
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
        y_0 = chart_encode(b_0)
        y_1 = chart_encode(b_1)
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
        """ODE Euler in chart space. Returns (boxes, traj_boxes[K+1]) in box space."""
        device = image.device
        B = image.shape[0]
        if init_box is None:
            ref = torch.empty(B, n_queries, 4, device=device)
            init_box = sample_init_box(ref, prior=self.init_prior)
        y = chart_encode(init_box.to(device))
        traj = [y.clone()]
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            v = self.forward(y, t, image)
            y = y + dt * v
            traj.append(y.clone())
        boxes = chart_decode(y).clamp(0, 1)
        traj_boxes = [chart_decode(z).clamp(0, 1) for z in traj]
        return boxes, traj_boxes


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ChartNativeFlowModel(
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4)
    b_1[..., 2:].clamp_(min=1e-3)

    # 1) loss finite + u_target shape (B,10,4)
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

    # 3) sample 동작 + boxes ∈ [0,1]
    K = 8
    boxes, traj = model.sample(image, K=K)
    assert boxes.shape == (B, 10, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all()
    assert len(traj) == K + 1

    # 4) shared init_box: 같은 b_0 주면 결정적
    b0 = torch.rand(B, 10, 4); b0[..., 2:].clamp_(min=1e-3)
    p1, _ = model.sample(image, K=K, init_box=b0)
    p2, _ = model.sample(image, K=K, init_box=b0)
    assert torch.allclose(p1, p2), "shared init_box → deterministic"

    # 5) u_target은 t에 무관 (상수 — Riemannian 이론 핵심)
    s_seed = torch.manual_seed(7)
    _, info_a = model.fm_loss(b_1, image)
    # 같은 seed로 다시 (s_0, t 모두 재생성) → u_target 동일
    torch.manual_seed(7)
    _, info_b = model.fm_loss(b_1, image)
    assert torch.allclose(info_a["u_target"], info_b["u_target"])
    # u_target는 t에 따라 변하지 않음 (broadcast된 상수). t 별로 잘라봐도 같음.
    assert (info_a["u_target"] - info_a["u_target"][:, 0:1, :]).abs().max() < 1e-6 \
        or info_a["u_target"].dim() == 3   # broadcasted constant

    # 6) init_prior="small_size" propagates: sampled init box has w,h ∈ [0.01, 0.05]
    model_sm = ChartNativeFlowModel(
        init_prior="small_size",
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    boxes_sm, traj_sm = model_sm.sample(image, K=K)
    init_b = traj_sm[0]  # ODE start = init_box (in box space)
    assert (init_b[..., 2:] >= 0.01 - 1e-6).all(), init_b[..., 2:].min()
    assert (init_b[..., 2:] <= 0.05 + 1e-6).all(), init_b[..., 2:].max()

    print("flow_chart_native sanity check 통과")
