"""Phase 2.5 — Linear (Euclidean) baseline with model living in chart space.

Counterpart to `csm-kr/riemannian_flow_det`'s "Linear" baseline:
- Model input/output state = chart y = (cx, cy, log w, log h)
- Trajectory = **box-space (cxcywh) straight line**: b_t = (1-t) b_0 + t b_1
- Target velocity in chart space = dy/dt — **state-dependent**:
    pos: u = b_1[..., :2] - b_0[..., :2]                  (constant, since y_pos = b_pos)
    siz: u = (b_1[..., 2:] - b_0[..., 2:]) / b_t[..., 2:]  (∝ 1/w_t — diverges for small w)
- ODE Euler integrates in chart space (same as ChartNativeFlowModel)

This is the 4th cell of the (model state, trajectory) × (eucl, riem) matrix:
- S-E (signal/signal), S-R (signal/chart), C-R (chart/chart), C-E (chart/box).
Mirrors the "weak" baseline csm-kr uses to show Riemannian wins in chart space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiTBackbone
from .trajectory import EPS, chart_decode, chart_encode, signal_decode


class ChartLinearFlowModel(nn.Module):
    """C-E baseline: chart-state model + box-space straight trajectory."""

    def __init__(self, **backbone_kwargs):
        super().__init__()
        self.backbone = DiTBackbone(**backbone_kwargs)

    def forward(
        self,
        y_t: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(y_t, t, image)

    @staticmethod
    def sample_init_box(reference: torch.Tensor) -> torch.Tensor:
        """Same shared init as ChartNativeFlowModel / SignalFlowModel."""
        s = torch.randn_like(reference).clamp_(-3, 3)
        return signal_decode(s)

    def fm_loss(
        self,
        b_1: torch.Tensor,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        b_0 = self.sample_init_box(b_1)
        B = b_1.shape[0]
        t = torch.rand(B, device=b_1.device)
        t_b = t.view(B, 1, 1)
        # Box-space straight line
        b_t = (1 - t_b) * b_0 + t_b * b_1
        # Map to chart for model input
        y_t = chart_encode(b_t)
        # Target velocity in chart space = d(chart_encode(b_t))/dt
        #   pos:  u_cx, u_cy = b_1[:2] - b_0[:2]   (constant, identity map for position)
        #   size: u_lw = d/dt log b_t[2] = (b_1[2] - b_0[2]) / b_t[2]
        pos_diff = (b_1[..., :2] - b_0[..., :2]).expand(*b_t.shape[:-1], 2)
        size_diff = b_1[..., 2:] - b_0[..., 2:]
        u_size = size_diff / b_t[..., 2:].clamp_min(EPS)
        u_target = torch.cat([pos_diff, u_size], dim=-1)
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
        """Same as ChartNativeFlowModel — integrate in chart space, decode at end."""
        device = image.device
        B = image.shape[0]
        if init_box is None:
            s = torch.randn(B, n_queries, 4, device=device).clamp_(-3, 3)
            init_box = signal_decode(s)
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
    model = ChartLinearFlowModel(
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4); b_1[..., 2:].clamp_(min=1e-3)

    # 1) loss finite + shape OK
    loss, info = model.fm_loss(b_1, image)
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss={loss}"
    assert info["u_pred"].shape == (B, 10, 4)
    assert info["u_target"].shape == (B, 10, 4)
    assert torch.isfinite(info["u_target"]).all(), "non-finite u_target (eps clamp 동작 확인)"

    # 2) backward 통과
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad"

    # 3) sample 동작
    K = 8
    boxes, traj = model.sample(image, K=K)
    assert boxes.shape == (B, 10, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all()
    assert len(traj) == K + 1

    # 4) shared init_box → 결정적
    b0 = torch.rand(B, 10, 4); b0[..., 2:].clamp_(min=1e-3)
    p1, _ = model.sample(image, K=K, init_box=b0)
    p2, _ = model.sample(image, K=K, init_box=b0)
    assert torch.allclose(p1, p2)

    # 5) u_target은 t에 의존 (state-dependent — Linear 셋업의 핵심)
    torch.manual_seed(7); _, infoA = model.fm_loss(b_1, image)
    torch.manual_seed(7); _, infoB = model.fm_loss(b_1, image)
    assert torch.allclose(infoA["u_target"], infoB["u_target"])
    # size 성분은 t에 따라 달라야 함 (1/b_t[2:] 인자)
    # — 같은 t 내에서는 batch 별로 다름. t 다른 경우는 별도 비교 필요. 여기선 finite 확인만.

    print("flow_chart_linear sanity check 통과")
