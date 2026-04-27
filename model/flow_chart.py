"""Phase 2 — Riemannian (chart) flow matching model.

Inherits backbone / forward / sample / sample_init from `SignalFlowModel`.
Only `fm_loss` differs: target velocity follows a chart-space (psi) straight
line decoded into signal space (`model/trajectory.py:riemannian_trajectory`).

See plans/model.md §7.4 and plans/active.md §A.
"""

import torch
import torch.nn.functional as F

from .flow_signal import SignalFlowModel
from .trajectory import riemannian_trajectory


class ChartFlowModel(SignalFlowModel):
    """Riemannian baseline. Same model output (signal-space velocity) and ODE as
    `SignalFlowModel`; only the training target trajectory is different."""

    def fm_loss(
        self,
        b_1: torch.Tensor,    # (B, n_q, 4) box space, w/h > 0
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> tuple[torch.Tensor, dict]:
        s_1_ref = b_1 * 6.0 - 3.0       # for shape/device of sample_init
        s_0 = self.sample_init(s_1_ref)
        b_0 = (s_0 + 3.0) / 6.0
        B = b_1.shape[0]
        t = torch.rand(B, device=b_1.device)
        s_t, u_target = riemannian_trajectory(b_0, b_1, t)
        u_pred = self.forward(s_t, t, image)
        loss = F.mse_loss(u_pred, u_target)
        return loss, {"u_pred": u_pred, "u_target": u_target, "s_t": s_t, "t": t}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ChartFlowModel(
        hidden_size=64,
        depth=2,
        num_heads=4,
        n_queries=10,
        encoder_pretrained=False,
    )

    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4)
    b_1[..., 2:].clamp_(min=1e-3)   # ensure w, h >= eps before chart_encode

    # 1) loss is finite scalar; u_pred shape is (B, 10, 4)
    loss, info = model.fm_loss(b_1, image)
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss: {loss}"
    assert info["u_pred"].shape == (B, 10, 4)
    assert info["u_target"].shape == (B, 10, 4)
    assert torch.isfinite(info["u_target"]).all(), "u_target has non-finite (eps clamp 동작 확인)"

    # 2) backward 통과, finite gradients
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad in Riemannian fm_loss"

    # 3) sample (ODE)는 SignalFlowModel과 동일하게 동작
    K = 8
    boxes, traj = model.sample(image, K=K)
    assert boxes.shape == (B, 10, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all()
    assert len(traj) == K + 1
    assert traj[0].shape == (B, 10, 4)

    # 4) sample은 stochastic (s_0 random init)
    boxes2, _ = model.sample(image, K=K)
    assert not torch.allclose(boxes, boxes2), "sampling이 deterministic"

    # 5) Riemannian fm_loss target은 Euclidean과 다름 (위치 같고 size 다름)
    from .trajectory import euclidean_trajectory
    torch.manual_seed(7)
    b0 = torch.rand(B, 10, 4); b0[..., 2:].clamp_(min=1e-3)
    b1 = torch.rand(B, 10, 4); b1[..., 2:].clamp_(min=1e-3)
    t = torch.rand(B)
    _, u_e = euclidean_trajectory(b0, b1, t)
    _, u_r = riemannian_trajectory(b0, b1, t)
    assert torch.allclose(u_e[..., :2], u_r[..., :2], atol=1e-5), "위치 성분 일치해야 함"
    assert (u_e[..., 2:] - u_r[..., 2:]).abs().max() > 1e-3, "size 성분 달라야 함"

    print("flow_chart sanity check 통과")
