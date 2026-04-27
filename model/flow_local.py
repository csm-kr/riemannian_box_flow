"""Local chart (body-frame) baseline — exp 010.

Implements `model.md §7.3`: scale-aware local chart anchored at current b_t.
- ψ_{b_t}(b) = ((c_x - c_x^t)/w_t, (c_y - c_y^t)/h_t, log(w/w_t), log(h/h_t))
- u_target (body frame) = ψ_{b_t}(b_1) / (1 - t)
- Exp_{b_t}(dt · u) = (c_x^t + dt·w_t·u_x, c_y^t + dt·h_t·u_y, w_t·exp(dt·u_w), h_t·exp(dt·u_h))

True Riemannian setup with parallel transport — same dynamics regardless of box scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DiTBackbone
from .trajectory import EPS, chart_decode, chart_encode, signal_decode


class LocalChartFlowModel(nn.Module):
    def __init__(self, t_max: float = 0.999, **backbone_kwargs):
        super().__init__()
        self.backbone = DiTBackbone(**backbone_kwargs)
        self.t_max = t_max  # avoid singularity at t=1

    def forward(self, c_t, t, image):
        return self.backbone(c_t, t, image)

    @staticmethod
    def sample_init_box(reference):
        s = torch.randn_like(reference).clamp_(-3, 3)
        return signal_decode(s)

    def fm_loss(self, b_1, image):
        b_0 = self.sample_init_box(b_1)
        c_0 = chart_encode(b_0)
        c_1 = chart_encode(b_1)
        B = b_1.shape[0]
        # t ~ U(0, t_max)  to avoid 1-t division blow-up
        t = torch.rand(B, device=b_1.device) * self.t_max
        t_b = t.view(B, 1, 1)

        # Geodesic trajectory: chart-straight (linear pos, log-linear size in box)
        c_t = (1 - t_b) * c_0 + t_b * c_1
        b_t = chart_decode(c_t)

        # Body-frame target velocity = ψ_{b_t}(b_1) / (1 - t)
        wt = b_t[..., 2:3].clamp_min(EPS)
        ht = b_t[..., 3:4].clamp_min(EPS)
        psi_x = (b_1[..., 0:1] - b_t[..., 0:1]) / wt
        psi_y = (b_1[..., 1:2] - b_t[..., 1:2]) / ht
        psi_w = (b_1[..., 2:3].clamp_min(EPS) / wt).log()
        psi_h = (b_1[..., 3:4].clamp_min(EPS) / ht).log()
        denom = (1 - t_b).clamp_min(1e-3)
        u_target = torch.cat([psi_x, psi_y, psi_w, psi_h], dim=-1) / denom

        # Model input: global chart at current b_t (so model knows where it is)
        u_pred = self.forward(c_t, t, image)
        loss = F.mse_loss(u_pred, u_target)
        return loss, {"u_pred": u_pred, "u_target": u_target, "c_t": c_t, "t": t}

    @torch.no_grad()
    def sample(self, image, K=10, n_queries=10, init_box=None):
        device = image.device
        B = image.shape[0]
        if init_box is None:
            s = torch.randn(B, n_queries, 4, device=device).clamp_(-3, 3)
            init_box = signal_decode(s)
        b = init_box.to(device)
        traj = [b.clone()]
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=device)
            c = chart_encode(b)
            u = self.forward(c, t, image)
            wt = b[..., 2:3].clamp_min(EPS)
            ht = b[..., 3:4].clamp_min(EPS)
            new_b = torch.cat([
                b[..., 0:1] + dt * wt * u[..., 0:1],
                b[..., 1:2] + dt * ht * u[..., 1:2],
                wt * (dt * u[..., 2:3]).exp(),
                ht * (dt * u[..., 3:4]).exp(),
            ], dim=-1)
            b = new_b
            traj.append(b.clone())
        boxes = b.clamp(0, 1)
        traj_boxes = [t.clamp(0, 1) for t in traj]
        return boxes, traj_boxes


if __name__ == "__main__":
    torch.manual_seed(0)
    model = LocalChartFlowModel(
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4); b_1[..., 2:].clamp_(min=1e-3)

    loss, info = model.fm_loss(b_1, image)
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss={loss}"
    assert info["u_target"].shape == (B, 10, 4)
    assert torch.isfinite(info["u_target"]).all(), "non-finite u_target"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad"

    K = 8
    boxes, traj = model.sample(image, K=K)
    assert boxes.shape == (B, 10, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all()

    b0 = torch.rand(B, 10, 4); b0[..., 2:].clamp_(min=1e-3)
    p1, _ = model.sample(image, K=K, init_box=b0)
    p2, _ = model.sample(image, K=K, init_box=b0)
    assert torch.allclose(p1, p2)
    print("flow_local sanity check 통과")
