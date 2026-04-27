"""Box-loss-trained chart-native baseline (exp 008).

Same architecture as ChartNativeFlowModel (chart state, chart-straight trajectory)
but trains on **box-space MSE on the predicted endpoint** instead of chart-space
MSE on velocity. Hypothesis: optimizing the metric we care about (box) closes
the gap caused by exp() amplification.
"""

import torch
import torch.nn.functional as F

from .flow_chart_native import ChartNativeFlowModel
from .trajectory import chart_decode, chart_encode


class ChartBoxLossFlowModel(ChartNativeFlowModel):
    """Chart model + chart trajectory + **box-space loss**."""

    def fm_loss(self, b_1, image):
        b_0 = self.sample_init_box(b_1)
        c_0 = chart_encode(b_0)
        c_1 = chart_encode(b_1)
        B = b_1.shape[0]
        t = torch.rand(B, device=b_1.device)
        t_b = t.view(B, 1, 1)
        c_t = (1 - t_b) * c_0 + t_b * c_1
        u_pred = self.forward(c_t, t, image)

        # Reconstruct predicted endpoint:  c_1_pred = c_t + (1-t) * u_pred
        # If u_pred = c_1 - c_0 exactly, c_1_pred = c_1 always.
        c_1_pred = c_t + (1 - t_b) * u_pred
        b_1_pred = chart_decode(c_1_pred)
        # Box-space MSE on predicted endpoint
        loss = F.mse_loss(b_1_pred, b_1)
        return loss, {"u_pred": u_pred, "c_t": c_t, "b_1_pred": b_1_pred, "t": t}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ChartBoxLossFlowModel(
        hidden_size=64, depth=2, num_heads=4, n_queries=10,
        encoder_pretrained=False,
    )
    B = 2
    image = torch.randn(B, 3, 224, 224)
    b_1 = torch.rand(B, 10, 4); b_1[..., 2:].clamp_(min=1e-3)

    loss, info = model.fm_loss(b_1, image)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad in box-loss training"
    print("flow_chart_boxloss sanity check 통과")
