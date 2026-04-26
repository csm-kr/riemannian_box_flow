"""t embedding: sinusoidal + 2-layer MLP, used by adaLN in DiT blocks."""

import math

import torch
import torch.nn as nn


def _sinusoidal(t: torch.Tensor, dim: int, max_period: int) -> torch.Tensor:
    """t: (B,) → (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        t_scale: float = 1000.0,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.t_scale = t_scale
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t ∈ [0, 1] → DDPM-style scaling so sinusoidal frequencies span useful range
        emb = _sinusoidal(t * self.t_scale, self.frequency_embedding_size, self.max_period)
        return self.mlp(emb)


if __name__ == "__main__":
    B, hidden = 4, 384

    embedder = TimeEmbed(hidden_size=hidden)

    # 1) shape
    t = torch.rand(B)
    out = embedder(t)
    assert out.shape == (B, hidden), f"shape 오류: {out.shape}"

    # 2) determinism
    out2 = embedder(t)
    assert torch.allclose(out, out2), "동일 t에 대해 출력 달라짐"

    # 3) distinguishes different t
    t_a = torch.full((B,), 0.1)
    t_b = torch.full((B,), 0.9)
    assert not torch.allclose(embedder(t_a), embedder(t_b)), "t 변화에 둔감"

    # 4) differentiable through MLP
    t_g = torch.rand(B, requires_grad=False)
    out_g = embedder(t_g)
    loss = out_g.sum()
    loss.backward()
    grads = [p.grad for p in embedder.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(g.abs().sum() > 0 for g in grads), "MLP gradient 흐르지 않음"

    # 5) dtype/device — fp32 default
    assert out.dtype == torch.float32

    print("time_embed sanity check 통과")
