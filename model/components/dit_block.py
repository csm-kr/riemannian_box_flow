"""DiT block: self-attn(queries) + cross-attn(patches+RoPE) + FFN, adaLN(t)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope2d import apply_rope2d


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttentionRoPE(nn.Module):
    """Cross-attention with 2D RoPE on context (key) tokens only."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,        # (B, n_q, C)
        ctx: torch.Tensor,      # (B, N_p, C)
        freqs_cis: torch.Tensor # (N_p, head_dim/2) complex
    ) -> torch.Tensor:
        B, n_q, C = x.shape
        N_p = ctx.shape[1]
        q = self.q_proj(x).view(B, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, N_p, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, N_p, self.num_heads, self.head_dim).transpose(1, 2)
        k = apply_rope2d(k, freqs_cis)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, n_q, C)
        return self.out_proj(attn)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn = CrossAttentionRoPE(hidden_size, num_heads)

        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )

        # 9 modulation params: (shift, scale, gate) × (self-attn, cross-attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,         # (B, n_q, C)
        t_emb: torch.Tensor,     # (B, C)
        ctx: torch.Tensor,       # (B, N_p, C)
        ctx_freqs_cis: torch.Tensor,  # (N_p, head_dim/2) complex
    ) -> torch.Tensor:
        s_sa, c_sa, g_sa, s_ca, c_ca, g_ca, s_mlp, c_mlp, g_mlp = (
            self.adaLN_modulation(t_emb).chunk(9, dim=1)
        )

        h = modulate(self.norm1(x), s_sa, c_sa)
        h_sa, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + g_sa.unsqueeze(1) * h_sa

        h = modulate(self.norm2(x), s_ca, c_ca)
        x = x + g_ca.unsqueeze(1) * self.cross_attn(h, ctx, ctx_freqs_cis)

        h = modulate(self.norm3(x), s_mlp, c_mlp)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)

        return x


if __name__ == "__main__":
    from .rope2d import precompute_2d_rope_cis

    B, n_q, N_p, C, n_heads = 2, 10, 16, 64, 8
    head_dim = C // n_heads
    block = DiTBlock(hidden_size=C, num_heads=n_heads)

    x = torch.randn(B, n_q, C)
    t_emb = torch.randn(B, C)
    ctx = torch.randn(B, N_p, C)
    cis = precompute_2d_rope_cis(head_dim, 4, 4)

    # 1) forward shape
    out = block(x, t_emb, ctx, cis)
    assert out.shape == x.shape, f"shape: {out.shape}"

    # 2) grad flow
    out.sum().backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(g.abs().sum() > 0 for g in grads), "grad 흐르지 않음"

    # 3) t_emb sensitivity
    block.zero_grad()
    with torch.no_grad():
        t_a = torch.zeros(B, C)
        t_b = torch.ones(B, C)
        out_a = block(x, t_a, ctx, cis)
        out_b = block(x, t_b, ctx, cis)
    assert not torch.allclose(out_a, out_b), "t_emb 변화에 둔감"

    # 4) ctx sensitivity (cross-attn 작동)
    with torch.no_grad():
        ctx2 = torch.randn_like(ctx)
        out_c = block(x, t_emb, ctx2, cis)
    assert not torch.allclose(out, out_c), "ctx 변화에 둔감"

    print("dit_block sanity check 통과")
