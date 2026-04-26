"""chart-agnostic backbone: image + (x_t, t) → (B, n_q, out_dim)."""

import torch
import torch.nn as nn

from .components.dit_block import DiTBlock
from .components.image_encoder import ImageEncoder
from .components.rope2d import precompute_2d_rope_cis
from .components.time_embed import TimeEmbed


class DiTBackbone(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        n_queries: int = 10,
        in_dim: int = 4,
        out_dim: int = 4,
        image_size: int = 224,
        encoder_pretrained: bool = True,
        encoder_freeze: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            pretrained=encoder_pretrained, freeze=encoder_freeze
        )
        self.adapter = nn.Linear(self.image_encoder.embed_dim, hidden_size)
        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.query_embed = nn.Embedding(n_queries, hidden_size)
        self.t_embed = TimeEmbed(hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.final_proj = nn.Linear(hidden_size, out_dim)

        head_dim = hidden_size // num_heads
        grid = image_size // self.image_encoder.patch_size  # 224/14 = 16
        cis = precompute_2d_rope_cis(head_dim, grid, grid)
        self.register_buffer("rope_cis", cis, persistent=False)

        self.n_queries = n_queries

    def forward(
        self,
        x_t: torch.Tensor,    # (B, n_q, in_dim)
        t: torch.Tensor,      # (B,)
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> torch.Tensor:
        B = x_t.shape[0]

        patch_tokens = self.image_encoder(image)               # (B, N_p, C_dino)
        ctx = self.adapter(patch_tokens)                       # (B, N_p, C)

        idx = torch.arange(self.n_queries, device=x_t.device)
        q_emb = self.query_embed(idx).unsqueeze(0).expand(B, -1, -1)
        x = q_emb + self.input_proj(x_t)                       # (B, n_q, C)

        t_emb = self.t_embed(t)                                # (B, C)

        for block in self.blocks:
            x = block(x, t_emb, ctx, self.rope_cis)

        shift, scale = self.final_modulation(t_emb).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.final_proj(x)


if __name__ == "__main__":
    backbone = DiTBackbone(
        hidden_size=64,
        depth=2,
        num_heads=4,
        n_queries=10,
        encoder_pretrained=False,
    )

    B = 2
    x_t = torch.randn(B, 10, 4)
    t = torch.rand(B)
    image = torch.randn(B, 3, 224, 224)

    # 1) forward shape
    out = backbone(x_t, t, image)
    assert out.shape == (B, 10, 4), f"shape: {out.shape}"

    # 2) grad flow on trainable params (encoder frozen)
    out.sum().backward()
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    grads = [p.grad for p in trainable if p.grad is not None]
    assert len(grads) > 0 and all(g.abs().sum() > 0 for g in grads), "trainable grad 흐르지 않음"

    # encoder는 frozen이라 trainable param 없음
    enc_trainable = sum(p.numel() for p in backbone.image_encoder.parameters() if p.requires_grad)
    assert enc_trainable == 0, "encoder가 frozen 안 됨"

    # 3) t 변화에 민감
    with torch.no_grad():
        t_a = torch.zeros(B)
        t_b = torch.ones(B)
        out_a = backbone(x_t, t_a, image)
        out_b = backbone(x_t, t_b, image)
    assert not torch.allclose(out_a, out_b), "t 변화에 둔감"

    # 4) image 변화에 민감
    with torch.no_grad():
        out_c = backbone(x_t, t, torch.randn_like(image))
    assert not torch.allclose(out, out_c), "image 변화에 둔감"

    print("backbone sanity check 통과")
