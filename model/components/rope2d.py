"""2D RoPE — image patch token에 적용되는 회전 위치 임베딩.

Last `head_dim`을 둘로 나눠 절반은 h축 회전, 절반은 w축 회전.
"""

import torch


def precompute_2d_rope_cis(
    head_dim: int,
    H: int,
    W: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """returns freqs_cis: (H*W, head_dim/2) complex (cos + i sin)."""
    assert head_dim % 4 == 0, f"head_dim({head_dim}) must be divisible by 4"
    half = head_dim // 2  # half for h-axis, half for w-axis
    n_freqs = half // 2   # each freq is applied to a 2-dim pair

    # log-spaced frequencies (1, 1/θ^(1/n), ..., 1/θ^((n-1)/n))
    freqs = 1.0 / (theta ** (torch.arange(n_freqs, device=device, dtype=torch.float32) / n_freqs))

    h_pos = torch.arange(H, device=device, dtype=torch.float32)
    w_pos = torch.arange(W, device=device, dtype=torch.float32)

    angles_h = torch.outer(h_pos, freqs)  # (H, n_freqs)
    angles_w = torch.outer(w_pos, freqs)  # (W, n_freqs)

    cis_h = torch.polar(torch.ones_like(angles_h), angles_h)  # (H, n_freqs)
    cis_w = torch.polar(torch.ones_like(angles_w), angles_w)  # (W, n_freqs)

    cis_h_grid = cis_h[:, None, :].expand(H, W, n_freqs)
    cis_w_grid = cis_w[None, :, :].expand(H, W, n_freqs)

    cis = torch.cat([cis_h_grid, cis_w_grid], dim=-1)  # (H, W, half)
    return cis.reshape(H * W, half)


def apply_rope2d(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """rotate x in-pair using freqs_cis.

    x:         (..., N, head_dim)
    freqs_cis: (N, head_dim/2) complex
    returns:   (..., N, head_dim)
    """
    *lead, N, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*lead, N, head_dim // 2, 2))
    out_complex = x_complex * freqs_cis
    out = torch.view_as_real(out_complex).reshape(*lead, N, head_dim)
    return out.type_as(x)


if __name__ == "__main__":
    head_dim = 32
    H, W = 4, 4
    N = H * W

    cis = precompute_2d_rope_cis(head_dim, H, W)
    assert cis.shape == (N, head_dim // 2), f"cis shape: {cis.shape}"
    assert cis.is_complex(), "cis must be complex"

    # 1) position (0,0) is identity (all angles zero → cis == 1+0j)
    cis_origin = cis[0]
    assert torch.allclose(cis_origin.real, torch.ones_like(cis_origin.real)), "origin real != 1"
    assert torch.allclose(cis_origin.imag, torch.zeros_like(cis_origin.imag)), "origin imag != 0"

    # 2) shape preserved
    B, n_heads = 2, 8
    x = torch.randn(B, n_heads, N, head_dim)
    x_rot = apply_rope2d(x, cis)
    assert x_rot.shape == x.shape

    # 3) norm preserved (rotation)
    assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-5), "norm 변화"

    # 4) origin row unchanged
    assert torch.allclose(x[..., 0, :], x_rot[..., 0, :], atol=1e-5), "origin row 변형됨"

    # 5) non-origin row changed
    assert not torch.allclose(x[..., 5, :], x_rot[..., 5, :]), "non-origin row 같음"

    # 6) RoPE relative-position invariant:
    # 같은 vector x를 모든 위치에 두고 rotate 했을 때,
    # 동일한 (Δh, Δw)를 갖는 위치쌍의 dot product는 같아야 한다.
    v = torch.randn(head_dim)
    v_grid = v.expand(N, head_dim).clone()           # (N, head_dim)
    v_rot = apply_rope2d(v_grid, cis)                # (N, head_dim)

    def at(h, w):
        return v_rot[h * W + w]

    # Δp = (1, 1)
    dot_a = (at(0, 0) * at(1, 1)).sum()
    dot_b = (at(0, 1) * at(1, 2)).sum()
    dot_c = (at(1, 0) * at(2, 1)).sum()
    assert torch.allclose(dot_a, dot_b, atol=1e-4), f"relative invariant 깨짐: {dot_a} vs {dot_b}"
    assert torch.allclose(dot_a, dot_c, atol=1e-4), f"relative invariant 깨짐: {dot_a} vs {dot_c}"

    # 7) different (Δh, Δw) → different dot product (axis sensitivity)
    dot_dh = (at(0, 0) * at(1, 0)).sum()  # Δp = (1, 0)
    dot_dw = (at(0, 0) * at(0, 1)).sum()  # Δp = (0, 1)
    assert not torch.allclose(dot_dh, dot_dw, atol=1e-3), "h-축과 w-축 구분 못 함"

    print("rope2d sanity check 통과")
