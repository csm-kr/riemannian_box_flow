"""Phase 2: signal/chart encoders + Euclidean/Riemannian middle trajectory.

See plans/active.md §A and plans/model.md §7.4.
"""

import torch


EPS = 1e-3


def signal_encode(b: torch.Tensor) -> torch.Tensor:
    """box [0,1]^4 → signal [-3, 3]^4. s = 6 b - 3."""
    return 6.0 * b - 3.0


def signal_decode(s: torch.Tensor) -> torch.Tensor:
    """signal → box. b = (s + 3) / 6."""
    return (s + 3.0) / 6.0


def chart_encode(b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """psi(b) = (cx, cy, log max(w, eps), log max(h, eps))."""
    pos = b[..., :2]
    siz = b[..., 2:].clamp_min(eps).log()
    return torch.cat([pos, siz], dim=-1)


def chart_decode(y: torch.Tensor) -> torch.Tensor:
    """psi_inv(y) = (y_cx, y_cy, exp(y_lw), exp(y_lh))."""
    pos = y[..., :2]
    siz = y[..., 2:].exp()
    return torch.cat([pos, siz], dim=-1)


def logit_encode(b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Symmetric chart: y_i = logit(b_i) = log(b_i / (1 - b_i)) for all 4 dims.

    All components (cx, cy, w, h) get the same multiplicative transform.
    eps clamp keeps b ∈ [eps, 1-eps] so logit stays finite.
    """
    b_c = b.clamp(min=eps, max=1.0 - eps)
    return (b_c / (1.0 - b_c)).log()


def logit_decode(y: torch.Tensor) -> torch.Tensor:
    """sigma(y) = 1 / (1 + exp(-y)). Inverse of logit_encode (modulo clamp)."""
    return torch.sigmoid(y)


def sample_init_box(reference: torch.Tensor, prior: str = "default") -> torch.Tensor:
    """Sample b_0 in box space. Used by both signal/chart-native models so the
    init prior is shared.

    prior:
      "default"    — s ~ clip(N(0,1), ±3); b = (s+3)/6 ∈ [0,1] (current Phase 1/2 default)
      "small_size" — pos same as default; w, h ~ U[0.01, 0.05] (exp 010/011)
    """
    if prior == "default":
        s = torch.randn_like(reference).clamp_(-3, 3)
        return (s + 3.0) / 6.0
    if prior == "small_size":
        s_pos = torch.randn_like(reference[..., :2]).clamp_(-3, 3)
        b_pos = (s_pos + 3.0) / 6.0
        b_size = torch.empty_like(reference[..., 2:]).uniform_(0.01, 0.05)
        return torch.cat([b_pos, b_size], dim=-1)
    raise ValueError(f"unknown init prior: {prior!r}")


def _broadcast_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape t (B,) to broadcast against ref (B, ...)."""
    extra = ref.dim() - 1
    return t.view(-1, *([1] * extra))


def euclidean_trajectory(b_0: torch.Tensor, b_1: torch.Tensor, t: torch.Tensor):
    """Linear interpolation in signal space.

    s_t = (1-t) s_0 + t s_1,  u = s_1 - s_0  (constant in t).
    Returns (s_t, u_target) with shape == b_0.shape.
    """
    s_0 = signal_encode(b_0)
    s_1 = signal_encode(b_1)
    t_b = _broadcast_t(t, b_0)
    s_t = (1 - t_b) * s_0 + t_b * s_1
    u = (s_1 - s_0).expand_as(s_t)
    return s_t, u


def riemannian_trajectory(b_0: torch.Tensor, b_1: torch.Tensor, t: torch.Tensor,
                          eps: float = EPS):
    """Chart-space straight line decoded into signal-space velocity.

    y_t = (1-t) y_0 + t y_1   (chart space)
    b_t = psi_inv(y_t)
    s_t = signal_encode(b_t)
    u_t = ds_t/dt  — analytical:
      pos: 6 (b_1[pos] - b_0[pos])      (linear in t)
      siz: 6 b_t[siz] (log w_1 - log w_0)
    """
    y_0 = chart_encode(b_0, eps=eps)
    y_1 = chart_encode(b_1, eps=eps)
    t_b = _broadcast_t(t, b_0)
    y_t = (1 - t_b) * y_0 + t_b * y_1
    b_t = chart_decode(y_t)
    s_t = signal_encode(b_t)

    pos_diff = (b_1[..., :2] - b_0[..., :2]).expand_as(b_t[..., :2])
    log_diff = y_1[..., 2:] - y_0[..., 2:]
    u_pos = 6.0 * pos_diff
    u_siz = 6.0 * b_t[..., 2:] * log_diff
    u = torch.cat([u_pos, u_siz], dim=-1)
    return s_t, u


# ---------------------------------------------------------------------------
# Sanity check (runs as `python -m model.trajectory`)
# ---------------------------------------------------------------------------

def _sanity():
    torch.manual_seed(0)
    B = 4

    # 1) signal round-trip
    b = torch.rand(B, 10, 4)
    assert torch.allclose(signal_decode(signal_encode(b)), b, atol=1e-6)
    s = torch.empty(B, 10, 4).uniform_(-3, 3)
    assert torch.allclose(signal_encode(signal_decode(s)), s, atol=1e-6)

    # 2) chart round-trip on size-only (positions are identity)
    b_safe = b.clone()
    b_safe[..., 2:].clamp_(min=EPS)              # ensure w, h >= eps
    y = chart_encode(b_safe)
    assert torch.allclose(chart_decode(y), b_safe, atol=1e-5)

    # 3) eps clamp: very small w, h still produce finite log
    b_tiny = torch.zeros(1, 1, 4)                # w=h=0
    y_tiny = chart_encode(b_tiny)
    assert torch.isfinite(y_tiny).all()
    # log(eps) component
    assert torch.allclose(y_tiny[..., 2], torch.tensor(float(torch.tensor(EPS).log())))

    # 4) shape check on trajectories
    b0 = torch.rand(B, 10, 4)
    b1 = torch.rand(B, 10, 4)
    b1[..., 2:].clamp_(min=EPS)
    b0[..., 2:].clamp_(min=EPS)
    t = torch.rand(B)
    s_t_e, u_e = euclidean_trajectory(b0, b1, t)
    s_t_r, u_r = riemannian_trajectory(b0, b1, t)
    for x in (s_t_e, u_e, s_t_r, u_r):
        assert x.shape == (B, 10, 4), x.shape

    # 5) boundary: t=0 / t=1 → s_t equals signal_encode(b_0)/b_1 in both
    t_zero = torch.zeros(B)
    t_one = torch.ones(B)
    s0 = signal_encode(b0)
    s1 = signal_encode(b1)
    s_t_e0, _ = euclidean_trajectory(b0, b1, t_zero)
    s_t_e1, _ = euclidean_trajectory(b0, b1, t_one)
    s_t_r0, _ = riemannian_trajectory(b0, b1, t_zero)
    s_t_r1, _ = riemannian_trajectory(b0, b1, t_one)
    assert torch.allclose(s_t_e0, s0, atol=1e-5)
    assert torch.allclose(s_t_e1, s1, atol=1e-5)
    assert torch.allclose(s_t_r0, s0, atol=1e-5)
    assert torch.allclose(s_t_r1, s1, atol=1e-5)

    # 6) position-component equality (Euclidean vs Riemannian)
    assert torch.allclose(u_e[..., :2], u_r[..., :2], atol=1e-5)

    # 7) numerical vs analytical u_t for Riemannian (size component)
    #    Use float64 to avoid float32 round-off dominating the central difference.
    b0d = b0.double(); b1d = b1.double()
    h = 1e-5
    t_mid = torch.full((B,), 0.5, dtype=torch.float64)
    s_p, _ = riemannian_trajectory(b0d, b1d, t_mid + h)
    s_m, _ = riemannian_trajectory(b0d, b1d, t_mid - h)
    u_num = (s_p - s_m) / (2 * h)
    _, u_ana = riemannian_trajectory(b0d, b1d, t_mid)
    assert torch.allclose(u_num, u_ana, atol=1e-6), \
        f"max diff = {(u_num - u_ana).abs().max().item():.3e}"

    # 8) Riemannian size velocity differs from Euclidean in general
    diff_siz = (u_e[..., 2:] - u_r[..., 2:]).abs().max().item()
    assert diff_siz > 1e-3, f"size velocity should differ; got {diff_siz}"

    # 9a) logit round-trip on safely-bounded box
    b_safe2 = torch.empty(B, 10, 4).uniform_(EPS, 1 - EPS)
    y_log = logit_encode(b_safe2)
    assert torch.allclose(logit_decode(y_log), b_safe2, atol=1e-5)
    # logit(0.5) = 0
    half = torch.full((1, 1, 4), 0.5)
    assert logit_encode(half).abs().max() < 1e-6
    # logit on extreme values stays finite via eps clamp
    extreme = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    y_ext = logit_encode(extreme)
    assert torch.isfinite(y_ext).all()

    # 9b) sample_init_box prior shapes / ranges
    ref = torch.empty(B, 10, 4)
    b_def = sample_init_box(ref, prior="default")
    assert b_def.shape == (B, 10, 4)
    assert (b_def >= 0).all() and (b_def <= 1).all()
    b_sm = sample_init_box(ref, prior="small_size")
    assert b_sm.shape == (B, 10, 4)
    assert (b_sm[..., :2] >= 0).all() and (b_sm[..., :2] <= 1).all()
    assert (b_sm[..., 2:] >= 0.01 - 1e-6).all() and (b_sm[..., 2:] <= 0.05 + 1e-6).all(), \
        f"size out of [0.01, 0.05]: min={b_sm[...,2:].min()}, max={b_sm[...,2:].max()}"
    try:
        sample_init_box(ref, prior="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown prior")

    print("model/trajectory sanity check 통과")


if __name__ == "__main__":
    _sanity()
