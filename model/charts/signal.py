"""Signal chart (Euclidean): box [0,1]^4 ↔ signal [-3,3]^4."""

import torch


def box_to_signal(b: torch.Tensor) -> torch.Tensor:
    """[0, 1] → [-3, 3]."""
    return 3.0 * (2.0 * b - 1.0)


def signal_to_box(s: torch.Tensor) -> torch.Tensor:
    """[-3, 3] → [0, 1]."""
    return (s / 3.0 + 1.0) / 2.0


if __name__ == "__main__":
    # 1) endpoints
    b_zero = torch.zeros(4)
    b_half = torch.full((4,), 0.5)
    b_one = torch.ones(4)
    assert torch.allclose(box_to_signal(b_zero), torch.full((4,), -3.0))
    assert torch.allclose(box_to_signal(b_half), torch.zeros(4))
    assert torch.allclose(box_to_signal(b_one), torch.full((4,), 3.0))

    s_neg = torch.full((4,), -3.0)
    s_zero = torch.zeros(4)
    s_pos = torch.full((4,), 3.0)
    assert torch.allclose(signal_to_box(s_neg), torch.zeros(4))
    assert torch.allclose(signal_to_box(s_zero), torch.full((4,), 0.5))
    assert torch.allclose(signal_to_box(s_pos), torch.ones(4))

    # 2) round-trip on random batched input
    torch.manual_seed(0)
    b = torch.rand(3, 10, 4)
    assert torch.allclose(signal_to_box(box_to_signal(b)), b, atol=1e-6)
    s = torch.empty(3, 10, 4).uniform_(-3, 3)
    assert torch.allclose(box_to_signal(signal_to_box(s)), s, atol=1e-6)

    # 3) shape / dtype / device preserved
    b32 = torch.rand(2, 4, dtype=torch.float32)
    out = box_to_signal(b32)
    assert out.shape == b32.shape and out.dtype == b32.dtype

    # 4) differentiable
    b_grad = torch.rand(4, requires_grad=True)
    s_grad = box_to_signal(b_grad)
    s_grad.sum().backward()
    assert b_grad.grad is not None and torch.allclose(b_grad.grad, torch.full((4,), 6.0))

    print("signal chart sanity check 통과")
