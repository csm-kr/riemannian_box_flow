"""Phase 1 training config (FM-only, signal flow)."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    data_root: str = "./data"
    wide_dataset: bool = False  # plans/riem_strength.md exp 005: 17× size variation

    # Model
    model: str = "signal"  # "signal" (Euclidean) | "chart" (Riemannian)
    # DiTBackbone kwargs
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 8
    n_queries: int = 10
    encoder_pretrained: bool = True
    encoder_freeze: bool = True
    # Init prior for b_0 sampling — see model/trajectory.py:sample_init_box.
    # "default": Phase 1/2 standard. "small_size": exp 010/011 (w,h ~ U[0.01, 0.05]).
    init_prior: str = "default"

    # Optim
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    total_steps: int = 35000
    batch_size: int = 64

    # Inference / vis
    K: int = 16

    # Logging
    out_root: str = "outputs"
    run_name: str = "run"
    log_every: int = 100
    val_every: int = 500
    val_max_batches: int = 8
    gif_every: int = 1000
    ckpt_every: int = 5000

    # Misc
    seed: int = 0
    num_workers: int = 2
    device: str = "cuda"
    amp_dtype: str = "fp32"  # "fp32" | "bf16"
