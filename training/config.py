"""Phase 1 training config (FM-only, signal flow)."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    data_root: str = "./data"

    # Model (DiTBackbone kwargs)
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 8
    n_queries: int = 10
    encoder_pretrained: bool = True
    encoder_freeze: bool = True

    # Optim
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    total_steps: int = 50000
    batch_size: int = 64

    # Inference / vis
    K: int = 16

    # Logging
    out_dir: str = "outputs"
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
