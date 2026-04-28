"""CLI entry point for flow matching training (Euclidean | Riemannian)."""

import argparse

from .config import TrainConfig
from .trainer import train


def _parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Flow matching training")
    # Model
    p.add_argument("--model",
                   choices=["signal", "chart", "chart_native", "chart_linear",
                            "hybrid", "chart_boxloss", "local", "logit_native",
                            "corner_logit"],
                   default=cfg.model,
                   help="S-E, S-R, C-R, C-E, exp007 hybrid, exp008 chart_boxloss, "
                        "exp009 local, exp012 logit_native (symmetric logit chart), "
                        "exp014 corner_logit (left/top corner-logit, in-canvas decode)")
    p.add_argument("--wide-dataset", action="store_true",
                   help="Use wide-scale GT box distribution (17× size, plans/riem_strength.md exp 005)")
    p.add_argument("--init-prior", choices=["default", "small_size"],
                   default=cfg.init_prior,
                   help="b_0 prior. 'small_size': w,h ~ U[0.01, 0.05] (exp 010/011)")
    p.add_argument("--hidden-size", type=int, default=cfg.hidden_size)
    p.add_argument("--depth", type=int, default=cfg.depth)
    p.add_argument("--num-heads", type=int, default=cfg.num_heads)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Use random-init encoder instead of DINOv2 weights")
    # Optim / schedule
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--total-steps", type=int, default=cfg.total_steps)
    p.add_argument("--warmup-steps", type=int, default=cfg.warmup_steps)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    p.add_argument("--amp", choices=["fp32", "bf16"], default=cfg.amp_dtype)
    # Inference / vis
    p.add_argument("--K", type=int, default=cfg.K, dest="K")
    # Logging
    p.add_argument("--out-root", type=str, default=cfg.out_root,
                   help="Parent dir for auto-numbered run dirs (default: outputs)")
    p.add_argument("--run-name", type=str, default=cfg.run_name,
                   help="Run name suffix; final dir = out_root/{NNN:03d}_{run_name}")
    p.add_argument("--log-every", type=int, default=cfg.log_every)
    p.add_argument("--val-every", type=int, default=cfg.val_every)
    p.add_argument("--gif-every", type=int, default=cfg.gif_every)
    p.add_argument("--ckpt-every", type=int, default=cfg.ckpt_every)
    # Misc
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)

    args = p.parse_args()

    cfg.model = args.model
    cfg.wide_dataset = args.wide_dataset
    cfg.init_prior = args.init_prior
    cfg.hidden_size = args.hidden_size
    cfg.depth = args.depth
    cfg.num_heads = args.num_heads
    cfg.encoder_pretrained = not args.no_pretrained

    cfg.lr = args.lr
    cfg.total_steps = args.total_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.batch_size = args.batch_size
    cfg.grad_clip = args.grad_clip
    cfg.amp_dtype = args.amp

    cfg.K = args.K

    cfg.out_root = args.out_root
    cfg.run_name = args.run_name
    cfg.log_every = args.log_every
    cfg.val_every = args.val_every
    cfg.gif_every = args.gif_every
    cfg.ckpt_every = args.ckpt_every

    cfg.seed = args.seed
    cfg.num_workers = args.num_workers
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
