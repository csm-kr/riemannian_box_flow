"""Phase 1 trainer: FM-only flow matching, signal space."""

import math
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Avoid /dev/shm exhaustion in Docker (default 64MB) when num_workers > 0
mp.set_sharing_strategy("file_system")

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.flow_signal import SignalFlowModel

from .config import TrainConfig
from .visualize import draw_trajectory_frames, save_gif


def _cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _make_loaders(cfg: TrainConfig):
    train_ds = MNISTBoxDataset(split="train", root=cfg.data_root)
    val_ds = MNISTBoxDataset(split="val", root=cfg.data_root)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def _validate(model, val_loader, device, max_batches: int) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        b_1 = batch["gt_boxes"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        loss, _ = model.fm_loss(b_1, image)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def _dump_gif(model, val_loader, device, K: int, n_queries: int, gif_path: Path):
    model.eval()
    batch = next(iter(val_loader))
    image = batch["image"][:1].to(device, non_blocking=True)
    gt = batch["gt_boxes"][0].cpu()
    _, traj_boxes = model.sample(image, K=K, n_queries=n_queries)
    traj = [b.squeeze(0).cpu() for b in traj_boxes]
    frames = draw_trajectory_frames(batch["image"][0], traj, gt_boxes=gt)
    save_gif(frames, gif_path, fps=6)
    model.train()


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}, total_steps={cfg.total_steps}, batch={cfg.batch_size}, "
          f"hidden={cfg.hidden_size}, depth={cfg.depth}, "
          f"encoder_pretrained={cfg.encoder_pretrained}")

    model = SignalFlowModel(
        hidden_size=cfg.hidden_size, depth=cfg.depth, num_heads=cfg.num_heads,
        n_queries=cfg.n_queries,
        encoder_pretrained=cfg.encoder_pretrained, encoder_freeze=cfg.encoder_freeze,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
    )

    train_loader, val_loader = _make_loaders(cfg)
    train_iter = _cycle(train_loader)

    out_dir = Path(cfg.out_dir)
    ckpt_dir = out_dir / "ckpt"
    gif_dir = out_dir / "gif"
    log_path = out_dir / "train_log.txt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.amp_dtype]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"

    log_f = open(log_path, "w")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    t0 = time.time()
    model.train()
    for step in range(1, cfg.total_steps + 1):
        batch = next(train_iter)
        b_1 = batch["gt_boxes"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)

        lr_now = _cosine_warmup_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.lr)
        for g in optim.param_groups:
            g["lr"] = lr_now

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, _ = model.fm_loss(b_1, image)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optim.step()

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            log(f"[{step:6d}/{cfg.total_steps}] loss={loss.item():.4f} "
                f"lr={lr_now:.2e} elapsed={elapsed:.1f}s")

        if step % cfg.val_every == 0:
            val_loss = _validate(model, val_loader, device, cfg.val_max_batches)
            log(f"[{step:6d}] val_loss={val_loss:.4f}")

        if step % cfg.gif_every == 0:
            gif_path = gif_dir / f"step_{step:06d}.gif"
            _dump_gif(model, val_loader, device, cfg.K, cfg.n_queries, gif_path)
            log(f"[{step:6d}] GIF dumped: {gif_path}")

        if step % cfg.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg.__dict__,
            }, ckpt_path)
            log(f"[{step:6d}] ckpt: {ckpt_path}")

    # Final GIF + ckpt
    final_gif = gif_dir / f"step_{cfg.total_steps:06d}_final.gif"
    _dump_gif(model, val_loader, device, cfg.K, cfg.n_queries, final_gif)
    final_ckpt = ckpt_dir / "final.pt"
    torch.save({
        "step": cfg.total_steps, "model": model.state_dict(),
        "cfg": cfg.__dict__,
    }, final_ckpt)
    log(f"DONE. final GIF: {final_gif}, final ckpt: {final_ckpt}")
    log_f.close()
