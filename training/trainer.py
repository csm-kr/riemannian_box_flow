"""Phase 1 trainer: FM-only flow matching, signal space."""

import math
import re
import time
from pathlib import Path

import cv2
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Avoid /dev/shm exhaustion in Docker (default 64MB) when num_workers > 0
mp.set_sharing_strategy("file_system")

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.flow_chart import ChartFlowModel
from model.flow_chart_boxloss import ChartBoxLossFlowModel
from model.flow_chart_linear import ChartLinearFlowModel
from model.flow_chart_native import ChartNativeFlowModel
from model.flow_corner_logit import CornerLogitFlowModel
from model.flow_hybrid import HybridFlowModel
from model.flow_local import LocalChartFlowModel
from model.flow_logit_native import LogitNativeFlowModel
from model.flow_signal import SignalFlowModel

from .config import TrainConfig
from .visualize import draw_trajectory_frames, save_gif


_MODEL_REGISTRY = {
    "signal":         SignalFlowModel,        # S-E
    "chart":          ChartFlowModel,         # S-R
    "chart_native":   ChartNativeFlowModel,   # C-R
    "chart_linear":   ChartLinearFlowModel,   # C-E
    "hybrid":         HybridFlowModel,        # exp 007: signal pos + chart size
    "chart_boxloss":  ChartBoxLossFlowModel,  # exp 008: chart model + box-space loss
    "local":          LocalChartFlowModel,    # exp 009: scale-aware Local chart (body frame)
    "logit_native":   LogitNativeFlowModel,   # exp 012: symmetric logit chart on all 4 dims
    "corner_logit":   CornerLogitFlowModel,   # exp 014: left/top corner-logit (in-canvas decode)
}


_RUN_DIR_RE = re.compile(r"^(\d{3})_")


def allocate_run_dir(out_root, run_name: str) -> Path:
    """Return next outputs/{NNN:03d}_{run_name}/ path. Creates out_root if missing.

    Scans existing children of out_root matching NNN_* and uses MAX(NNN) + 1.
    Does not create the returned dir itself — caller is responsible.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for child in out_root.iterdir():
        if not child.is_dir():
            continue
        m = _RUN_DIR_RE.match(child.name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return out_root / f"{max_n + 1:03d}_{run_name}"


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
    train_ds = MNISTBoxDataset(split="train", root=cfg.data_root, wide=cfg.wide_dataset)
    val_ds = MNISTBoxDataset(split="val", root=cfg.data_root, wide=cfg.wide_dataset)
    # train: persistent workers — avoid spawning new workers each epoch (cycles via _cycle).
    # val:   num_workers=0 — _validate / _dump_gif call iter(val_loader) repeatedly,
    #        which under file_system sharing leaks /dev/shm tensors over thousands of steps.
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
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
def _dump_gif(
    model, val_loader, device, K: int, n_queries: int, gif_path: Path,
    writer: SummaryWriter | None = None, step: int = 0,
):
    model.eval()
    batch = next(iter(val_loader))
    image = batch["image"][:1].to(device, non_blocking=True)
    gt = batch["gt_boxes"][0].cpu()
    _, traj_boxes = model.sample(image, K=K, n_queries=n_queries)
    traj = [b.squeeze(0).cpu() for b in traj_boxes]
    frames = draw_trajectory_frames(batch["image"][0], traj, gt_boxes=gt)
    save_gif(frames, gif_path, fps=6)
    if writer is not None:
        # last frame as TB image (BGR uint8 HWC → RGB CHW)
        last_rgb = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
        writer.add_image("val/traj_last", last_rgb.transpose(2, 0, 1), step)
    model.train()


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}, model={cfg.model}, total_steps={cfg.total_steps}, "
          f"batch={cfg.batch_size}, hidden={cfg.hidden_size}, depth={cfg.depth}, "
          f"encoder_pretrained={cfg.encoder_pretrained}")

    if cfg.model not in _MODEL_REGISTRY:
        raise ValueError(f"unknown model: {cfg.model!r}, choose from {list(_MODEL_REGISTRY)}")
    model_cls = _MODEL_REGISTRY[cfg.model]
    model_kwargs = dict(
        hidden_size=cfg.hidden_size, depth=cfg.depth, num_heads=cfg.num_heads,
        n_queries=cfg.n_queries,
        encoder_pretrained=cfg.encoder_pretrained, encoder_freeze=cfg.encoder_freeze,
    )
    # init_prior is supported on signal + chart_native + logit_native + corner_logit.
    # Other models still use their hardcoded prior;
    # warn if a non-default prior is requested for them.
    if cfg.model in ("signal", "chart_native", "logit_native", "corner_logit"):
        model_kwargs["init_prior"] = cfg.init_prior
    elif cfg.init_prior != "default":
        raise ValueError(
            f"--init-prior={cfg.init_prior!r} only supported for "
            f"--model {{signal,chart_native,logit_native,corner_logit}}, got {cfg.model!r}"
        )
    model = model_cls(**model_kwargs).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
    )

    train_loader, val_loader = _make_loaders(cfg)
    train_iter = _cycle(train_loader)

    out_dir = allocate_run_dir(cfg.out_root, cfg.run_name)
    ckpt_dir = out_dir / "ckpt"
    gif_dir = out_dir / "gif"
    tb_dir = out_dir / "tb"
    log_path = out_dir / "train_log.txt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"run dir: {out_dir}")

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
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr_now, step)

        if step % cfg.val_every == 0:
            val_loss = _validate(model, val_loader, device, cfg.val_max_batches)
            log(f"[{step:6d}] val_loss={val_loss:.4f}")
            writer.add_scalar("val/loss", val_loss, step)

        if step % cfg.gif_every == 0:
            gif_path = gif_dir / f"step_{step:06d}.gif"
            _dump_gif(model, val_loader, device, cfg.K, cfg.n_queries, gif_path,
                      writer=writer, step=step)
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
    _dump_gif(model, val_loader, device, cfg.K, cfg.n_queries, final_gif,
              writer=writer, step=cfg.total_steps)
    final_ckpt = ckpt_dir / "final.pt"
    torch.save({
        "step": cfg.total_steps, "model": model.state_dict(),
        "cfg": cfg.__dict__,
    }, final_ckpt)
    log(f"DONE. final GIF: {final_gif}, final ckpt: {final_ckpt}")
    writer.close()
    log_f.close()
