"""Phase 2 — N-way comparison GIF + per-index GIF + key-time snapshots.

For shared-init paired comparison: same b_0 is fed to every model.
Snapshots at t = 0.1, 0.5, 1.0 are saved as separate PNGs.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.mnist_box_dataset import MNISTBoxDataset
from model.trajectory import sample_init_box, signal_decode
from training.visualize import draw_trajectory_frames, save_gif

from .compare import cache_val_batches, load_model


SNAPSHOT_TIMES = (0.1, 0.5, 1.0)
GT_DASHED = True


def _label_panel(frame, text, color=(255, 255, 255)):
    """Big readable label at the top-left, with a darker background strip."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 22), (40, 40, 40), -1)
    cv2.putText(frame, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def _vsep(frame, x, color=(255, 255, 255)):
    cv2.line(frame, (x, 0), (x, frame.shape[0] - 1), color, 1)


def _snapshot_indices(K):
    """Return frame indices in traj (length K+1) closest to SNAPSHOT_TIMES."""
    return [int(round(t * K)) for t in SNAPSHOT_TIMES]


@torch.no_grad()
def make_n_way_gifs(ckpts, names, out_dir, *, n_samples=10, K=10,
                    seed=0, batch_size=10, val_root="./data",
                    init_prior: str = "default"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshots").mkdir(exist_ok=True)
    for name in names:
        (out_dir / "per_index" / _safe(name)).mkdir(parents=True, exist_ok=True)

    models = [load_model(c, device)[0] for c in ckpts]
    val_ds = Subset(MNISTBoxDataset(split="val", root=val_root),
                    list(range(n_samples)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = cache_val_batches(val_loader, device)

    snap_idx = _snapshot_indices(K)
    print(f"compare: {names} | K={K}, n_samples={n_samples}")
    print(f"snapshot frames: {snap_idx} (corresponding to t={SNAPSHOT_TIMES})")

    for batch_idx, batch in enumerate(batches):
        image = batch["image"]; gt = batch["gt_boxes"]
        B = image.shape[0]

        # Shared init b_0 across all models
        torch.manual_seed(seed + batch_idx)
        ref = torch.empty(B, 10, 4, device=image.device)
        init_box = sample_init_box(ref, prior=init_prior).clamp(0, 1)

        trajs = []
        for model in models:
            _, traj = model.sample(image, K=K, init_box=init_box)
            trajs.append(traj)

        for b in range(B):
            sample_idx = batch_idx * batch_size + b
            img = batch["image"][b]
            gt_b = gt[b].cpu()

            # Per-model frame sequences (224×224 each, K+1 frames)
            per_model_frames = []
            for name, traj in zip(names, trajs):
                t_seq = [t[b].cpu() for t in traj]
                frames = draw_trajectory_frames(
                    img, t_seq, gt_boxes=(gt_b if GT_DASHED else None)
                )
                # Label each frame with model name
                frames = [_label_panel(f.copy(), name) for f in frames]
                per_model_frames.append(frames)

            # N-panel horizontal GIF (224 × 224*N per frame)
            n = len(names)
            combined = []
            for k in range(K + 1):
                row = np.concatenate([per_model_frames[m][k] for m in range(n)], axis=1)
                # vertical separator lines between panels
                for m in range(1, n):
                    _vsep(row, m * per_model_frames[0][0].shape[1])
                combined.append(row)
            save_gif(combined, out_dir / "compare" / f"sample_{sample_idx:02d}.gif", fps=4)

            # Snapshots at t=0.1, 0.5, 1.0 — save individual PNG of N-panel frame
            for t_val, k in zip(SNAPSHOT_TIMES, snap_idx):
                snap = combined[k]
                snap_path = out_dir / "snapshots" / \
                    f"sample_{sample_idx:02d}_t{t_val:.1f}.png"
                cv2.imwrite(str(snap_path), snap)

        # Per-index GIFs and snapshots only for first sample, first batch
        if batch_idx == 0:
            img0 = batch["image"][0]; gt0 = gt[0].cpu()
            for name, traj in zip(names, trajs):
                t_seq = [t[0].cpu() for t in traj]
                model_dir = out_dir / "per_index" / _safe(name)
                for i in range(10):
                    frames_i = draw_trajectory_frames(
                        img0, [t[i:i+1] for t in t_seq],
                        gt_boxes=(gt0[i:i+1] if GT_DASHED else None),
                    )
                    save_gif(frames_i, model_dir / f"idx_{i}.gif", fps=4)
                    # snapshots per index per t
                    for t_val, k in zip(SNAPSHOT_TIMES, snap_idx):
                        cv2.imwrite(
                            str(model_dir / f"idx_{i}_t{t_val:.1f}.png"),
                            frames_i[k],
                        )

    print(f"saved compare GIFs   → {out_dir / 'compare'}")
    print(f"saved snapshots      → {out_dir / 'snapshots'} (per-sample N-panel)")
    print(f"saved per-index GIFs → {out_dir / 'per_index'}")


def _safe(name):
    """Make a name safe for use as a directory / filename."""
    return name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out-dir", default="outputs/comparison_4way/gifs")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--init-prior", choices=["default", "small_size"], default="default")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = _parse()
        assert len(args.ckpts) == len(args.names)
        make_n_way_gifs(args.ckpts, args.names, args.out_dir,
                        n_samples=args.n_samples, K=args.K, seed=args.seed,
                        batch_size=args.batch_size,
                        init_prior=args.init_prior)
    else:
        ckpt = "outputs/001_fullrun/ckpt/final.pt"
        if not Path(ckpt).exists():
            print(f"smoke skipped — {ckpt} not found")
            raise SystemExit
        make_n_way_gifs([ckpt, ckpt], ["A", "B"], "/tmp/gifs_smoke",
                        n_samples=2, K=10, batch_size=2)
        print("gifs smoke 통과")
