"""4 space별 native init box를 사각형으로 시각화 (canvas / wide view).

- S-E:  s ~ clip(N(0,1), ±3) → b = (s+3)/6
- S-R:  init은 signal과 동일 (model state = signal)
- C-R:  y ~ N(0,1) in chart psi → b = (y_pos, exp(y_size))
- Logit: y ~ N(0,1) → b = sigmoid(y)

Saves:
  outputs/init_viz/01_canvas_view.png
  outputs/init_viz/02_wide_view.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

from model.trajectory import chart_decode, logit_decode, signal_decode


N_SAMPLES = 6
N_BOXES = 10


def gen_signal_native(N):
    s = torch.randn(N, 4).clamp_(-3, 3)
    return signal_decode(s)            # b ∈ [0, 1]


def gen_chart_native(N):
    y = torch.randn(N, 4)
    return chart_decode(y)             # pos = y_pos (raw, ~[−3, 3]); size = exp(y_size)


def gen_logit_native(N):
    y = torch.randn(N, 4)
    return logit_decode(y)             # b = sigmoid(y) ∈ (0, 1)


SPACES = [
    ("S-E (Signal Eucl)\ns ~ clip(N(0,1), ±3)",        gen_signal_native, "tab:blue"),
    ("S-R (psi-traj + signal model)\nsame init as S-E", gen_signal_native, "tab:purple"),
    ("C-R (chart_native)\ny ~ N(0,1) in chart psi",     gen_chart_native,  "tab:olive"),
    ("Logit (013)\ny ~ N(0,1) in logit (4 dims)",       gen_logit_native,  "tab:orange"),
]


def collect(seed: int = 0):
    """Pre-generate samples shared across the two views."""
    torch.manual_seed(seed)
    out = []
    for title, gen, color in SPACES:
        s_list = [gen(N_BOXES) for _ in range(N_SAMPLES)]
        out.append((title, color, s_list))
    return out


def render(samples, view_lim, save_path: Path, title: str):
    fig, axes = plt.subplots(
        len(SPACES), N_SAMPLES,
        figsize=(2.5 * N_SAMPLES, 2.7 * len(SPACES)),
    )
    for row, (sp_title, color, s_list) in enumerate(samples):
        for col in range(N_SAMPLES):
            ax = axes[row, col]
            boxes = s_list[col].numpy()
            ax.set_xlim(*view_lim)
            ax.set_ylim(view_lim[1], view_lim[0])     # y top→bottom
            ax.set_aspect("equal")
            ax.set_facecolor("#f8f8f8")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(sp_title, fontsize=9)
            if row == 0:
                ax.set_title(f"#{col}", fontsize=9)

            # canvas frame [0,1]^2
            ax.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1, fill=False,
                edgecolor="black", linewidth=1.2,
            ))
            for cx, cy, w, h in boxes:
                x0 = cx - w / 2
                y0 = cy - h / 2
                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), w, h, fill=False,
                    edgecolor=color, linewidth=1.0,
                ))
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    out = Path("outputs/init_viz")
    samples = collect(seed=0)

    render(
        samples, view_lim=(-0.05, 1.05),
        save_path=out / "01_canvas_view.png",
        title=(f"Init boxes inside canvas [0,1]²  —  {N_BOXES} boxes per sample, "
               "native init per space"),
    )
    render(
        samples, view_lim=(-6, 6),
        save_path=out / "02_wide_view.png",
        title=("Same boxes, wide view [−6, 6]²  —  "
               "Signal/Logit stay near canvas; Chart escapes & oversizes"),
    )
