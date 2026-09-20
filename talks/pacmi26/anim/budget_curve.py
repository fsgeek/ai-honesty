"""Animated budget curve for the PACMI '26 talk (recorded version).

The three judge lines draw in order: no judge, text-guided (length), then
tensor-guided, with the pooled +pp margins appearing as the tensor line reaches
each budget. Values are parsed from scripts/regenerate_budget_curve.py so the
clip cannot drift from the paper's Figure 3. The composed (oracle) judge is
omitted here: one idea per clip. It remains in the static figure.

Usage:  uv run python budget_curve.py            # writes budget_curve.mp4 + frames/
"""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

HERE = Path(__file__).resolve().parent
PAPER_SCRIPT = HERE.parents[2] / "scripts" / "regenerate_budget_curve.py"
OUT_MP4 = HERE / "budget_curve.mp4"
FRAMES = HERE / "frames"

FPS = 30
SECONDS_PER_LINE = 1.8
PAUSE_BETWEEN = 0.5
LEAD_IN = 0.6
HOLD = 2.5

STYLES = {  # same IBM palette as the paper figure
    "No judge":             dict(color="gray", marker="s", ls="--"),
    "Text-guided (length)": dict(color="#dc267f", marker="o", ls="-"),
    "Tensor-guided":        dict(color="#785ef0", marker="^", ls="-"),
}


def load_paper_data() -> tuple[dict[str, list[float]], list[int]]:
    """Pull the `data` and `budgets` literals out of the paper's figure script."""
    tree = ast.parse(PAPER_SCRIPT.read_text())
    found: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("data", "budgets"):
                found[name] = ast.literal_eval(node.value)
    return found["data"], found["budgets"]  # type: ignore[return-value]


def main() -> None:
    data, budgets = load_paper_data()
    order = list(STYLES)
    x = np.array(budgets, dtype=float)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#fafafa")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.14)
    ax.set_xlim(5, 38)
    ax.set_ylim(72, 95)
    ax.set_xticks(budgets)
    ax.set_xlabel("Verification budget (% of outputs checked)", fontsize=16)
    ax.set_ylabel("End-to-end accuracy (%)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("What each verification budget buys", fontsize=22, weight="bold", x=0.10, ha="left")

    lines, marks, labels = {}, {}, {}
    for name in order:
        s = STYLES[name]
        (ln,) = ax.plot([], [], color=s["color"], ls=s["ls"], lw=2.5)
        mk = ax.scatter([], [], color=s["color"], marker=s["marker"], s=70, zorder=5)
        pretty = {"No judge": "No judge", "Text-guided (length)": "Text (length)",
                  "Tensor-guided": "Tensor (entropy)"}[name]
        lb = ax.text(31, 0, pretty, color=s["color"], fontsize=15, va="center", weight="bold")
        lb.set_visible(False)
        lines[name], marks[name], labels[name] = ln, mk, lb

    tensor = np.array(data["Tensor-guided"])
    text = np.array(data["Text-guided (length)"])
    margins = []
    for b, t, l in zip(budgets, tensor, text):
        m = ax.annotate(f"+{t - l:.1f} pp", xy=(b, t), xytext=(0, 18), textcoords="offset points",
                        ha="center", fontsize=16, color="#555", weight="bold")
        m.set_visible(False)
        margins.append(m)

    n_lead = int(LEAD_IN * FPS)
    n_line = int(SECONDS_PER_LINE * FPS)
    n_pause = int(PAUSE_BETWEEN * FPS)
    n_hold = int(HOLD * FPS)
    total = n_lead + len(order) * (n_line + n_pause) + n_hold

    def draw(frame: int) -> None:
        f = frame - n_lead
        for i, name in enumerate(order):
            start = i * (n_line + n_pause)
            y = np.array(data[name], dtype=float)
            if f < start:
                lines[name].set_data([], [])
                marks[name].set_offsets(np.empty((0, 2)))
                labels[name].set_visible(False)
                continue
            frac = min(1.0, (f - start) / n_line)
            # draw the polyline progressively along x
            x_end = x[0] + frac * (x[-1] - x[0])
            xs = np.concatenate([x[x <= x_end], [x_end]])
            ys = np.interp(xs, x, y)
            lines[name].set_data(xs, ys)
            reached = x <= x_end + 1e-9
            marks[name].set_offsets(np.column_stack([x[reached], y[reached]]))
            if frac >= 1.0:
                labels[name].set_position((31, y[-1]))
                labels[name].set_visible(True)
            if name == "Tensor-guided":
                for j, m in enumerate(margins):
                    m.set_visible(bool(reached[j]))

    FRAMES.mkdir(exist_ok=True)
    writer = FFMpegWriter(fps=FPS, bitrate=3000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    keyframes = {n_lead + n_line + n_pause + n_line // 2, total - 1}
    with writer.saving(fig, str(OUT_MP4), dpi=100):
        for fr in range(total):
            draw(fr)
            writer.grab_frame()
            if fr in keyframes:
                fig.savefig(FRAMES / f"budget_{fr:04d}.png", dpi=100)
    print(f"wrote {OUT_MP4} ({total / FPS:.1f}s, {total} frames)")


if __name__ == "__main__":
    main()
