"""Animated observability gap for the PACMI '26 talk (recorded version).

Two world states, one query, one response string. The telemetry produced during
generation (entropy trace, log-probabilities, attention) travels with the
response toward the interface and falls away there; only the string reaches the
supervisor. This is the diagram from the deck's third slide, given time.

The clip is a rendering of the argument, not of data, so it is the one that
needs a human eye on it before it ships.

Usage:  uv run python observability_gap.py       # writes observability_gap.mp4 + frames/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT_MP4 = HERE / "observability_gap.mp4"
FRAMES = HERE / "frames"

FPS = 30
KNOWABLE = "#FE6100"
UNKNOWABLE = "#785EF0"
TELEMETRY = "#648FFF"
INK = "#222"

# timeline (seconds)
T_WORLDS = 0.0      # worlds appear
T_QUERY = 1.2       # query box appears, arrows from both worlds
T_GEN = 2.6         # response + telemetry chips appear at the model
T_TRAVEL = 4.0      # response and chips move toward the interface
T_DROP = 6.0        # chips fall away at the interface line
T_ARRIVE = 7.4      # string reaches the supervisor
T_CAPTION = 8.4     # caption
T_END = 11.0


def tint(hex_color: str, amount: float = 0.82) -> tuple[float, float, float]:
    """Mix a colour toward white; used instead of RGBA so patch alpha stays free for fading."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def box(ax, x, y, w, h, text, fc="#fff", ec=INK, fs=15, weight="normal", color=INK):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       fc=fc, ec=ec, lw=1.5)
    ax.add_patch(p)
    t = ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, color=color)
    return p, t


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def main() -> None:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#fafafa")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # static geometry
    WX_A, WX_B, WY = 4.0, 12.0, 7.6
    QX, QY = 8.0, 5.9
    MX, MY = 8.0, 4.3          # where generation happens
    IY = 2.9                    # interface y
    SX, SY = 8.0, 1.05          # supervisor
    RY_FINAL = 2.2              # where the string comes to rest, below the line

    # elements, all created then toggled
    wa = box(ax, WX_A, WY, 4.6, 1.1, "World $w_A$: Yamamoto (2021) exists", fc=tint(KNOWABLE), ec=KNOWABLE)
    wb = box(ax, WX_B, WY, 4.6, 1.1, "World $w_B$: no such study", fc=tint(UNKNOWABLE), ec=UNKNOWABLE)
    q = box(ax, QX, QY, 6.2, 1.0, 'query $q$: "What did Yamamoto (2021) conclude?"', fs=14)
    arr_a = FancyArrowPatch((WX_A + 0.8, WY - 0.6), (QX - 1.6, QY + 0.55), arrowstyle="-|>",
                            mutation_scale=18, color=KNOWABLE, lw=2)
    arr_b = FancyArrowPatch((WX_B - 0.8, WY - 0.6), (QX + 1.6, QY + 0.55), arrowstyle="-|>",
                            mutation_scale=18, color=UNKNOWABLE, lw=2)
    ax.add_patch(arr_a); ax.add_patch(arr_b)

    resp = box(ax, MX, MY, 4.4, 0.9, "response string $r$", fc="#eee", fs=15, weight="bold")
    chips = []
    for i, lab in enumerate(["entropy trace", "log-probs", "attention"]):
        cx = MX + 3.2 + i * 0.0
        cy = MY + 0.55 - i * 0.55
        chips.append(box(ax, cx, cy, 2.2, 0.42, lab, fc=tint(TELEMETRY, 0.7), ec=TELEMETRY, fs=11, color="#234"))

    iface_line = ax.plot([1.5, 14.5], [IY, IY], color=INK, lw=1.2, ls="--")[0]
    iface_lab = ax.text(1.6, IY + 0.22, "text interface", fontsize=12, va="bottom", ha="left", color="#555")
    sup = box(ax, SX, SY, 4.4, 0.9, "supervisor sees $(q, r)$", fs=15)
    caption = ax.text(8, 0.3, "", ha="center", va="center", fontsize=17, weight="bold", color=INK)

    everything = [*wa, *wb, *q, arr_a, arr_b, *resp, *[e for c in chips for e in c], iface_line, iface_lab, *sup, caption]

    resp_home = (MX, MY)
    chip_home = [(MX + 3.2, MY + 0.55 - i * 0.55) for i in range(3)]

    def move_box(b, x, y, w, h, alpha=1.0):
        p, t = b
        p.set_bounds(x - w / 2, y - h / 2, w, h)
        t.set_position((x, y))
        p.set_alpha(alpha); t.set_alpha(alpha)

    def draw(sec: float) -> None:
        for e in everything:
            e.set_visible(False)
        # worlds
        if sec >= T_WORLDS:
            for e in (*wa, *wb):
                e.set_visible(True)
        if sec >= T_QUERY:
            for e in (*q, arr_a, arr_b):
                e.set_visible(True)
        if sec >= T_GEN:
            for e in (*resp, *[e for c in chips for e in c]):
                e.set_visible(True)
            iface_line.set_visible(True); iface_lab.set_visible(True)
            sup[0].set_visible(True); sup[1].set_visible(True)
        # travel: response and chips descend toward the interface
        if T_TRAVEL <= sec:
            f = ease((sec - T_TRAVEL) / (T_DROP - T_TRAVEL))
            ry = resp_home[1] - f * (resp_home[1] - (IY + 0.55))
            move_box(resp, MX, ry, 4.4, 0.9)
            for i, c in enumerate(chips):
                cy = chip_home[i][1] - f * (chip_home[i][1] - (IY + 0.55 + 0.5 - i * 0.5))
                move_box(c, chip_home[i][0], cy, 2.2, 0.42)
        # drop: chips fall below the line and fade; string crosses to the supervisor
        if T_DROP <= sec:
            f = ease((sec - T_DROP) / (T_ARRIVE - T_DROP))
            for i, c in enumerate(chips):
                cy = (IY + 0.55 + 0.5 - i * 0.5) - f * 1.6
                cx = chip_home[i][0] + f * (1.5 + 0.4 * i)
                move_box(c, cx, cy, 2.2, 0.42, alpha=max(0.0, 1.0 - 1.15 * f))
            ry = (IY + 0.55) - f * ((IY + 0.55) - RY_FINAL)
            move_box(resp, MX, ry, 4.4, 0.9)
        if sec >= T_ARRIVE:
            move_box(resp, MX, RY_FINAL, 4.4, 0.9)
        if sec >= T_CAPTION:
            caption.set_visible(True)
            caption.set_text("Same string in both worlds. The signals that differed never crossed the line.")

    FRAMES.mkdir(exist_ok=True)
    total = int(T_END * FPS)
    keyframes = {int(T_GEN * FPS) + 5, int((T_DROP + T_ARRIVE) / 2 * FPS), total - 1}
    writer = FFMpegWriter(fps=FPS, bitrate=3000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(OUT_MP4), dpi=100):
        for fr in range(total):
            draw(fr / FPS)
            writer.grab_frame()
            if fr in keyframes:
                fig.savefig(FRAMES / f"gap_{fr:04d}.png", dpi=100)
    print(f"wrote {OUT_MP4} ({total / FPS:.1f}s, {total} frames)")


if __name__ == "__main__":
    main()
