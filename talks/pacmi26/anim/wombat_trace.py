"""Animated entropy trace for the PACMI '26 talk (recorded version).

Reads the artifact's entropy_trace_wombat.json and renders tokens landing one
at a time, coloured by the paper's fixed entropy bins, with the per-token
entropy trace drawing beneath them in sync. Ends on the ground-truth reveal.

The clip says nothing the paper's Figure 1 does not say; it only adds time.

Usage:  uv run python wombat_trace.py            # writes wombat_trace.mp4 + frames/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

HERE = Path(__file__).resolve().parent
TRACE = Path.home() / "projects/pacmi26-observability/data/entropy_trace_wombat.json"
OUT_MP4 = HERE / "wombat_trace.mp4"
FRAMES = HERE / "frames"

# Same edges as the artifact's fig1_entropy_trace.py
BIN_EDGES = [0.19, 0.46, 1.04, 1.56, 2.02, 3.15]
# Same palette as papers/pacmi26/epistemic_honest.tex (white = confident, deep red = uncertain)
BIN_BG = ["#ffffff", "#f7f7f7", "#d9d9d9", "#ffc7c7", "#ff7878", "#db3030", "#9e0000"]
BIN_FG = ["#000000", "#000000", "#000000", "#000000", "#ffffff", "#ffffff", "#ffffff"]

FPS = 30
SEC_PER_TOKEN = 0.22
HOLD_AFTER_TEXT = 1.2
REVEAL_SECONDS = 3.0
LEAD_IN = 1.0

CHAR_W = 0.0148          # axes-fraction width of one monospace character at the font size below
LINE_H = 0.19
WRAP_CHARS = 64
FONT_SIZE = 16


def entropy_bin(e: float) -> int:
    return sum(e >= edge for edge in BIN_EDGES)


def layout(tokens: list[dict]) -> list[tuple[float, float]]:
    """Greedy line wrap in character units; returns (x, y) of each token's left edge."""
    pos, x, line = [], 0, 0
    for t in tokens:
        n = len(t["token_text"])
        if x + n > WRAP_CHARS and x > 0:
            x, line = 0, line + 1
        pos.append((x * CHAR_W, 1.0 - line * LINE_H))
        x += n
    return pos


def main() -> None:
    d = json.loads(TRACE.read_text())
    tokens = [t for t in d["tokens"] if not t["token_text"].startswith("<|")]
    n = len(tokens)
    pos = layout(tokens)
    ent = [t["entropy"] for t in tokens]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#fafafa")
    ax_q = fig.add_axes((0.05, 0.86, 0.9, 0.1)); ax_q.axis("off")
    ax_t = fig.add_axes((0.05, 0.50, 0.9, 0.33)); ax_t.axis("off")
    ax_e = fig.add_axes((0.08, 0.12, 0.84, 0.34))
    ax_r = fig.add_axes((0.05, 0.0, 0.9, 0.08)); ax_r.axis("off")

    ax_q.text(0, 0.5, "What shape is wombat scat?", fontsize=22, weight="bold", va="center")
    ax_q.text(1, 0.5, "OLMo-3 7B Instruct", fontsize=13, color="#555", va="center", ha="right")

    ax_e.set_xlim(-0.5, n - 0.5)
    ax_e.set_ylim(0, max(ent) * 1.15)
    ax_e.set_ylabel("entropy (nats)", fontsize=12)
    ax_e.set_xlabel("token", fontsize=12)
    ax_e.set_xticks([])
    for edge in BIN_EDGES:
        ax_e.axhline(edge, color="#bbb", lw=0.6, ls=":")
    ax_e.spines[["top", "right"]].set_visible(False)
    bars = ax_e.bar(range(n), ent, color=[BIN_BG[entropy_bin(e)] for e in ent],
                    edgecolor="#333", lw=0.5)
    for b in bars:
        b.set_visible(False)

    texts = []
    for t, (x, y) in zip(tokens, pos):
        b = entropy_bin(t["entropy"])
        tx = ax_t.text(x, y, t["token_text"], fontsize=FONT_SIZE, family="monospace",
                       va="top", ha="left", color=BIN_FG[b],
                       bbox=dict(boxstyle="square,pad=0.15", fc=BIN_BG[b], ec="#999", lw=0.4))
        tx.set_visible(False)
        texts.append(tx)

    legend_y = 0.5
    ax_r.text(0, legend_y, "entropy:", fontsize=12, va="center", color="#333")
    for i, lab in enumerate(["low", "", "", "mid", "", "", "high"]):
        ax_r.text(0.09 + i * 0.045, legend_y, f" {lab} " if lab else "    ", fontsize=11,
                  family="monospace", va="center", color=BIN_FG[i],
                  bbox=dict(boxstyle="square,pad=0.2", fc=BIN_BG[i], ec="#999", lw=0.4))
    reveal = ax_t.text(0, 1.0 - 4.4 * LINE_H, "", fontsize=18, va="top", ha="left",
                       color="#9e0000", weight="bold")

    n_lead = int(LEAD_IN * FPS)
    n_tok = int(n * SEC_PER_TOKEN * FPS)
    n_hold = int(HOLD_AFTER_TEXT * FPS)
    n_rev = int(REVEAL_SECONDS * FPS)
    total = n_lead + n_tok + n_hold + n_rev

    def draw(frame: int) -> None:
        if frame < n_lead:
            shown = 0
        elif frame < n_lead + n_tok:
            shown = min(n, int((frame - n_lead) / (SEC_PER_TOKEN * FPS)) + 1)
        else:
            shown = n
        for i in range(n):
            texts[i].set_visible(i < shown)
            bars[i].set_visible(i < shown)
        if frame >= n_lead + n_tok + n_hold:
            reveal.set_text("Wombat scat is cube-shaped. This answer is a fabrication.")
        else:
            reveal.set_text("")

    FRAMES.mkdir(exist_ok=True)
    writer = FFMpegWriter(fps=FPS, bitrate=3000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    keyframes = {0, n_lead + int(0.3 * n_tok), n_lead + n_tok - 1, total - 1}
    with writer.saving(fig, str(OUT_MP4), dpi=100):
        for f in range(total):
            draw(f)
            writer.grab_frame()
            if f in keyframes:
                fig.savefig(FRAMES / f"frame_{f:04d}.png", dpi=100)
    print(f"wrote {OUT_MP4} ({total / FPS:.1f}s, {total} frames) and {len(keyframes)} stills")


if __name__ == "__main__":
    main()
