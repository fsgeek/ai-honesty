#!/usr/bin/env python3
"""Regenerate the budget curve figure with the length-only text baseline.

Text baseline now uses response length alone (AUC 0.85-0.97 per model),
the strongest available text-channel signal. Previous versions used a
mixed judge (self-report + hedging + length) that was hobbled by the
inverted self-report signal.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

LEGEND_SIZE = 15
LABEL_SIZE = 18
TICK_SIZE = 15
FIG_W_SIZE = 8
FIG_H_SIZE = 4

# Data from consistent Monte Carlo simulation (1000 trials, seed=42)
# with stratified evaluator labels (75/80 human calibration, 93.8%)
# Text baseline uses raw word count (the strongest text-channel signal,
# per-model AUC 0.85-0.97), not the lossy length_score heuristic.
data = {
    "No judge":                [75.8, 75.8, 75.8],
    "Text-guided (length)":    [79.2, 82.8, 87.6],
    "Tensor-guided":           [81.7, 86.7, 90.2],
    "Composed":                [81.1, 87.7, 91.8],
}

budgets = [10, 20, 30]

fig, ax = plt.subplots(1, 1, figsize=(FIG_W_SIZE, FIG_H_SIZE))

# Colorblind-friendly: blue, orange, brown, gray — no red-green
# Distinct markers + linestyles ensure greyscale readability
styles = {
    "No judge":              {"color": "#888888", "marker": "s", "linestyle": "--", "linewidth": 1.5},
    "Text-guided (length)":  {"color": "#e66101", "marker": "o", "linestyle": "-", "linewidth": 2},
    "Tensor-guided":         {"color": "#2166ac", "marker": "^", "linestyle": "-", "linewidth": 2},
    "Composed":              {"color": "#5e3c99", "marker": "D", "linestyle": "-.", "linewidth": 2},
}

for label, values in data.items():
    s = styles[label]
    ax.plot(budgets, values, label=label, marker=s["marker"],
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            color=s["color"], markersize=8)

# Annotate the growing gap at each budget level, offset above the higher
# line so the label doesn't collide with the Composed curve.
for i, b in enumerate(budgets):
    tensor_y = data["Tensor-guided"][i]
    length_y = data["Text-guided (length)"][i]
    composed_y = data["Composed"][i]
    gap = tensor_y - length_y
    top_y = max(tensor_y, composed_y)
    ax.annotate(f"+{gap:.1f}pp",
                xy=(b, top_y), fontsize=TICK_SIZE - 3, color="#555555",
                ha="center", va="bottom",
                xytext=(b, top_y + 1.6))

ax.set_xlabel("Verification Budget (%)", fontsize=LABEL_SIZE)
ax.set_ylabel("End-to-End Accuracy (%)", fontsize=LABEL_SIZE)
ax.set_xticks(budgets)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.set_xlim(5, 35)
ax.set_ylim(73, 97)
ax.legend(loc="lower right", fontsize=LEGEND_SIZE)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save to all figure locations
for ext in ["pdf", "png"]:
    for dest in [
        f"papers/sosp/figures/exp27_aggregate_budget_curve.{ext}",
        f"papers/pacmi26/figures/exp27_aggregate_budget_curve.{ext}",
        f"arxiv/exp27_aggregate_budget_curve.{ext}",
        f"exp27_aggregate_budget_curve.{ext}",
    ]:
        fig.savefig(dest, bbox_inches="tight", dpi=150)
        print(f"Saved: {dest}")
