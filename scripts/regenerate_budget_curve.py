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

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

styles = {
    "No judge":              {"color": "#888888", "marker": "s", "linestyle": "--", "linewidth": 1.5},
    "Text-guided (length)":  {"color": "#d62728", "marker": "o", "linestyle": "-", "linewidth": 2},
    "Tensor-guided":         {"color": "#1f77b4", "marker": "^", "linestyle": "-", "linewidth": 2},
    "Composed":              {"color": "#2ca02c", "marker": "D", "linestyle": "-", "linewidth": 2},
}

for label, values in data.items():
    s = styles[label]
    ax.plot(budgets, values, label=label, marker=s["marker"],
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            color=s["color"], markersize=8)

# Annotate the growing gap at each budget level
for i, b in enumerate(budgets):
    tensor_y = data["Tensor-guided"][i]
    length_y = data["Text-guided (length)"][i]
    gap = tensor_y - length_y
    mid_y = (tensor_y + length_y) / 2
    ax.annotate(f"+{gap:.1f}pp",
                xy=(b, mid_y), fontsize=8, color="#555555",
                ha="left", va="center",
                xytext=(b + 1.5, mid_y))

ax.set_xlabel("Verification Budget (%)", fontsize=11)
ax.set_ylabel("End-to-End Accuracy (%)", fontsize=11)
ax.set_xticks(budgets)
ax.set_xlim(5, 35)
ax.set_ylim(73, 95)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save to paper figures directory
outpath = "papers/sosp/figures/exp27_aggregate_budget_curve.pdf"
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved: {outpath}")

# Also save PNG for quick preview
outpath_png = "papers/sosp/figures/exp27_aggregate_budget_curve.png"
fig.savefig(outpath_png, bbox_inches="tight", dpi=150)
print(f"Saved: {outpath_png}")
