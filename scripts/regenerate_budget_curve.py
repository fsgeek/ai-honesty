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

# Regenerated from the deterministic seeded simulation
# (experiment27_realistic_verification.py, seed=42, 1000 trials; text baseline =
# raw word count). These reproduce exactly from committed data — see the PACMI
# artifact (pacmi26-observability, REPRODUCTION.md). They supersede earlier
# hand-entered values that did not reproduce from any committed run (<1pp shift,
# no claim change).
data = {
    "No judge":                [75.8, 75.8, 75.8],
    "Text-guided (length)":    [78.5, 82.1, 87.5],
    "Tensor-guided":           [81.7, 86.8, 90.9],
    "Composed":                [80.2, 87.1, 91.5],
}

budgets = [10, 20, 30]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

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

# Save to all figure locations
for ext in ["pdf", "png"]:
    # Scoped to the PACMI paper: only its prose was reconciled to these
    # reproducible numbers. SOSP and arXiv still carry the earlier values and
    # must be updated together with their own prose/tables when next revised.
    for dest in [
        f"papers/pacmi26/figures/exp27_aggregate_budget_curve.{ext}",
    ]:
        fig.savefig(dest, bbox_inches="tight", dpi=150)
        print(f"Saved: {dest}")
