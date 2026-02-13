#!/usr/bin/env python3
"""Regenerate the budget curve figure with the fixed text baseline.

Reads the re-evaluation CSV and produces the updated figure for the paper.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

# Fixed data from re-evaluation with stratified evaluator
# Text-guided now uses hedging + length only (no self-report)
data = {
    "No judge":        [75.8, 75.8, 75.8],
    "Text-guided":     [76.1, 76.9, 78.5],
    "Tensor-guided":   [82.1, 87.5, 91.9],
    "Composed":        [80.5, 87.9, 92.5],
}

budgets = [10, 20, 30]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

styles = {
    "No judge":      {"color": "#888888", "marker": "s", "linestyle": "--", "linewidth": 1.5},
    "Text-guided":   {"color": "#d62728", "marker": "o", "linestyle": "-", "linewidth": 2},
    "Tensor-guided": {"color": "#1f77b4", "marker": "^", "linestyle": "-", "linewidth": 2},
    "Composed":      {"color": "#2ca02c", "marker": "D", "linestyle": "-", "linewidth": 2},
}

for label, values in data.items():
    s = styles[label]
    ax.plot(budgets, values, label=label, marker=s["marker"],
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            color=s["color"], markersize=8)

# Annotate the key comparison
ax.annotate("82.1%", xy=(10, 82.1), xytext=(13, 83.5),
            fontsize=9, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
ax.annotate("78.5%", xy=(30, 78.5), xytext=(26, 76.5),
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))

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
