#!/usr/bin/env python3
"""Regenerate the budget curve figure with the length-only text baseline.

Text baseline now uses response length alone, the strongest available
text-channel signal (measured AUC 0.45-0.67 per model on
exp27_bounded_verification; an earlier "0.85-0.97" note here did not
reproduce and is withdrawn). Previous versions used a
mixed judge (self-report + hedging + length); self-report contributed
little because it is compressed at the top of its scale (see issue #1 --
the earlier "inverted self-report" reading was a parse artifact).
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ACM's PDF checker rejects Type 3 fonts; matplotlib emits them by default.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Regenerated from the deterministic seeded simulation
# (experiment27_realistic_verification.py, seed=42, 1000 trials; text baseline =
# raw word count; ties broken at random within each trial, see the artifact's
# fig3_budget_curve.py v1.0.2 — np.argsort's default sort broke ties
# differently under numpy 1.x vs 2.x). These reproduce exactly from committed data — see the PACMI
# artifact (pacmi26-observability, REPRODUCTION.md). They supersede earlier
# hand-entered values that did not reproduce from any committed run (<1pp shift,
# no claim change).
data = {
    "No judge":                [75.8, 75.8, 75.8],
    "Text-guided (length)":    [78.4, 82.2, 87.6],
    "Tensor-guided":           [81.7, 86.8, 90.9],
    "Composed":                [81.5, 87.1, 91.5],
}

budgets = [10, 20, 30]

#FIG_W_SIZE, FIG_H_SIZE = 3.35, 2.25  # inches = \columnwidth in acmart sigplan
# Sized for the column at scale 1.0; see the note in
# regenerate_confidence_distributions.py on why bbox_inches="tight" is
# avoided. eval.tex includes this at width=\columnwidth.
FIG_W_SIZE=8
FIG_H_SIZE=4
LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 18, 15, 15
LINEWIDTH=1.5
MARKER_SIZE=5
fig, ax = plt.subplots(1, 1, figsize=(FIG_W_SIZE, FIG_H_SIZE)) # pyright: ignore[reportUnknownMemberType]

# IBM Design Library colorblind-safe palette, matching the public artifact's
# scripts/fig3_budget_curve.py so paper and artifact agree.
# Distinct markers + linestyles ensure greyscale readability
styles: dict[str, dict[str, object]] = {
    "No judge":              {"color": "gray", "marker": "s", "linestyle": "--", "linewidth": LINEWIDTH},
    "Text-guided (length)":  {"color": "#dc267f", "marker": "o", "linestyle": "-", "linewidth": LINEWIDTH},
    "Tensor-guided":         {"color": "#785ef0", "marker": "^", "linestyle": "-", "linewidth": LINEWIDTH},
    "Composed":              {"color": "#648fff", "marker": "D", "linestyle": "-.", "linewidth": LINEWIDTH},
}

for label, values in data.items():
    s = styles[label]
    ax.plot(budgets, values, label=label, marker=s["marker"], # pyright: ignore[reportUnknownMemberType]
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            color=s["color"], markersize=MARKER_SIZE)

# Annotate the growing gap at each budget level, offset above the higher
# line so the label doesn't collide with the Composed curve.
for i, b in enumerate(budgets):
    tensor_y = data["Tensor-guided"][i]
    length_y = data["Text-guided (length)"][i]
    composed_y = data["Composed"][i]
    gap = tensor_y - length_y
    top_y = max(tensor_y, composed_y)
    ax.annotate(f"+{gap:.1f}pp", # pyright: ignore[reportUnknownMemberType]
                xy=(b, top_y), fontsize=TICK_SIZE, color="#555555",
                ha="center", va="bottom",
                xytext=(b, top_y + 1.6))

ax.set_xlabel("Verification Budget (%)", fontsize=LABEL_SIZE) # pyright: ignore[reportUnknownMemberType]
ax.set_ylabel("End-to-End Accuracy (%)", fontsize=LABEL_SIZE) # pyright: ignore[reportUnknownMemberType]
ax.set_xticks(budgets) # pyright: ignore[reportUnknownMemberType]
ax.tick_params(axis="both", labelsize=TICK_SIZE) # pyright: ignore[reportUnknownMemberType]
ax.set_xlim(5, 35)
ax.set_ylim(73, 97)
# The "No judge" baseline is flat at 75.8, so an in-axes legend at lower
# right sits on top of it. Put the legend above the plot instead.
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, # pyright: ignore[reportUnknownMemberType]
          fontsize=LEGEND_SIZE, frameon=False, handlelength=1.8,
          columnspacing=1.2, borderaxespad=0.0)
ax.grid(True, alpha=0.3) # pyright: ignore[reportUnknownMemberType]

plt.tight_layout()

# Save to all figure locations
for ext in ["pdf", "png"]:
    # 2026-08-18: SOSP and arXiv prose/tables were reconciled to these
    # reproducible numbers as well; all four locations now carry the same
    # figure. The canonical generator is scripts/fig3_budget_curve.py in the
    # public artifact (fsgeek/pacmi26-observability, v1.0.1); this script is
    # kept as a convenience and mirrors its data.
    for dest in [
        f"papers/sosp/figures/exp27_aggregate_budget_curve.{ext}",
        f"papers/pacmi26/figures/exp27_aggregate_budget_curve.{ext}",
        f"arxiv/exp27_aggregate_budget_curve.{ext}",
        f"exp27_aggregate_budget_curve.{ext}",
    ]:
        fig.savefig(dest, dpi=150) # pyright: ignore[reportUnknownMemberType]
        print(f"Saved: {dest}")
