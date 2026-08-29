#!/usr/bin/env python3
"""Regenerate confidence distribution figure with colorblind-friendly palette.

Figure 1 in SOSP paper: Self-reported confidence distributions for
knowable vs unknowable queries across four model architectures.
"""

import matplotlib

matplotlib.use("Agg")
import csv

import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Sized for a two-column figure at \columnwidth = 241pt = 3.35in.
# bbox_inches="tight" is deliberately NOT used: it resizes the canvas in
# response to the font sizes, so compensating for the scale factor becomes
# a moving target. With figsize fixed at 3.35in the scale factor is 1.0 and
# these sizes are the rendered sizes. Body text is 10pt.
TITLE_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 15
SUPTITLE_SIZE = 10
FIG_W_SIZE=8
FIG_H_SIZE=4

DATA_FILE = "exp27_bounded_verification_20260206_205725.csv"

def load_data() -> list[dict[str, str]]:
    with open(DATA_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    return rows

def main():
    rows = load_data()

    families = {
        "OLMo": "allenai/olmo-3-7b-instruct",
        "Llama": "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen": "Qwen/Qwen3-4B-Instruct-2507",
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    }

    # Shared x/y labels and a single figure-level legend: at column width
    # there is no room to repeat them in each of the four panels. sharex/
    # sharey also removes three redundant tick label sets.
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_SIZE, FIG_H_SIZE), sharex=True, sharey=True) # type: ignore

    # IBM Design Library colorblind-safe palette, matching the public artifact's
    # scripts/fig2_confidence_distributions.py so paper and artifact agree.
    # Hatching for greyscale distinction.
    color_know = "#fe6100"     # orange
    color_unknow = "#785ef0"   # purple

    for ax, (label, model_id) in zip(axes.flat, families.items()):
        know_conf = np.array([float(r["self_report_confidence"]) for r in rows
                     if r["model_id"] == model_id and r["category"] == "knowable"])
        unknow_conf = np.array([float(r["self_report_confidence"]) for r in rows
                       if r["model_id"] == model_id and r["category"] == "unknowable"])

        # Empirical CDF
        know_sorted = np.sort(know_conf)
        unknow_sorted = np.sort(unknow_conf)

        know_cdf = np.arange(1, len(know_sorted) + 1) / len(know_sorted)
        unknow_cdf = np.arange(1, len(unknow_sorted) + 1) / len(unknow_sorted)

        ax.plot(know_sorted, know_cdf, label="Knowable", color=color_know, linewidth=2)

        ax.plot(unknow_sorted, unknow_cdf, label="Unknowable", color=color_unknow, linewidth=2)


        ax.set_title(label, fontsize=SUPTITLE_SIZE, fontweight="bold", pad=2)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)

    # Legend above the panels, axis labels below: putting both at the
    # bottom overlaps them.
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=LEGEND_SIZE, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.supxlabel("Self-Reported Confidence", fontsize=LABEL_SIZE)
    fig.supylabel("CDF", fontsize=LABEL_SIZE)

    fig.subplots_adjust(bottom=0.17, wspace=0.1)

    for ext in ["pdf", "png"]:
        for dest in [
            f"papers/sosp/figures/exp27_confidence_distributions.{ext}",
            f"papers/pacmi26/figures/exp27_confidence_distributions.{ext}",
            f"arxiv/exp27_confidence_distributions.{ext}",
            f"exp27_confidence_distributions.{ext}",
        ]:
            fig.savefig(dest, bbox_inches="tight", dpi=150) # pyright: ignore[reportUnknownMemberType]
            print(f"Saved: {dest}")

    plt.close(fig)

if __name__ == "__main__":
    main()
