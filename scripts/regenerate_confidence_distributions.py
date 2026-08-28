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
TITLE_SIZE = 8
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7
SUPTITLE_SIZE = 9

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
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.75), sharex=True, sharey=True) # type: ignore

    # IBM Design Library colorblind-safe palette, matching the public artifact's
    # scripts/fig2_confidence_distributions.py so paper and artifact agree.
    # Hatching for greyscale distinction.
    color_know = "#fe6100"     # orange
    color_unknow = "#785ef0"   # purple

    for ax, (label, model_id) in zip(axes.flat, families.items()):
        know_conf = [float(r["self_report_confidence"]) for r in rows
                     if r["model_id"] == model_id and r["category"] == "knowable"]
        unknow_conf = [float(r["self_report_confidence"]) for r in rows
                       if r["model_id"] == model_id and r["category"] == "unknowable"]

        bins = np.linspace(0, 1, 20)
        ax.hist(know_conf, bins=bins, alpha=0.7, label="Knowable",
                color=color_know, edgecolor="black", linewidth=0.5,
                hatch="//", density=True)
        ax.hist(unknow_conf, bins=bins, alpha=0.7, label="Unknowable",
                color=color_unknow, edgecolor="black", linewidth=0.5,
                hatch="\\\\", density=True)

        ax.set_title(label, fontsize=TITLE_SIZE, fontweight="bold", pad=2)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        # Log scale: nearly all mass sits at the top of the confidence
        # range, so a linear density axis renders every other bar invisible.
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e2)

    # Legend above the panels, axis labels below: putting both at the
    # bottom overlaps them.
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=LEGEND_SIZE, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.supxlabel("Self-Reported Confidence", fontsize=LABEL_SIZE, y=0.005)
    fig.supylabel("Density (log)", fontsize=LABEL_SIZE, x=0.01)

    plt.tight_layout(rect=(0.03, 0.03, 1.0, 0.94), h_pad=0.6)

    for ext in ["pdf", "png"]:
        for dest in [
            f"papers/sosp/figures/exp27_confidence_distributions.{ext}",
            f"papers/pacmi26/figures/exp27_confidence_distributions.{ext}",
            f"arxiv/exp27_confidence_distributions.{ext}",
            f"exp27_confidence_distributions.{ext}",
        ]:
            fig.savefig(dest, dpi=150) # pyright: ignore[reportUnknownMemberType]
            print(f"Saved: {dest}")

    plt.close(fig)

if __name__ == "__main__":
    main()
