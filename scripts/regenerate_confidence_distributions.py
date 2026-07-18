#!/usr/bin/env python3
"""Regenerate confidence distribution figure with colorblind-friendly palette.

Figure 1 in SOSP paper: Self-reported confidence distributions for
knowable vs unknowable queries across four model architectures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

TITLE_SIZE = 15
LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 11
SUPTITLE_SIZE = 16

DATA_FILE = "exp27_bounded_verification_20260206_205725.csv"

def load_data():
    rows = []
    with open(DATA_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    rows = load_data()

    families = {
        "OLMo": "allenai/olmo-3-7b-instruct",
        "Llama": "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen": "Qwen/Qwen3-4B-Instruct-2507",
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Self-Reported Confidence: Knowable vs Unknowable", fontsize=SUPTITLE_SIZE, fontweight="bold")

    # Colorblind-friendly: blue for knowable, orange for unknowable
    # Hatching for greyscale distinction
    color_know = "#2166ac"
    color_unknow = "#e66101"

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

        ax.set_title(label, fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Self-Reported Confidence", fontsize=LABEL_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        for dest in [
            f"papers/sosp/figures/exp27_confidence_distributions.{ext}",
            f"papers/pacmi26/figures/exp27_confidence_distributions.{ext}",
            f"arxiv/exp27_confidence_distributions.{ext}",
            f"exp27_confidence_distributions.{ext}",
        ]:
            fig.savefig(dest, bbox_inches="tight", dpi=150)
            print(f"Saved: {dest}")

    plt.close(fig)

if __name__ == "__main__":
    main()
