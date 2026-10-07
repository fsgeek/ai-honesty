"""Cross-model entropy agreement heatmap (arXiv figure exp27_crossmodel_correlation).

Per-query Spearman correlation of mean entropy between the four local models.
The previous version of this figure had no committed script and its axis labels
were permuted: cells were in alphabetical order (Llama, Mistral, OLMo, Qwen)
but labelled OLMo, Llama, Qwen, Mistral (found in the Oct 6 2026 audit).
Rows and columns are now labelled from the same ordered list.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "exp27_bounded_verification_20260206_205725.csv"
OUT = ROOT / "arxiv" / "exp27_crossmodel_correlation"
ORDER = ["OLMo", "Llama", "Qwen", "Mistral"]


def main():
    df = pd.read_csv(CSV)
    wide = df.pivot_table(index="query", columns="family", values="mean_entropy")
    corr = wide[ORDER].corr(method="spearman").loc[ORDER, ORDER]
    off = corr.values[np.triu_indices(len(ORDER), 1)]
    print(corr.round(3))
    print(f"mean pairwise rho = {off.mean():.3f}")

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(ORDER)), ORDER)
    ax.set_yticks(range(len(ORDER)), ORDER)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.85 else "black", fontsize=10)
    ax.set_title("Cross-model entropy agreement\n(per-query Spearman correlation)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="Spearman")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200)


if __name__ == "__main__":
    main()
