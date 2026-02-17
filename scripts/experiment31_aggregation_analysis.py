"""
Experiment 31 Aggregation Analysis: Architecture-Dependent Signal Selection

Analyzes whether different entropy aggregations (mean, max, std) provide
better discrimination for different model architectures. Key finding:
Llama-4-Maverick (128-expert MoE) jumps from AUC 0.651 (mean) to 0.899
(max), suggesting MoE routing smooths mean entropy but peak uncertainty
survives.

Also computes the coefficient of variation (CV) within responses as a
shape metric: retrieval (knowable) shows high CV (spiky), fabrication
(unknowable) shows lower CV (uniform).

Analyzes both API results (Experiment 31) and local results (Experiment 27)
for comparison.

Run: python scripts/experiment31_aggregation_analysis.py

Input files:
  - exp31_frontier_api_20260216_105455.csv (API run 1: 3 models)
  - exp31_frontier_api_20260216_135647.csv (API run 2: 3 models)
  - exp27_bounded_verification_20260206_205725.csv (local: 4 models)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

os.environ["PYTHONUNBUFFERED"] = "1"

KNOWABLE_CATEGORIES_API = {"control", "wombat"}


def analyze_api_models():
    """Analyze API experiment results."""
    print("=" * 80)
    print("API MODELS (Experiment 31, Together.ai, top-5 logprobs)")
    print("=" * 80)

    f1 = "exp31_frontier_api_20260216_105455.csv"
    f2 = "exp31_frontier_api_20260216_135647.csv"

    frames = []
    for f in [f1, f2]:
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
        else:
            print(f"WARNING: {f} not found, skipping.")

    if not frames:
        print("No API data found.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"] != "DeepSeek-V3"]  # 1-token limitation
    print(f"Loaded {len(df)} probes across {df['model'].nunique()} models")
    print(f"(DeepSeek-V3 excluded: 1-token logprob limitation)\n")

    _analyze_aggregation(df, "category", KNOWABLE_CATEGORIES_API, "model")


def analyze_local_models():
    """Analyze local experiment results."""
    print("\n" + "=" * 80)
    print("LOCAL MODELS (Experiment 27, full vocabulary logits)")
    print("=" * 80)

    f = "exp27_bounded_verification_20260206_205725.csv"
    if not os.path.exists(f):
        print(f"WARNING: {f} not found, skipping.")
        return

    df = pd.read_csv(f)
    print(f"Loaded {len(df)} probes across {df['family'].nunique()} families\n")

    _analyze_aggregation(df, "is_knowable", {True}, "family")


def _analyze_aggregation(df, label_col, knowable_vals, group_col):
    """Generic aggregation analysis for any dataset."""

    # --- AUC comparison ---
    print(
        f"{'Model':<25} {'AUC(mean)':>10} {'AUC(max)':>10} "
        f"{'AUC(std)':>10} {'Best':>8} {'Improvement':>12}"
    )
    print("-" * 80)

    for group in sorted(df[group_col].unique()):
        gdf = df[df[group_col] == group]

        if label_col == "is_knowable":
            labels = (~gdf["is_knowable"]).astype(int).values
        else:
            labels = (
                gdf["category"]
                .apply(lambda c: 0 if c in knowable_vals else 1)
                .values
            )

        metrics = {}
        for col, name in [
            ("mean_entropy", "mean"),
            ("max_entropy", "max"),
            ("entropy_std", "std"),
        ]:
            if col not in gdf.columns:
                continue
            try:
                metrics[name] = roc_auc_score(labels, gdf[col].values)
            except Exception:
                metrics[name] = float("nan")

        if not metrics:
            continue

        best_name = max(metrics, key=metrics.get)
        improvement = metrics[best_name] - metrics.get("mean", 0)

        cols = " ".join(f"{metrics.get(n, float('nan')):>10.3f}" for n in ["mean", "max", "std"])
        print(f"{group:<25} {cols} {best_name:>8} {improvement:>+12.3f}")

    # --- CV analysis ---
    print(f"\n{'Model':<25} {'Category':<15} {'Mean H':>8} {'Std H':>8} "
          f"{'Max H':>8} {'CV':>8} {'N':>5}")
    print("-" * 80)

    for group in sorted(df[group_col].unique()):
        gdf = df[df[group_col] == group]

        if label_col == "is_knowable":
            categories = sorted(gdf["category"].unique())
        else:
            categories = ["control", "wombat", "glavinsky", "westphalia", "private"]

        for cat in categories:
            if label_col == "is_knowable":
                cdf = gdf[gdf["category"] == cat]
            else:
                cdf = gdf[gdf["category"] == cat]

            if len(cdf) == 0:
                continue

            m = cdf["mean_entropy"].mean()
            s = cdf["entropy_std"].mean()
            mx = cdf["max_entropy"].mean()
            cv = (
                cdf["entropy_std"] / cdf["mean_entropy"].replace(0, np.nan)
            ).median()
            n = len(cdf)
            print(
                f"{group:<25} {cat:<15} {m:>8.4f} {s:>8.4f} "
                f"{mx:>8.4f} {cv:>8.2f} {n:>5}"
            )
        print()


def main():
    analyze_api_models()
    analyze_local_models()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings:

1. ARCHITECTURE-DEPENDENT AGGREGATION
   - Dense models (Mistral): entropy_std is best signal
   - MoE models (Llama-4-Maverick): max_entropy is best signal
   - Well-calibrated large models (Qwen3-235B): mean is sufficient
   - Llama-4-Maverick: AUC 0.651 (mean) -> 0.899 (max)

2. SHAPE DISTINCTION (CV)
   - Retrieval (knowable): high CV (spiky, mostly confident)
   - Fabrication (unknowable): lower CV (uniformly uncertain)
   - Universal across all models tested (local and API)

3. CONSTRAINT INTERACTION
   - Under full observation (local), mean is good enough for all architectures
   - Under constrained observation (top-5 API), aggregation choice is critical
   - The signal degradation from API constraints exposes architecture dependence

4. IMPLICATION FOR COMPOSED SYSTEMS
   - A composition gate using fixed aggregation will fail for some architectures
   - The gate must be architecture-aware (see papers/rlm/ for follow-on work)
""")


if __name__ == "__main__":
    main()
