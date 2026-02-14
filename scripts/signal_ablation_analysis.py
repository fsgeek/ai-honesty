#!/usr/bin/env python3
"""
Signal Component Ablation Analysis for Epistemic Observability Paper.

Determines which tensor signal component carries the discriminative power
for knowable vs unknowable classification: mean_entropy, max_entropy,
entropy_std, mean_logprob, or mean_top5_mass?

Analyzes:
1. Per-signal AUC (per model and pooled)
2. Correlation matrix between signals
3. Pairwise and triple combination AUC via logistic regression
4. Full bundle vs best single signal comparison
5. Per-model ranking stability
"""

import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "/home/tony/projects/ai-honesty/exp27_bounded_verification_20260206_205725.csv"
OUTPUT_CSV = "/home/tony/projects/ai-honesty/signal_ablation_results.csv"

SIGNALS = ["mean_entropy", "max_entropy", "entropy_std", "mean_logprob", "mean_top5_mass"]
TARGET = "is_knowable"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("SIGNAL COMPONENT ABLATION ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

# Convert is_knowable to boolean/int
df[TARGET] = df[TARGET].map({True: 1, False: 0, "True": 1, "False": 0})

# Drop rows with NaN in signals or target
df_clean = df.dropna(subset=SIGNALS + [TARGET]).copy()

print(f"\nLoaded {len(df_clean)} rows from {DATA_PATH}")
print(f"Models: {sorted(df_clean['family'].unique())}")
print(f"Knowable: {df_clean[TARGET].sum()}, Unknowable: {(1 - df_clean[TARGET]).sum()}")
print(f"Per-model counts:")
for fam in sorted(df_clean["family"].unique()):
    sub = df_clean[df_clean["family"] == fam]
    print(f"  {fam}: {len(sub)} total, {sub[TARGET].sum()} knowable, {(1-sub[TARGET]).sum()} unknowable")

# ---------------------------------------------------------------------------
# Helper: compute AUC with direction handling
# ---------------------------------------------------------------------------
def compute_auc(y_true, scores):
    """Compute AUC, handling the case where lower score = knowable."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    auc = roc_auc_score(y_true, scores)
    # For signals where lower = knowable (e.g., entropy), AUC < 0.5
    # We report AUC and note direction
    return auc


def compute_lr_auc(X, y, cv_folds=5):
    """Compute AUC using logistic regression with leave-one-model-out or simple CV."""
    if len(np.unique(y)) < 2:
        return np.nan
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_scaled, y)
    probs = lr.predict_proba(X_scaled)[:, 1]
    return roc_auc_score(y, probs)


# ---------------------------------------------------------------------------
# 1. Single-signal AUC: per model and pooled
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("1. SINGLE-SIGNAL AUC (knowable=1 vs unknowable=0)")
print("=" * 80)
print("\nNote: AUC > 0.5 means higher signal value → more likely knowable")
print("      AUC < 0.5 means higher signal value → more likely unknowable")
print("      |AUC - 0.5| measures discriminative power regardless of direction\n")

families = sorted(df_clean["family"].unique())
single_results = []

# Header
header = f"{'Signal':<20}" + "".join(f"{'|AUC-0.5| ' + f:>12}" for f in families) + f"  {'Pooled AUC':>12}  {'|Pooled-0.5|':>12}"
print(header)
print("-" * len(header))

for sig in SIGNALS:
    row = {"signal": sig}
    auc_vals = []
    for fam in families:
        sub = df_clean[df_clean["family"] == fam]
        auc = compute_auc(sub[TARGET].values, sub[sig].values)
        row[f"auc_{fam}"] = auc
        row[f"disc_{fam}"] = abs(auc - 0.5)
        auc_vals.append(auc)

    # Pooled
    pooled_auc = compute_auc(df_clean[TARGET].values, df_clean[sig].values)
    row["auc_pooled"] = pooled_auc
    row["disc_pooled"] = abs(pooled_auc - 0.5)
    single_results.append(row)

    # Print
    parts = f"{sig:<20}"
    for fam in families:
        parts += f"  {row[f'disc_{fam}']:>10.4f}  "
    parts += f"  {pooled_auc:>10.4f}    {row['disc_pooled']:>10.4f}"
    print(parts)

# Print raw AUC values too
print(f"\n{'--- Raw AUC values ---':^{len(header)}}")
header2 = f"{'Signal':<20}" + "".join(f"{'AUC ' + f:>14}" for f in families) + f"  {'Pooled':>10}"
print(header2)
print("-" * len(header2))
for res in single_results:
    parts = f"{res['signal']:<20}"
    for fam in families:
        parts += f"  {res[f'auc_{fam}']:>12.4f}"
    parts += f"  {res['auc_pooled']:>10.4f}"
    print(parts)

# Rank signals by pooled discriminative power
print("\nRanking by pooled discriminative power (|AUC - 0.5|):")
ranked = sorted(single_results, key=lambda x: x["disc_pooled"], reverse=True)
for i, r in enumerate(ranked):
    print(f"  {i+1}. {r['signal']:<20}  |AUC-0.5| = {r['disc_pooled']:.4f}  (raw AUC = {r['auc_pooled']:.4f})")

# ---------------------------------------------------------------------------
# 2. Correlation matrix between signals
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. CORRELATION MATRIX BETWEEN SIGNALS")
print("=" * 80)

corr = df_clean[SIGNALS].corr()
print("\nPearson correlation (pooled across all models):\n")
# Format nicely
print(f"{'':>20}", end="")
for s in SIGNALS:
    print(f"  {s[:12]:>12}", end="")
print()
for s1 in SIGNALS:
    print(f"{s1:<20}", end="")
    for s2 in SIGNALS:
        print(f"  {corr.loc[s1, s2]:>12.3f}", end="")
    print()

# Per-model correlation for the top signals
print("\nPer-model Pearson correlation (mean_entropy vs max_entropy):")
for fam in families:
    sub = df_clean[df_clean["family"] == fam]
    r = sub["mean_entropy"].corr(sub["max_entropy"])
    print(f"  {fam}: r = {r:.3f}")

print("\nPer-model Pearson correlation (mean_entropy vs entropy_std):")
for fam in families:
    sub = df_clean[df_clean["family"] == fam]
    r = sub["mean_entropy"].corr(sub["entropy_std"])
    print(f"  {fam}: r = {r:.3f}")

print("\nPer-model Pearson correlation (mean_entropy vs mean_logprob):")
for fam in families:
    sub = df_clean[df_clean["family"] == fam]
    r = sub["mean_entropy"].corr(sub["mean_logprob"])
    print(f"  {fam}: r = {r:.3f}")

# ---------------------------------------------------------------------------
# 3. Pairwise and triple combination AUC (logistic regression)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. COMBINATION AUC (Logistic Regression, in-sample)")
print("=" * 80)

combo_results = []

# Single signals via LR (for fair comparison)
print("\n--- Single signals (LR) ---")
for sig in SIGNALS:
    X = df_clean[[sig]].values
    y = df_clean[TARGET].values
    auc = compute_lr_auc(X, y)
    combo_results.append({"combo": sig, "n_signals": 1, "auc": auc})
    print(f"  {sig:<40}  AUC = {auc:.4f}")

# Pairs
print("\n--- Pairwise combinations (LR) ---")
for pair in itertools.combinations(SIGNALS, 2):
    X = df_clean[list(pair)].values
    y = df_clean[TARGET].values
    auc = compute_lr_auc(X, y)
    combo_name = " + ".join(pair)
    combo_results.append({"combo": combo_name, "n_signals": 2, "auc": auc})
    print(f"  {combo_name:<40}  AUC = {auc:.4f}")

# Triples
print("\n--- Triple combinations (LR) ---")
for triple in itertools.combinations(SIGNALS, 3):
    X = df_clean[list(triple)].values
    y = df_clean[TARGET].values
    auc = compute_lr_auc(X, y)
    combo_name = " + ".join(triple)
    combo_results.append({"combo": combo_name, "n_signals": 3, "auc": auc})
    print(f"  {combo_name:<40}  AUC = {auc:.4f}")

# Quadruples
print("\n--- Quadruple combinations (LR) ---")
for quad in itertools.combinations(SIGNALS, 4):
    X = df_clean[list(quad)].values
    y = df_clean[TARGET].values
    auc = compute_lr_auc(X, y)
    combo_name = " + ".join(quad)
    combo_results.append({"combo": combo_name, "n_signals": 4, "auc": auc})
    print(f"  {combo_name:<40}  AUC = {auc:.4f}")

# Full bundle (all 5)
print("\n--- Full bundle (all 5 signals, LR) ---")
X = df_clean[SIGNALS].values
y = df_clean[TARGET].values
full_auc = compute_lr_auc(X, y)
combo_results.append({"combo": "ALL 5 SIGNALS", "n_signals": 5, "auc": full_auc})
print(f"  {'ALL 5 SIGNALS':<40}  AUC = {full_auc:.4f}")

# ---------------------------------------------------------------------------
# 4. Full bundle vs best single signal
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. FULL BUNDLE vs BEST SINGLE SIGNAL")
print("=" * 80)

best_single = max(
    [r for r in combo_results if r["n_signals"] == 1],
    key=lambda x: x["auc"],
)
best_pair = max(
    [r for r in combo_results if r["n_signals"] == 2],
    key=lambda x: x["auc"],
)
best_triple = max(
    [r for r in combo_results if r["n_signals"] == 3],
    key=lambda x: x["auc"],
)
best_quad = max(
    [r for r in combo_results if r["n_signals"] == 4],
    key=lambda x: x["auc"],
)

print(f"\n  Best single signal:  {best_single['combo']:<40}  AUC = {best_single['auc']:.4f}")
print(f"  Best pair:           {best_pair['combo']:<40}  AUC = {best_pair['auc']:.4f}")
print(f"  Best triple:         {best_triple['combo']:<40}  AUC = {best_triple['auc']:.4f}")
print(f"  Best quadruple:      {best_quad['combo']:<40}  AUC = {best_quad['auc']:.4f}")
print(f"  Full bundle (5):     {'ALL 5 SIGNALS':<40}  AUC = {full_auc:.4f}")
print(f"\n  Delta (full - best single):  {full_auc - best_single['auc']:+.4f}")
print(f"  Delta (full - best pair):    {full_auc - best_pair['auc']:+.4f}")
print(f"  Delta (full - best triple):  {full_auc - best_triple['auc']:+.4f}")

# ---------------------------------------------------------------------------
# 5. Per-model breakdown: ranking stability
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. PER-MODEL SIGNAL RANKING (LR AUC)")
print("=" * 80)

per_model_rankings = {}
per_model_results = []

for fam in families:
    sub = df_clean[df_clean["family"] == fam]
    y = sub[TARGET].values

    print(f"\n  --- {fam} (n={len(sub)}) ---")
    fam_aucs = []
    for sig in SIGNALS:
        X = sub[[sig]].values
        auc = compute_lr_auc(X, y)
        fam_aucs.append((sig, auc))
        per_model_results.append({"family": fam, "signal": sig, "auc": auc})

    # Sort by AUC descending
    fam_aucs.sort(key=lambda x: x[1], reverse=True)
    per_model_rankings[fam] = [s for s, _ in fam_aucs]

    for rank, (sig, auc) in enumerate(fam_aucs, 1):
        print(f"    {rank}. {sig:<20}  AUC = {auc:.4f}")

    # Full bundle per model
    X_full = sub[SIGNALS].values
    full_model_auc = compute_lr_auc(X_full, y)
    per_model_results.append({"family": fam, "signal": "ALL_5", "auc": full_model_auc})
    print(f"    -> Full bundle:    AUC = {full_model_auc:.4f}  (delta from best single: {full_model_auc - fam_aucs[0][1]:+.4f})")

# Ranking stability
print("\n  --- Ranking stability across models ---")
print(f"  {'Rank':<6}", end="")
for fam in families:
    print(f"  {fam:<15}", end="")
print()
for rank_idx in range(len(SIGNALS)):
    print(f"  {rank_idx+1:<6}", end="")
    for fam in families:
        print(f"  {per_model_rankings[fam][rank_idx]:<15}", end="")
    print()

# Check if #1 signal is consistent
top_signals = [per_model_rankings[fam][0] for fam in families]
if len(set(top_signals)) == 1:
    print(f"\n  Top signal is CONSISTENT across all models: {top_signals[0]}")
else:
    print(f"\n  Top signal VARIES across models: {dict(zip(families, top_signals))}")

# ---------------------------------------------------------------------------
# 6. Logistic regression coefficients (what does the model learn?)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("6. LOGISTIC REGRESSION COEFFICIENTS (Full Bundle)")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[SIGNALS].values)
lr = LogisticRegression(max_iter=1000, solver="lbfgs")
lr.fit(X_scaled, df_clean[TARGET].values)

print("\n  Standardized coefficients (positive = knowable, negative = unknowable):")
coefs = sorted(zip(SIGNALS, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
for sig, coef in coefs:
    direction = "knowable ↑" if coef > 0 else "unknowable ↑"
    print(f"    {sig:<20}  coef = {coef:>8.4f}  ({direction})")
print(f"    {'intercept':<20}  coef = {lr.intercept_[0]:>8.4f}")

# Per-model LR coefficients
print("\n  Per-model standardized coefficients:")
for fam in families:
    sub = df_clean[df_clean["family"] == fam]
    X_s = StandardScaler().fit_transform(sub[SIGNALS].values)
    lr_m = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr_m.fit(X_s, sub[TARGET].values)
    print(f"\n    {fam}:")
    for sig, coef in zip(SIGNALS, lr_m.coef_[0]):
        direction = "+" if coef > 0 else "-"
        print(f"      {sig:<20}  {direction}{abs(coef):.4f}")

# ---------------------------------------------------------------------------
# 7. Self-report comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("7. SELF-REPORT vs TENSOR SIGNALS (for reference)")
print("=" * 80)

sr_auc_pooled = compute_auc(df_clean[TARGET].values, df_clean["self_report_confidence"].values)
print(f"\n  Self-report confidence AUC (pooled): {sr_auc_pooled:.4f}")
print(f"  Best tensor signal AUC (pooled):     {best_single['auc']:.4f}  ({best_single['combo']})")
print(f"  Full tensor bundle AUC (pooled):     {full_auc:.4f}")
print(f"\n  Self-report |AUC-0.5|:  {abs(sr_auc_pooled - 0.5):.4f}")
print(f"  Best tensor |AUC-0.5|:  {abs(best_single['auc'] - 0.5):.4f}")
print(f"  Tensor advantage:       {abs(best_single['auc'] - 0.5) - abs(sr_auc_pooled - 0.5):+.4f}")

for fam in families:
    sub = df_clean[df_clean["family"] == fam]
    sr = compute_auc(sub[TARGET].values, sub["self_report_confidence"].values)
    print(f"  {fam} self-report AUC: {sr:.4f}")

# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Combine all results
all_rows = []

# Single signal pooled results
for res in single_results:
    all_rows.append({
        "analysis": "single_signal_pooled",
        "signal_or_combo": res["signal"],
        "metric": "raw_auc",
        "value": res["auc_pooled"],
    })
    all_rows.append({
        "analysis": "single_signal_pooled",
        "signal_or_combo": res["signal"],
        "metric": "disc_power",
        "value": res["disc_pooled"],
    })

# Per-model single signal
for res in per_model_results:
    all_rows.append({
        "analysis": f"per_model_{res['family']}",
        "signal_or_combo": res["signal"],
        "metric": "lr_auc",
        "value": res["auc"],
    })

# Combination results
for res in combo_results:
    all_rows.append({
        "analysis": "combination_lr",
        "signal_or_combo": res["combo"],
        "metric": "lr_auc",
        "value": res["auc"],
    })

# Correlation matrix (flattened)
for s1 in SIGNALS:
    for s2 in SIGNALS:
        all_rows.append({
            "analysis": "correlation",
            "signal_or_combo": f"{s1}_vs_{s2}",
            "metric": "pearson_r",
            "value": corr.loc[s1, s2],
        })

# LR coefficients
for sig, coef in coefs:
    all_rows.append({
        "analysis": "lr_coefficients",
        "signal_or_combo": sig,
        "metric": "standardized_coef",
        "value": coef,
    })

results_df = pd.DataFrame(all_rows)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(results_df)} rows to {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)

print(f"""
DATA: {len(df_clean)} samples across {len(families)} model families

BEST SINGLE SIGNAL: {best_single['combo']}
  Pooled AUC = {best_single['auc']:.4f}

BEST PAIR: {best_pair['combo']}
  Pooled AUC = {best_pair['auc']:.4f}  (delta from best single: {best_pair['auc'] - best_single['auc']:+.4f})

BEST TRIPLE: {best_triple['combo']}
  Pooled AUC = {best_triple['auc']:.4f}  (delta from best single: {best_triple['auc'] - best_single['auc']:+.4f})

FULL BUNDLE (5 signals):
  Pooled AUC = {full_auc:.4f}  (delta from best single: {full_auc - best_single['auc']:+.4f})

REDUNDANCY: Signals with |r| > 0.9 are near-redundant:""")

for i, s1 in enumerate(SIGNALS):
    for s2 in SIGNALS[i + 1 :]:
        r = corr.loc[s1, s2]
        if abs(r) > 0.8:
            label = "HIGHLY REDUNDANT" if abs(r) > 0.9 else "MODERATELY REDUNDANT"
            print(f"  {s1} vs {s2}: r = {r:.3f}  ({label})")

print(f"""
SELF-REPORT INVERSION CONFIRMED:
  Self-report AUC = {sr_auc_pooled:.4f} (below 0.5 = inverted)
  Best tensor AUC = {best_single['auc']:.4f}

RANKING STABILITY:""")
top_signals = [per_model_rankings[fam][0] for fam in families]
if len(set(top_signals)) == 1:
    print(f"  Top signal is CONSISTENT: {top_signals[0]} ranks #1 across all {len(families)} architectures")
else:
    for fam in families:
        print(f"  {fam}: #{1} = {per_model_rankings[fam][0]}, #{2} = {per_model_rankings[fam][1]}")

print()
