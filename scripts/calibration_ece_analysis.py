#!/usr/bin/env python3
"""
Expected Calibration Error (ECE) analysis for epistemic observability signals.

Determines whether entropy, self-report confidence, and length signals are
well-calibrated as uncertainty estimates. A signal can have high AUC (good ranking)
but poor calibration (predicted probability doesn't match observed frequency).

Well-calibrated signals allow setting production decision thresholds;
poorly-calibrated ones only allow ranking.

Outputs:
  - calibration_ece_summary.csv: ECE + Brier score per signal per model + pooled
  - calibration_reliability_data.csv: Bin-level data for reliability diagrams
"""

import sys
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "exp27_bounded_verification_20260206_205725.csv"
N_BINS = 10
OUTPUT_SUMMARY = "calibration_ece_summary.csv"
OUTPUT_RELIABILITY = "calibration_reliability_data.csv"

# ── Load data ──────────────────────────────────────────────────────────────
print("=" * 72)
print("CALIBRATION ANALYSIS: Expected Calibration Error (ECE)")
print("=" * 72)
print()

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")
print(f"  Models: {sorted(df['family'].unique())}")
print(f"  Knowable: {df['is_knowable'].sum()}, Unknowable: {(~df['is_knowable']).sum()}")
print()

# ── Signal transformations ─────────────────────────────────────────────────
# Each signal must be transformed to P(knowable) in [0, 1].
#
# mean_entropy: higher entropy → MORE uncertain → LESS likely knowable
#   Transform: 1 - (entropy / max_entropy) → P(knowable)
#   But raw entropy range varies by model. Use min-max normalization per model,
#   then invert. Also do a pooled version with global normalization.
#
# self_report_confidence: The model's stated confidence. KNOWN to be INVERTED
#   (models report higher confidence on fabrications). We'll compute ECE both
#   raw (treating it as P(knowable)) and inverted (1 - conf as P(knowable)).
#
# length_score: Already in [0, 1]. Higher length_score → shorter response.
#   Short responses correlate with knowable (terse factual answers).
#   Treat length_score directly as P(knowable).

def normalize_signal(values, invert=False):
    """Min-max normalize to [0, 1], optionally invert."""
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.full_like(values, 0.5, dtype=float)
    normed = (values - vmin) / (vmax - vmin)
    if invert:
        normed = 1.0 - normed
    return normed


def compute_ece(y_true, y_prob, n_bins=N_BINS):
    """
    Compute Expected Calibration Error.

    ECE = sum over bins of (|bin_size/N| * |avg_predicted - avg_observed|)
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge for last bin
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        bin_count = mask.sum()
        if bin_count == 0:
            bin_data.append({
                'bin_lo': lo, 'bin_hi': hi, 'bin_mid': (lo + hi) / 2,
                'bin_count': 0, 'avg_predicted': np.nan,
                'avg_observed': np.nan, 'bin_ece_contrib': 0.0
            })
            continue

        avg_predicted = y_prob[mask].mean()
        avg_observed = y_true[mask].mean()
        weight = bin_count / len(y_true)
        contrib = weight * abs(avg_predicted - avg_observed)
        ece += contrib

        bin_data.append({
            'bin_lo': lo, 'bin_hi': hi, 'bin_mid': (lo + hi) / 2,
            'bin_count': int(bin_count),
            'avg_predicted': avg_predicted,
            'avg_observed': avg_observed,
            'bin_ece_contrib': contrib
        })

    return ece, bin_data


def compute_metrics(y_true, y_prob, n_bins=N_BINS):
    """Compute ECE, Brier score, and AUC for a signal."""
    # Clip to [0,1] for safety
    y_prob_clipped = np.clip(y_prob, 0.0, 1.0)

    ece, bin_data = compute_ece(y_true, y_prob_clipped, n_bins)
    brier = brier_score_loss(y_true, y_prob_clipped)

    # AUC (handles edge cases)
    try:
        auc = roc_auc_score(y_true, y_prob_clipped)
    except ValueError:
        auc = np.nan

    # Also get sklearn's calibration_curve for comparison
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob_clipped, n_bins=n_bins, strategy='uniform'
        )
    except ValueError:
        fraction_of_positives = np.array([])
        mean_predicted_value = np.array([])

    return {
        'ece': ece,
        'brier': brier,
        'auc': auc,
        'bin_data': bin_data,
        'sklearn_frac_pos': fraction_of_positives,
        'sklearn_mean_pred': mean_predicted_value,
    }


# ── Build signal dictionary ───────────────────────────────────────────────
# y_true: is_knowable (True=1, False=0)
y_true_all = df['is_knowable'].astype(int).values

signals = {}

# 1. Entropy → P(knowable): normalize then invert (high entropy → low P(knowable))
#    Global normalization (pooled)
signals['entropy_global'] = normalize_signal(df['mean_entropy'].values, invert=True)

# 2. Entropy → P(knowable): per-model normalization then invert
entropy_per_model = np.zeros(len(df))
for fam in df['family'].unique():
    mask = df['family'] == fam
    entropy_per_model[mask] = normalize_signal(
        df.loc[mask, 'mean_entropy'].values, invert=True
    )
signals['entropy_per_model'] = entropy_per_model

# 3. Self-report confidence as P(knowable) — raw
#    Known to be INVERTED per the project findings, so this should be poorly calibrated
signals['self_report_raw'] = df['self_report_confidence'].values

# 4. Self-report confidence inverted: 1 - conf as P(knowable)
signals['self_report_inverted'] = 1.0 - df['self_report_confidence'].values

# 5. Length score as P(knowable) — higher length_score = shorter response
signals['length_score'] = df['length_score'].values

# 6. Max entropy (global normalization, inverted)
signals['max_entropy_global'] = normalize_signal(df['max_entropy'].values, invert=True)

# 7. Entropy std (global normalization, inverted)
signals['entropy_std_global'] = normalize_signal(df['entropy_std'].values, invert=True)

# ── Compute metrics per model and pooled ──────────────────────────────────
summary_rows = []
reliability_rows = []

model_groups = list(df['family'].unique()) + ['POOLED']

for model_name in model_groups:
    if model_name == 'POOLED':
        mask = np.ones(len(df), dtype=bool)
    else:
        mask = (df['family'] == model_name).values

    y_true = y_true_all[mask]
    n = mask.sum()

    print(f"\n{'─' * 72}")
    print(f"Model: {model_name}  (n={n}, knowable={y_true.sum()}, unknowable={n - y_true.sum()})")
    print(f"{'─' * 72}")

    for sig_name, sig_values_all in signals.items():
        sig_values = sig_values_all[mask]

        results = compute_metrics(y_true, sig_values)

        summary_rows.append({
            'model': model_name,
            'signal': sig_name,
            'n': int(n),
            'ece': results['ece'],
            'brier': results['brier'],
            'auc': results['auc'],
        })

        # Store bin-level reliability data
        for bd in results['bin_data']:
            reliability_rows.append({
                'model': model_name,
                'signal': sig_name,
                **bd
            })

        print(f"  {sig_name:25s}  ECE={results['ece']:.4f}  Brier={results['brier']:.4f}  AUC={results['auc']:.4f}")

# ── Summary table ──────────────────────────────────────────────────────────
print("\n")
print("=" * 72)
print("SUMMARY TABLE")
print("=" * 72)

summary_df = pd.DataFrame(summary_rows)
reliability_df = pd.DataFrame(reliability_rows)

# Pivot for readability
for metric in ['ece', 'brier', 'auc']:
    print(f"\n{'─' * 40}")
    print(f"  {metric.upper()}")
    print(f"{'─' * 40}")
    pivot = summary_df.pivot(index='signal', columns='model', values=metric)
    # Reorder columns
    col_order = [c for c in ['OLMo', 'Llama', 'Qwen', 'Mistral', 'POOLED'] if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string(float_format='{:.4f}'.format))

# ── Interpretation ─────────────────────────────────────────────────────────
print("\n")
print("=" * 72)
print("INTERPRETATION")
print("=" * 72)

pooled = summary_df[summary_df['model'] == 'POOLED'].set_index('signal')

print("\nPooled ECE comparison (lower is better calibrated):")
ece_sorted = pooled['ece'].sort_values()
for sig, val in ece_sorted.items():
    marker = " ***" if val == ece_sorted.min() else ""
    print(f"  {sig:25s}  ECE = {val:.4f}{marker}")

print("\nPooled Brier score comparison (lower is better):")
brier_sorted = pooled['brier'].sort_values()
for sig, val in brier_sorted.items():
    marker = " ***" if val == brier_sorted.min() else ""
    print(f"  {sig:25s}  Brier = {val:.4f}{marker}")

print("\nPooled AUC comparison (higher is better for ranking):")
auc_sorted = pooled['auc'].sort_values(ascending=False)
for sig, val in auc_sorted.items():
    marker = " ***" if val == auc_sorted.max() else ""
    print(f"  {sig:25s}  AUC = {val:.4f}{marker}")

# Key insight: compare ranking ability (AUC) vs calibration (ECE)
best_auc_sig = pooled['auc'].idxmax()
best_ece_sig = pooled['ece'].idxmin()

print(f"\nKey finding:")
print(f"  Best ranking signal (AUC):       {best_auc_sig} (AUC={pooled.loc[best_auc_sig, 'auc']:.4f})")
print(f"  Best calibrated signal (ECE):    {best_ece_sig} (ECE={pooled.loc[best_ece_sig, 'ece']:.4f})")

if best_auc_sig != best_ece_sig:
    print(f"  --> Ranking and calibration disagree: high AUC does not imply good calibration.")
else:
    print(f"  --> Same signal wins both: good ranking AND good calibration.")

# Check if entropy is well-calibrated enough for threshold-setting
entropy_ece = pooled.loc['entropy_global', 'ece']
print(f"\n  Entropy (global) ECE = {entropy_ece:.4f}")
if entropy_ece < 0.05:
    print("  --> Excellent calibration. Entropy values can be used directly as probability thresholds.")
elif entropy_ece < 0.10:
    print("  --> Good calibration. Entropy values are reasonably usable as probability thresholds.")
elif entropy_ece < 0.15:
    print("  --> Moderate calibration. Entropy provides directional thresholds but would benefit from recalibration (e.g., Platt scaling).")
else:
    print("  --> Poor calibration. Entropy is useful for ranking (AUC) but NOT for direct probability thresholds.")
    print("     Production use requires post-hoc recalibration (Platt scaling or isotonic regression).")

# ── Per-model consistency ──────────────────────────────────────────────────
print("\n")
print("=" * 72)
print("PER-MODEL ECE CONSISTENCY")
print("=" * 72)

for sig_name in signals:
    per_model = summary_df[(summary_df['signal'] == sig_name) & (summary_df['model'] != 'POOLED')]
    eces = per_model['ece'].values
    print(f"  {sig_name:25s}  mean={eces.mean():.4f}  std={eces.std():.4f}  range=[{eces.min():.4f}, {eces.max():.4f}]")

# ── Reliability diagram detail for entropy_global (POOLED) ─────────────────
print("\n")
print("=" * 72)
print("RELIABILITY DIAGRAM DATA: entropy_global (POOLED)")
print("=" * 72)

entropy_pooled_bins = reliability_df[
    (reliability_df['model'] == 'POOLED') &
    (reliability_df['signal'] == 'entropy_global')
]
print(f"{'Bin':>8s}  {'Count':>6s}  {'Predicted':>10s}  {'Observed':>10s}  {'Gap':>8s}")
print(f"{'─' * 50}")
for _, row in entropy_pooled_bins.iterrows():
    gap = abs(row['avg_predicted'] - row['avg_observed']) if not np.isnan(row['avg_predicted']) else 0
    pred_str = f"{row['avg_predicted']:.4f}" if not np.isnan(row['avg_predicted']) else "   N/A"
    obs_str = f"{row['avg_observed']:.4f}" if not np.isnan(row['avg_observed']) else "   N/A"
    gap_str = f"{gap:.4f}" if not np.isnan(row['avg_predicted']) else "   N/A"
    print(f"[{row['bin_lo']:.1f},{row['bin_hi']:.1f})  {int(row['bin_count']):>6d}  {pred_str:>10s}  {obs_str:>10s}  {gap_str:>8s}")

# ── Save outputs ───────────────────────────────────────────────────────────
summary_df.to_csv(OUTPUT_SUMMARY, index=False)
print(f"\nSaved summary to {OUTPUT_SUMMARY}")

reliability_df.to_csv(OUTPUT_RELIABILITY, index=False)
print(f"Saved reliability diagram data to {OUTPUT_RELIABILITY}")

# ── Paper-relevant one-liner ───────────────────────────────────────────────
print("\n")
print("=" * 72)
print("PAPER-RELEVANT SUMMARY")
print("=" * 72)
entropy_auc = pooled.loc['entropy_global', 'auc']
entropy_brier = pooled.loc['entropy_global', 'brier']
sr_raw_ece = pooled.loc['self_report_raw', 'ece']
sr_raw_auc = pooled.loc['self_report_raw', 'auc']
sr_inv_ece = pooled.loc['self_report_inverted', 'ece']
sr_inv_auc = pooled.loc['self_report_inverted', 'auc']
length_ece = pooled.loc['length_score', 'ece']
length_auc = pooled.loc['length_score', 'auc']

print(f"""
Tensor entropy (mean_entropy, globally normalized, inverted):
  AUC = {entropy_auc:.4f}  |  ECE = {entropy_ece:.4f}  |  Brier = {entropy_brier:.4f}

Self-report confidence (raw, treating stated confidence as P(knowable)):
  AUC = {sr_raw_auc:.4f}  |  ECE = {sr_raw_ece:.4f}  |  Brier = {brier_sorted.loc['self_report_raw']:.4f}
  NOTE: AUC < 0.5 confirms self-report inversion (models more confident on fabrications)

Self-report confidence (inverted, 1-conf as P(knowable)):
  AUC = {sr_inv_auc:.4f}  |  ECE = {sr_inv_ece:.4f}  |  Brier = {brier_sorted.loc['self_report_inverted']:.4f}

Length score (shorter = more likely knowable):
  AUC = {length_auc:.4f}  |  ECE = {length_ece:.4f}  |  Brier = {brier_sorted.loc['length_score']:.4f}

Key claim for paper:
  Tensor entropy achieves strong discrimination (AUC={entropy_auc:.3f}) but ECE={entropy_ece:.4f}
  indicates {'good' if entropy_ece < 0.10 else 'moderate' if entropy_ece < 0.15 else 'poor'} calibration.
  {'The signal is usable for threshold-based decisions.' if entropy_ece < 0.10 else 'The signal is best used for ranking/triage, not direct probability estimation. Post-hoc recalibration (Platt scaling) would be needed for production threshold-setting.'}
""")

print("Done.")
