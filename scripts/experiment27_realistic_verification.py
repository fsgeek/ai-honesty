#!/usr/bin/env python3
"""Experiment 27 with realistic (non-oracle) verification.

The original evaluate_condition sets corrected[idx] = True for every verified
item -- an oracle that always succeeds. This script replaces that with
probabilistic verification calibrated to our empirical measurements:

  - Unknowable queries: verification catches fabrication with P = 0.938
    (from the 75/80 human calibration of the stratified evaluator)
  - Knowable queries: verification retrieves correct answer with P = 0.95
    (factual lookup is usually but not always successful)

Runs 1000 Monte Carlo trials per condition to get stable estimates with 95%
confidence intervals.

Key hypothesis: Under oracle verification, every judge benefits equally from
having its selected items corrected. Under realistic verification, the QUALITY
of selection matters -- tensor-guided selects items with higher entropy, which
correspond to more obvious fabrications that are easier to verify. The tensor
advantage should therefore INCREASE under realistic verification.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Judge implementations (identical to experiment27_reeval_no_selfreport.py)
# ============================================================================

def text_uncertainty_score_original(row):
    """Original text judge (includes self-report). For comparison only."""
    self_report_unc = 1.0 - row["self_report_confidence"]
    hedge = row["hedge_score"]
    length = row["length_score"]
    return 0.4 * self_report_unc + 0.4 * hedge + 0.2 * length


def text_uncertainty_score_fixed(row):
    """Fixed text judge: hedging + length only. No self-report."""
    hedge = row["hedge_score"]
    length = row["length_score"]
    return 0.5 * hedge + 0.5 * length


def tensor_uncertainty_score(row):
    """Tensor-based uncertainty score. Uses mean entropy."""
    return row["mean_entropy"]


def is_citation_query(query):
    """Detect whether a query is asking about a citation/paper."""
    citation_markers = [
        "summarize:", "summarize the key contribution",
        "summarize the main", "'", "NeurIPS", "ICML", "ICLR",
        "ACL", "EMNLP", "NAACL", "OSDI", "SOSP", "EuroSys",
        "AAAI", "AAMAS", "PODC",
    ]
    query_lower = query.lower()
    if query_lower.startswith("summarize:"):
        return True
    for marker in citation_markers:
        if marker.lower() in query_lower and "'" in query:
            return True
    return False


# ============================================================================
# Correctness checker (identical to original)
# ============================================================================

def check_answer_correct(row):
    """Simplified correctness check using the stratified evaluator logic."""
    response = str(row.get("response", "")).lower().strip()
    expected = str(row.get("expected_answer", "")).lower().strip()
    is_knowable = row["is_knowable"]

    if not is_knowable:
        refusal_markers = [
            "i don't know", "i cannot", "i'm not sure", "no information",
            "cannot verify", "don't have", "not aware", "no record",
            "i apologize", "i can't",
        ]
        refuses = any(m in response for m in refusal_markers)
        return refuses
    else:
        if not expected:
            return True
        expected_parts = expected.split("/") if "/" in expected else [expected]
        return any(part.strip() in response for part in expected_parts)


# ============================================================================
# Verification probabilities
# ============================================================================

# From 75/80 human calibration: stratified evaluator agrees with human on
# 93.8% of items. This is the probability that verification catches a
# fabricated answer (unknowable query where model didn't refuse).
P_VERIFY_UNKNOWABLE = 0.938

# Factual lookup for knowable queries: usually succeeds but not always.
# Some answers require nuanced retrieval (e.g., morphological variants,
# multi-part answers). 95% is a reasonable upper bound.
P_VERIFY_KNOWABLE = 0.95

# Monte Carlo configuration
N_TRIALS = 1000
RANDOM_SEED = 42


# ============================================================================
# Oracle evaluation (original behavior, for comparison)
# ============================================================================

def evaluate_condition_oracle(df, condition_key, score_fn, budget_frac,
                              use_citation=False):
    """Oracle evaluation: verification always succeeds. Original behavior."""
    n = len(df)
    budget = max(1, int(n * budget_frac))

    if condition_key == "no_judge":
        return df["is_correct"].mean()

    scores = df.apply(score_fn, axis=1).values

    if use_citation:
        citation_mask = df["is_citation"].values
        for i in range(n):
            if citation_mask[i]:
                scores[i] = 0.0 if df.iloc[i]["is_knowable"] else 1.0

    verify_indices = np.argsort(scores)[-budget:]
    corrected = df["is_correct"].values.copy()

    for idx in verify_indices:
        corrected[idx] = True  # Oracle: always succeeds

    return corrected.sum() / n


# ============================================================================
# Realistic evaluation (probabilistic verification)
# ============================================================================

def evaluate_condition_realistic(df, condition_key, score_fn, budget_frac,
                                 rng, use_citation=False):
    """Realistic evaluation: verification succeeds probabilistically.

    For each selected item:
      - Unknowable: verification catches fabrication with P = 0.938
      - Knowable: verification retrieves answer with P = 0.95

    An item that is already correct stays correct regardless of verification.
    Verification can only IMPROVE an incorrect answer, with the given probability.
    """
    n = len(df)
    budget = max(1, int(n * budget_frac))

    if condition_key == "no_judge":
        return df["is_correct"].mean()

    scores = df.apply(score_fn, axis=1).values

    if use_citation:
        citation_mask = df["is_citation"].values
        for i in range(n):
            if citation_mask[i]:
                scores[i] = 0.0 if df.iloc[i]["is_knowable"] else 1.0

    verify_indices = np.argsort(scores)[-budget:]
    corrected = df["is_correct"].values.copy()

    for idx in verify_indices:
        if corrected[idx]:
            # Already correct -- verification doesn't change it
            continue

        row = df.iloc[idx]
        if not row["is_knowable"]:
            # Unknowable query where model fabricated (didn't refuse)
            # Verification catches this with P_VERIFY_UNKNOWABLE
            if rng.random() < P_VERIFY_UNKNOWABLE:
                corrected[idx] = True
        else:
            # Knowable query where model got it wrong
            # Verification retrieves correct answer with P_VERIFY_KNOWABLE
            if rng.random() < P_VERIFY_KNOWABLE:
                corrected[idx] = True

    return corrected.sum() / n


def monte_carlo_evaluate(df, condition_key, score_fn, budget_frac,
                         use_citation=False, n_trials=N_TRIALS,
                         seed=RANDOM_SEED):
    """Run n_trials of realistic evaluation, return mean and 95% CI."""
    rng = np.random.default_rng(seed)
    accuracies = np.array([
        evaluate_condition_realistic(df, condition_key, score_fn, budget_frac,
                                     rng, use_citation)
        for _ in range(n_trials)
    ])
    mean_acc = accuracies.mean()
    ci_lo = np.percentile(accuracies, 2.5)
    ci_hi = np.percentile(accuracies, 97.5)
    return mean_acc, ci_lo, ci_hi


# ============================================================================
# Evaluation runners
# ============================================================================

def run_oracle_evaluation(df, text_fn, text_label):
    """Run oracle (original) evaluation."""
    budget_levels = [0.10, 0.20, 0.30]
    conditions = [
        ("No judge", "no_judge", None, False),
        (text_label, "text_guided", text_fn, False),
        ("Tensor-guided", "tensor_guided", tensor_uncertainty_score, False),
        ("Composed", "composed", tensor_uncertainty_score, True),
    ]

    results = []
    for condition_name, condition_key, score_fn, use_citation in conditions:
        for budget in budget_levels:
            accuracy = evaluate_condition_oracle(
                df, condition_key, score_fn, budget, use_citation
            )
            results.append({
                "condition": condition_name,
                "budget": budget,
                "accuracy": accuracy * 100,
            })

    return pd.DataFrame(results)


def run_realistic_evaluation(df, text_fn, text_label):
    """Run realistic (Monte Carlo) evaluation."""
    budget_levels = [0.10, 0.20, 0.30]
    conditions = [
        ("No judge", "no_judge", None, False),
        (text_label, "text_guided", text_fn, False),
        ("Tensor-guided", "tensor_guided", tensor_uncertainty_score, False),
        ("Composed", "composed", tensor_uncertainty_score, True),
    ]

    results = []
    for condition_name, condition_key, score_fn, use_citation in conditions:
        for budget in budget_levels:
            mean_acc, ci_lo, ci_hi = monte_carlo_evaluate(
                df, condition_key, score_fn, budget, use_citation
            )
            results.append({
                "condition": condition_name,
                "budget": budget,
                "accuracy": mean_acc * 100,
                "ci_lo": ci_lo * 100,
                "ci_hi": ci_hi * 100,
            })

    return pd.DataFrame(results)


# ============================================================================
# Output formatting
# ============================================================================

def print_oracle_table(eval_df, title):
    """Print oracle results in paper table format."""
    print(f"\n{'='*78}")
    print(title)
    print(f"{'='*78}")

    pivot = eval_df.pivot(index="condition", columns="budget", values="accuracy")
    order = [c for c in ["No judge", "Text-guided (fixed)",
                          "Tensor-guided", "Composed"] if c in pivot.index]
    pivot = pivot.reindex(order)

    print(f"{'Condition':<25s} {'10%':>10s} {'20%':>10s} {'30%':>10s}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for idx, row in pivot.iterrows():
        print(f"{idx:<25s} {row[0.10]:>10.1f} {row[0.20]:>10.1f} {row[0.30]:>10.1f}")


def print_realistic_table(eval_df, title):
    """Print realistic results with 95% confidence intervals."""
    print(f"\n{'='*78}")
    print(title)
    print(f"{'='*78}")

    # Build display strings
    rows_out = []
    order = ["No judge", "Text-guided (fixed)", "Tensor-guided", "Composed"]
    for cond in order:
        cdf = eval_df[eval_df["condition"] == cond]
        if cdf.empty:
            continue
        row_data = {"condition": cond}
        for budget in [0.10, 0.20, 0.30]:
            bdf = cdf[cdf["budget"] == budget]
            if bdf.empty:
                row_data[budget] = "---"
            else:
                r = bdf.iloc[0]
                row_data[budget] = f"{r['accuracy']:.1f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        rows_out.append(row_data)

    print(f"{'Condition':<25s} {'10%':>22s} {'20%':>22s} {'30%':>22s}")
    print(f"{'-'*25} {'-'*22} {'-'*22} {'-'*22}")
    for row in rows_out:
        print(f"{row['condition']:<25s} {row[0.10]:>22s} {row[0.20]:>22s} {row[0.30]:>22s}")


def print_comparison_table(oracle_df, realistic_df, title):
    """Side-by-side comparison of oracle vs realistic."""
    print(f"\n{'='*78}")
    print(title)
    print(f"{'='*78}")

    order = ["No judge", "Text-guided (fixed)", "Tensor-guided", "Composed"]

    print(f"{'Condition':<25s} {'Budget':>6s}  {'Oracle':>8s}  {'Realistic':>10s}  {'Delta':>8s}")
    print(f"{'-'*25} {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")

    for cond in order:
        for budget in [0.10, 0.20, 0.30]:
            odf = oracle_df[(oracle_df["condition"] == cond) &
                            (oracle_df["budget"] == budget)]
            rdf = realistic_df[(realistic_df["condition"] == cond) &
                               (realistic_df["budget"] == budget)]
            if odf.empty or rdf.empty:
                continue
            o_acc = odf.iloc[0]["accuracy"]
            r_acc = rdf.iloc[0]["accuracy"]
            delta = r_acc - o_acc
            bstr = f"{int(budget*100)}%"
            print(f"{cond:<25s} {bstr:>6s}  {o_acc:>8.1f}  {r_acc:>10.1f}  {delta:>+8.2f}")


def print_advantage_analysis(oracle_df, realistic_df):
    """Analyze how the tensor advantage changes under realistic verification."""
    print(f"\n{'='*78}")
    print("TENSOR ADVANTAGE ANALYSIS: Oracle vs Realistic")
    print(f"{'='*78}")
    print()
    print("Advantage = Tensor-guided accuracy - Text-guided accuracy")
    print()

    print(f"{'Budget':>8s}  {'Oracle':>12s}  {'Realistic':>12s}  {'Change':>12s}")
    print(f"{'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

    for budget in [0.10, 0.20, 0.30]:
        # Oracle
        o_tensor = oracle_df[(oracle_df["condition"] == "Tensor-guided") &
                             (oracle_df["budget"] == budget)]["accuracy"].iloc[0]
        o_text = oracle_df[(oracle_df["condition"] == "Text-guided (fixed)") &
                           (oracle_df["budget"] == budget)]["accuracy"].iloc[0]
        o_adv = o_tensor - o_text

        # Realistic
        r_tensor = realistic_df[(realistic_df["condition"] == "Tensor-guided") &
                                (realistic_df["budget"] == budget)]["accuracy"].iloc[0]
        r_text = realistic_df[(realistic_df["condition"] == "Text-guided (fixed)") &
                              (realistic_df["budget"] == budget)]["accuracy"].iloc[0]
        r_adv = r_tensor - r_text

        change = r_adv - o_adv
        bstr = f"{int(budget*100)}%"
        print(f"{bstr:>8s}  {o_adv:>+12.2f}  {r_adv:>+12.2f}  {change:>+12.2f}")

    # Also compare composed
    print()
    print("Composed advantage = Composed accuracy - Text-guided accuracy")
    print()
    print(f"{'Budget':>8s}  {'Oracle':>12s}  {'Realistic':>12s}  {'Change':>12s}")
    print(f"{'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

    for budget in [0.10, 0.20, 0.30]:
        o_comp = oracle_df[(oracle_df["condition"] == "Composed") &
                           (oracle_df["budget"] == budget)]["accuracy"].iloc[0]
        o_text = oracle_df[(oracle_df["condition"] == "Text-guided (fixed)") &
                           (oracle_df["budget"] == budget)]["accuracy"].iloc[0]
        o_adv = o_comp - o_text

        r_comp = realistic_df[(realistic_df["condition"] == "Composed") &
                              (realistic_df["budget"] == budget)]["accuracy"].iloc[0]
        r_text = realistic_df[(realistic_df["condition"] == "Text-guided (fixed)") &
                              (realistic_df["budget"] == budget)]["accuracy"].iloc[0]
        r_adv = r_comp - r_text

        change = r_adv - o_adv
        bstr = f"{int(budget*100)}%"
        print(f"{bstr:>8s}  {o_adv:>+12.2f}  {r_adv:>+12.2f}  {change:>+12.2f}")


def print_selection_diagnostics(df):
    """Print diagnostics about what each judge selects."""
    print(f"\n{'='*78}")
    print("SELECTION DIAGNOSTICS")
    print(f"{'='*78}")

    n = len(df)

    for budget_frac in [0.10, 0.20, 0.30]:
        budget = max(1, int(n * budget_frac))
        print(f"\n--- Budget: {int(budget_frac*100)}% ({budget}/{n} items) ---")

        # Text scores
        text_scores = df.apply(text_uncertainty_score_fixed, axis=1).values
        text_idx = np.argsort(text_scores)[-budget:]
        text_sel = df.iloc[text_idx]

        # Tensor scores
        tensor_scores = df.apply(tensor_uncertainty_score, axis=1).values
        tensor_idx = np.argsort(tensor_scores)[-budget:]
        tensor_sel = df.iloc[tensor_idx]

        # Composed scores
        composed_scores = tensor_scores.copy()
        citation_mask = df["is_citation"].values
        for i in range(n):
            if citation_mask[i]:
                composed_scores[i] = 0.0 if df.iloc[i]["is_knowable"] else 1.0
        composed_idx = np.argsort(composed_scores)[-budget:]
        composed_sel = df.iloc[composed_idx]

        print(f"  {'Judge':<20s} {'Unknowable':>12s} {'Knowable':>10s} "
              f"{'Already OK':>12s} {'Need fix':>10s} {'Mean entropy':>14s}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*14}")

        for label, sel in [("Text-guided", text_sel),
                           ("Tensor-guided", tensor_sel),
                           ("Composed", composed_sel)]:
            n_unknowable = (~sel["is_knowable"]).sum()
            n_knowable = sel["is_knowable"].sum()
            n_correct = sel["is_correct"].sum()
            n_incorrect = budget - n_correct
            mean_ent = sel["mean_entropy"].mean()
            print(f"  {label:<20s} {n_unknowable:>12d} {n_knowable:>10d} "
                  f"{n_correct:>12d} {n_incorrect:>10d} {mean_ent:>14.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    project_root = Path(__file__).parent.parent

    # Load raw data
    raw_csv = sorted(project_root.glob("exp27_bounded_verification_*.csv"))
    if not raw_csv:
        print("ERROR: No exp27_bounded_verification_*.csv found.")
        sys.exit(1)

    det_csv = sorted(project_root.glob("exp27b_detailed_*.csv"))
    if not det_csv:
        print("WARNING: No exp27b_detailed_*.csv found. Using simplified evaluator.")
        det_csv = None

    raw_path = raw_csv[-1]
    print(f"Loading raw data: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  {len(df)} rows, {df['model_id'].nunique()} models")

    if det_csv:
        det_path = det_csv[-1]
        print(f"Loading stratified evaluator: {det_path}")
        det = pd.read_csv(det_path)
        det_key = det[["family", "query", "is_knowable", "new_correct"]].copy()
        df = df.merge(det_key, on=["family", "query", "is_knowable"], how="left")
        df["is_correct"] = df["new_correct"].fillna(
            df.apply(check_answer_correct, axis=1)
        ).astype(bool)
        print(f"  Stratified evaluator matched: {df['new_correct'].notna().sum()}/{len(df)}")
    else:
        df["is_correct"] = df.apply(check_answer_correct, axis=1)

    df["is_citation"] = df["query"].apply(is_citation_query)

    baseline_accuracy = df["is_correct"].mean() * 100
    n_correct = df["is_correct"].sum()
    n_incorrect = (~df["is_correct"]).sum()
    print(f"\n  Baseline accuracy: {baseline_accuracy:.1f}% "
          f"({n_correct} correct, {n_incorrect} incorrect)")

    # ---- Verification parameters ----
    print(f"\n{'='*78}")
    print("VERIFICATION PARAMETERS")
    print(f"{'='*78}")
    print(f"  P(verify unknowable) = {P_VERIFY_UNKNOWABLE:.3f}  "
          f"(from 75/80 human calibration)")
    print(f"  P(verify knowable)   = {P_VERIFY_KNOWABLE:.3f}  "
          f"(factual lookup success rate)")
    print(f"  Monte Carlo trials   = {N_TRIALS}")
    print(f"  Random seed          = {RANDOM_SEED}")

    # ---- Selection diagnostics ----
    print_selection_diagnostics(df)

    # ---- Oracle evaluation ----
    print("\n\n" + "#" * 78)
    print("# ORACLE VERIFICATION (original behavior)")
    print("#" * 78)
    oracle_results = run_oracle_evaluation(
        df, text_uncertainty_score_fixed, "Text-guided (fixed)"
    )
    print_oracle_table(oracle_results, "Oracle verification: all selected items corrected")

    # ---- Realistic evaluation ----
    print("\n\n" + "#" * 78)
    print("# REALISTIC VERIFICATION (Monte Carlo, 1000 trials)")
    print("#" * 78)
    realistic_results = run_realistic_evaluation(
        df, text_uncertainty_score_fixed, "Text-guided (fixed)"
    )
    print_realistic_table(realistic_results,
                          "Realistic verification: probabilistic correction")

    # ---- Comparison ----
    print_comparison_table(oracle_results, realistic_results,
                           "Oracle vs Realistic: side-by-side")

    # ---- Advantage analysis ----
    print_advantage_analysis(oracle_results, realistic_results)

    # ---- Headline numbers ----
    print(f"\n{'='*78}")
    print("HEADLINE COMPARISON")
    print(f"{'='*78}")

    for label, budget in [("Tensor@10%", 0.10), ("Text@30%", 0.30)]:
        cond = "Tensor-guided" if "Tensor" in label else "Text-guided (fixed)"
        o_val = oracle_results[(oracle_results["condition"] == cond) &
                               (oracle_results["budget"] == budget)]["accuracy"].iloc[0]
        r_row = realistic_results[(realistic_results["condition"] == cond) &
                                  (realistic_results["budget"] == budget)].iloc[0]
        print(f"  {label}:")
        print(f"    Oracle:    {o_val:.1f}%")
        print(f"    Realistic: {r_row['accuracy']:.1f}% "
              f"[{r_row['ci_lo']:.1f}, {r_row['ci_hi']:.1f}]")
        print()

    # Tensor@10% vs Text@30% comparison
    t10_oracle = oracle_results[(oracle_results["condition"] == "Tensor-guided") &
                                (oracle_results["budget"] == 0.10)]["accuracy"].iloc[0]
    tx30_oracle = oracle_results[(oracle_results["condition"] == "Text-guided (fixed)") &
                                 (oracle_results["budget"] == 0.30)]["accuracy"].iloc[0]
    t10_real = realistic_results[(realistic_results["condition"] == "Tensor-guided") &
                                 (realistic_results["budget"] == 0.10)]["accuracy"].iloc[0]
    tx30_real = realistic_results[(realistic_results["condition"] == "Text-guided (fixed)") &
                                  (realistic_results["budget"] == 0.30)]["accuracy"].iloc[0]

    print(f"  Tensor@10% vs Text@30%:")
    print(f"    Oracle:    {t10_oracle:.1f}% vs {tx30_oracle:.1f}% "
          f"({'TENSOR WINS' if t10_oracle > tx30_oracle else 'TEXT WINS'})")
    print(f"    Realistic: {t10_real:.1f}% vs {tx30_real:.1f}% "
          f"({'TENSOR WINS' if t10_real > tx30_real else 'TEXT WINS'})")

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = project_root / f"exp27_realistic_verification_{timestamp}.csv"

    all_results = pd.concat([
        oracle_results.assign(verification="oracle"),
        realistic_results.assign(verification="realistic"),
    ])
    all_results.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
