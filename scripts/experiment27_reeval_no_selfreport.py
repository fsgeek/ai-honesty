#!/usr/bin/env python3
"""Re-evaluate Experiment 27 with fixed text-guided baseline.

The original text_uncertainty_score included self-reported confidence,
which has AUC < 0.5 (systematically inverted). Including a counter-productive
signal in the baseline inflated the tensor advantage.

This script re-runs ONLY the evaluation phase using the existing raw data CSV.
No GPU needed.

Changes from original:
  - text_uncertainty_score_v2: uses hedging + length only (no self-report)
  - Reports both original and fixed baselines for comparison
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Judge implementations
# ============================================================================

def text_uncertainty_score_original(row):
    """Original text judge (includes self-report). For comparison only."""
    self_report_unc = 1.0 - row["self_report_confidence"]
    hedge = row["hedge_score"]
    length = row["length_score"]
    return 0.4 * self_report_unc + 0.4 * hedge + 0.2 * length


def text_uncertainty_score_fixed(row):
    """Fixed text judge: hedging + length only. No self-report.

    Equal weighting of the two text-channel signals that actually
    discriminate in the correct direction.
    """
    hedge = row["hedge_score"]
    length = row["length_score"]
    return 0.5 * hedge + 0.5 * length


def tensor_uncertainty_score(row):
    """Tensor-based uncertainty score. Uses mean entropy."""
    return row["mean_entropy"]


def is_citation_query(query):
    """Detect whether a query is asking about a citation/paper.

    Matches the original experiment27_bounded_verification.py logic:
    detects 'Summarize: Author...' pattern and venue name mentions.
    """
    citation_markers = [
        "summarize:", "summarize the key contribution",
        "summarize the main", "'", "NeurIPS", "ICML", "ICLR",
        "ACL", "EMNLP", "NAACL", "OSDI", "SOSP", "EuroSys",
        "AAAI", "AAMAS", "PODC",
    ]
    query_lower = query.lower()
    # Check for "Summarize: Author..." pattern
    if query_lower.startswith("summarize:"):
        return True
    # Check for venue names (papers)
    for marker in citation_markers:
        if marker.lower() in query_lower and "'" in query:
            return True
    return False


# ============================================================================
# Evaluation engine (from experiment27, simplified)
# ============================================================================

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question directly."

KNOWABLE_KEYWORDS = [
    "capital", "element", "symbol", "planet", "continent",
    "ocean", "inventor", "discovered", "composed", "wrote",
    "painted", "built", "founded", "first",
]


def check_answer_correct(row):
    """Simplified correctness check using the stratified evaluator logic."""
    response = str(row.get("response", "")).lower().strip()
    expected = str(row.get("expected_answer", "")).lower().strip()
    is_knowable = row["is_knowable"]

    if not is_knowable:
        # For unknowable: correct if the model refuses/hedges
        refusal_markers = [
            "i don't know", "i cannot", "i'm not sure", "no information",
            "cannot verify", "don't have", "not aware", "no record",
            "i apologize", "i can't",
        ]
        refuses = any(m in response for m in refusal_markers)
        return refuses
    else:
        # For knowable: correct if the expected answer appears
        if not expected:
            return True
        # Morphological normalization
        expected_parts = expected.split("/") if "/" in expected else [expected]
        return any(part.strip() in response for part in expected_parts)


def evaluate_condition(df, condition_key, score_fn, budget_frac, use_citation=False):
    """Evaluate one condition at one budget level."""
    n = len(df)
    budget = max(1, int(n * budget_frac))

    if condition_key == "no_judge":
        return df["is_correct"].mean()

    scores = df.apply(score_fn, axis=1).values

    if use_citation:
        # Composed judge: override citation queries with lookup
        citation_mask = df["is_citation"].values
        for i in range(n):
            if citation_mask[i]:
                scores[i] = 0.0 if df.iloc[i]["is_knowable"] else 1.0

    verify_indices = np.argsort(scores)[-budget:]
    corrected = df["is_correct"].values.copy()

    for idx in verify_indices:
        row = df.iloc[idx]
        if not row["is_knowable"]:
            corrected[idx] = True
        else:
            corrected[idx] = True

    return corrected.sum() / n


def run_evaluation(df, text_fn, text_label):
    """Run four-condition evaluation."""
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
            accuracy = evaluate_condition(df, condition_key, score_fn, budget, use_citation)
            results.append({
                "condition": condition_name,
                "budget": budget,
                "accuracy": accuracy * 100,
            })

    return pd.DataFrame(results)


def print_table(eval_df, title):
    """Print results in paper table format."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    pivot = eval_df.pivot(index="condition", columns="budget", values="accuracy")
    # Reorder rows
    order = [c for c in ["No judge", "Text-guided (original)", "Text-guided (fixed)",
                          "Tensor-guided", "Composed"] if c in pivot.index]
    pivot = pivot.reindex(order)

    print(f"{'Condition':<30s} {'10%':>8s} {'20%':>8s} {'30%':>8s}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for idx, row in pivot.iterrows():
        print(f"{idx:<30s} {row[0.10]:>8.1f} {row[0.20]:>8.1f} {row[0.30]:>8.1f}")


def main():
    project_root = Path(__file__).parent.parent

    # Load raw data (has all signals)
    raw_csv = sorted(project_root.glob("exp27_bounded_verification_*.csv"))
    if not raw_csv:
        print("ERROR: No exp27_bounded_verification_*.csv found.")
        sys.exit(1)

    # Load stratified evaluator results (has corrected is_correct)
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
        # Merge stratified correctness into raw data
        det_key = det[["family", "query", "is_knowable", "new_correct"]].copy()
        df = df.merge(det_key, on=["family", "query", "is_knowable"], how="left")
        df["is_correct"] = df["new_correct"].fillna(
            df.apply(check_answer_correct, axis=1)
        ).astype(bool)
        print(f"  Stratified evaluator matched: {df['new_correct'].notna().sum()}/{len(df)}")
    else:
        df["is_correct"] = df.apply(check_answer_correct, axis=1)

    df["is_citation"] = df["query"].apply(is_citation_query)

    # --- Aggregate (all models pooled) ---
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (all models pooled)")
    print("=" * 70)

    # Original baseline
    eval_orig = run_evaluation(df, text_uncertainty_score_original, "Text-guided (original)")
    print_table(eval_orig, "Original text baseline (with self-report)")

    # Fixed baseline
    eval_fixed = run_evaluation(df, text_uncertainty_score_fixed, "Text-guided (fixed)")
    print_table(eval_fixed, "Fixed text baseline (hedging + length only)")

    # --- Per-model breakdown ---
    print("\n" + "=" * 70)
    print("PER-MODEL RESULTS")
    print("=" * 70)

    for model_id in sorted(df["model_id"].unique()):
        model_df = df[df["model_id"] == model_id].copy()
        model_df["is_correct"] = model_df.apply(check_answer_correct, axis=1)

        eval_orig_m = run_evaluation(model_df, text_uncertainty_score_original, "Text-guided (original)")
        eval_fixed_m = run_evaluation(model_df, text_uncertainty_score_fixed, "Text-guided (fixed)")

        # Merge for comparison
        merged = pd.concat([eval_orig_m, eval_fixed_m]).drop_duplicates(
            subset=["condition", "budget"], keep="last"
        )
        print_table(merged, f"\n{model_id}")

    # --- Save results ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = project_root / f"exp27_reeval_no_selfreport_{timestamp}.csv"

    all_results = pd.concat([
        eval_orig.assign(baseline="original"),
        eval_fixed.assign(baseline="fixed"),
    ])
    all_results.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    # --- Key comparison ---
    print("\n" + "=" * 70)
    print("KEY COMPARISON: Text-guided at 30% budget")
    print("=" * 70)
    orig_30 = eval_orig[(eval_orig["condition"] == "Text-guided (original)") &
                         (eval_orig["budget"] == 0.30)]["accuracy"].iloc[0]
    fixed_30 = eval_fixed[(eval_fixed["condition"] == "Text-guided (fixed)") &
                           (eval_fixed["budget"] == 0.30)]["accuracy"].iloc[0]
    tensor_10 = eval_fixed[(eval_fixed["condition"] == "Tensor-guided") &
                            (eval_fixed["budget"] == 0.10)]["accuracy"].iloc[0]

    print(f"  Original text-guided @30%: {orig_30:.1f}%")
    print(f"  Fixed text-guided @30%:    {fixed_30:.1f}%")
    print(f"  Tensor-guided @10%:        {tensor_10:.1f}%")
    print(f"  Tensor@10% vs Fixed@30%:   {'TENSOR WINS' if tensor_10 > fixed_30 else 'TEXT WINS'}")


if __name__ == "__main__":
    main()
