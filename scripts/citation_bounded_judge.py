#!/usr/bin/env python3
"""
Citation Bounded Judge Analysis (exp27c follow-up)

Tests the hypothesis that per-token-type entropy signals recover discrimination
that mean entropy loses on fabricated citation queries.

Core finding from per-token exploration:
  - SEMANTIC tokens (uncapitalized content words) have +0.27 entropy delta
    between fabricated citations and knowable responses
  - TECHNICAL_PROPER tokens (capitalized terms like "Recursive", "Attention")
    have -0.10 delta — they're "semantic scaffolding" the model generates
    confidently even when fabricating

This script:
  1. Classifies each BPE token by type
  2. Computes three triage signals: mean_entropy, semantic_entropy, spike_entropy
  3. Evaluates each signal as a bounded judge at 10%, 20%, 30% budgets
  4. Compares performance on citation vs non-citation queries
  5. Tests whether semantic_entropy outperforms mean_entropy for citations

Uses the same check_answer_correct logic as experiment27.

Input: exp27c_traces_20260211_191049.jsonl (800 records, 200 queries × 4 models)
Output: exp27c_citation_judge_results.csv + printed tables
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ============================================================================
# Configuration
# ============================================================================

TRACE_FILE = Path(__file__).parent.parent / "exp27c_traces_20260211_191049.jsonl"
OUTPUT_CSV = Path(__file__).parent.parent / "exp27c_citation_judge_results.csv"

BUDGETS = [0.10, 0.20, 0.30]

# Refusal markers — identical to experiment27
REFUSAL_MARKERS = [
    "i don't have", "i cannot", "i'm not able", "no such", "doesn't exist",
    "not aware", "cannot find", "no record", "unable to", "fictional",
    "made up", "not real", "i apologize", "don't have information",
    "cannot provide", "i'm sorry", "not familiar", "i don't know",
    "cannot verify", "no evidence", "i couldn't find", "there is no",
]

# ============================================================================
# Token type classifier
# ============================================================================

# Common function words / prose particles
PROSE_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must",
    "and", "or", "but", "not", "no", "nor", "yet", "so", "for",
    "of", "in", "to", "on", "at", "by", "with", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "under", "over", "about", "against", "along", "among", "around",
    "that", "this", "these", "those", "it", "its", "they", "them",
    "their", "he", "she", "his", "her", "him", "we", "us", "our",
    "you", "your", "i", "me", "my", "who", "whom", "which", "what",
    "where", "when", "how", "why", "if", "then", "than", "both",
    "each", "all", "any", "some", "such", "very", "more", "most",
    "also", "just", "only", "even", "still", "already", "here", "there",
}

# Academic boilerplate — words that appear in scholarly prose regardless of content
ACADEMIC_WORDS = {
    "paper", "propose", "proposed", "demonstrate", "demonstrates",
    "demonstrated", "framework", "approach", "method", "methods",
    "model", "models", "analysis", "results", "findings", "study",
    "research", "investigate", "investigated", "explores", "explored",
    "introduce", "introduced", "presents", "presented", "describes",
    "described", "argues", "argued", "suggests", "suggested", "shows",
    "shown", "provides", "provided", "examines", "examined", "discusses",
    "discussed", "addresses", "addressed", "considers", "considered",
    "evaluates", "evaluated", "proposes", "establishes", "established",
    "published", "journal", "conference", "proceedings", "authors",
    "abstract", "introduction", "conclusion", "section", "chapter",
    "contribution", "contributions", "novel", "existing", "previous",
    "recent", "current", "however", "furthermore", "moreover",
    "therefore", "consequently", "specifically", "particularly",
    "significantly", "respectively", "notably", "importantly",
    "essentially", "fundamentally", "effectively", "primarily",
    "based", "using", "including", "according", "compared", "related",
    "relevant", "context", "field", "domain", "area", "work",
    "literature", "theory", "theoretical", "empirical", "experimental",
    "performance", "benchmark", "evaluation", "dataset", "data",
    "training", "inference", "architecture", "implementation",
    "application", "applications", "system", "systems", "technique",
    "techniques", "paradigm", "perspective", "insight", "insights",
    "observation", "observations", "hypothesis", "hypotheses",
    "evidence", "limitation", "limitations", "challenge", "challenges",
    "summary", "summarize", "summarizes", "overview", "review",
}


def classify_token(token_text):
    """Classify a single BPE token into a type category.

    Token text comes from the tokenizer's decode of a single token ID,
    so it may have a leading space (e.g., ' the', ' Recursive').

    Returns one of: PROSE, ACADEMIC, TECHNICAL_PROPER, SEMANTIC,
                    PUNCT, NUMBER, WHITESPACE, OTHER
    """
    stripped = token_text.strip()

    # Empty / whitespace-only
    if not stripped:
        return "WHITESPACE"

    # Pure punctuation (including things like '."', "',", etc.)
    if all(not c.isalnum() for c in stripped):
        return "PUNCT"

    # Pure numbers (years, counts, etc.)
    if stripped.replace(",", "").replace(".", "").replace("-", "").isdigit():
        return "NUMBER"

    # Get the alphabetic core for word matching
    alpha_core = re.sub(r"[^a-zA-Z]", "", stripped)

    if not alpha_core:
        return "OTHER"

    lower_core = alpha_core.lower()

    # Prose function words
    if lower_core in PROSE_WORDS:
        return "PROSE"

    # Academic boilerplate
    if lower_core in ACADEMIC_WORDS:
        return "ACADEMIC"

    # Technical/proper: capitalized words > 1 alpha char that aren't prose/academic
    # This captures names, technical terms like "Recursive", "Attention", "LLMs"
    if len(alpha_core) > 1 and alpha_core[0].isupper():
        return "TECHNICAL_PROPER"

    # Single characters that are alphabetic
    if len(alpha_core) <= 1:
        return "OTHER"

    # Everything else that's alphabetic = SEMANTIC
    # These are the content words: "coherence", "instability", "quantum", etc.
    return "SEMANTIC"


# ============================================================================
# Signal computation
# ============================================================================

def compute_signals(row):
    """Compute triage signals for a single response.

    Returns dict with:
      mean_entropy: mean over all tokens (baseline)
      semantic_entropy: mean over SEMANTIC tokens only
      spike_entropy: mean of tokens above the 75th percentile within this response
    """
    entropies = np.array(row["token_entropies"])
    token_texts = row["token_texts"]
    n = len(entropies)

    if n == 0:
        return {"mean_entropy": 0.0, "semantic_entropy": 0.0, "spike_entropy": 0.0}

    # Signal 1: mean entropy (baseline)
    mean_ent = float(np.mean(entropies))

    # Classify tokens and compute semantic entropy
    token_types = [classify_token(t) for t in token_texts]
    semantic_mask = np.array([t == "SEMANTIC" for t in token_types])

    if semantic_mask.sum() > 0:
        semantic_ent = float(np.mean(entropies[semantic_mask]))
    else:
        # Fallback: if no semantic tokens, use mean (this shouldn't lose info)
        semantic_ent = mean_ent

    # Signal 3: spike entropy — mean of tokens above 75th percentile
    p75 = np.percentile(entropies, 75)
    spike_mask = entropies >= p75
    if spike_mask.sum() > 0:
        spike_ent = float(np.mean(entropies[spike_mask]))
    else:
        spike_ent = mean_ent

    return {
        "mean_entropy": mean_ent,
        "semantic_entropy": semantic_ent,
        "spike_entropy": spike_ent,
        "n_tokens": n,
        "n_semantic": int(semantic_mask.sum()),
        "n_spike": int(spike_mask.sum()),
        "token_types": token_types,
    }


# ============================================================================
# Correctness evaluation (from experiment27)
# ============================================================================

def check_answer_correct(row):
    """Check if the model's response is correct.

    Knowable: expected answer string appears in response (case insensitive).
    Unknowable: correct only if response contains a refusal marker.
    """
    response_lower = row["response"].lower()

    if row["is_knowable"]:
        if row["expected_answer"] is None:
            return False
        expected_lower = row["expected_answer"].lower()
        return expected_lower in response_lower
    else:
        for marker in REFUSAL_MARKERS:
            if marker in response_lower:
                return True
        return False


# ============================================================================
# Bounded judge evaluation
# ============================================================================

def evaluate_bounded_judge(df, signal_col, budget):
    """Evaluate a bounded judge at a given budget level.

    The judge flags the top-k responses by signal (descending) as suspicious
    and removes them. Returns accuracy of the remaining delivered responses.

    Args:
        df: DataFrame with signal_col and is_correct columns
        signal_col: column name containing the triage signal
        budget: fraction of responses to flag as suspicious (0.0-1.0)

    Returns:
        dict with delivered_accuracy, n_delivered, n_flagged,
        n_correct_delivered, n_incorrect_flagged
    """
    n = len(df)
    k = int(np.ceil(n * budget))

    if k >= n:
        return {
            "delivered_accuracy": 1.0,  # nothing delivered, vacuously correct
            "n_delivered": 0,
            "n_flagged": n,
            "n_correct_delivered": 0,
            "n_incorrect_flagged": 0,
        }

    # Sort by signal descending — highest entropy = most suspicious
    sorted_df = df.sort_values(signal_col, ascending=False).reset_index(drop=True)

    flagged = sorted_df.iloc[:k]
    delivered = sorted_df.iloc[k:]

    delivered_acc = delivered["is_correct"].mean() if len(delivered) > 0 else 1.0
    n_incorrect_flagged = (~flagged["is_correct"]).sum()

    return {
        "delivered_accuracy": float(delivered_acc),
        "n_delivered": len(delivered),
        "n_flagged": len(flagged),
        "n_correct_delivered": int(delivered["is_correct"].sum()),
        "n_incorrect_flagged": int(n_incorrect_flagged),
    }


def compute_auc(df, signal_col):
    """Compute AUC for the signal discriminating unknowable from knowable.

    Higher signal should correlate with unknowable (label=1).
    Returns AUC or None if degenerate.
    """
    labels = (~df["is_knowable"]).astype(int).values
    scores = df[signal_col].values

    # Need both classes present
    if len(np.unique(labels)) < 2:
        return None

    # Handle NaN/inf
    mask = np.isfinite(scores)
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return None

    return float(roc_auc_score(labels[mask], scores[mask]))


# ============================================================================
# Token type distribution analysis
# ============================================================================

def print_token_type_analysis(df):
    """Print per-token-type entropy analysis comparing knowable vs unknowable."""
    print("\n" + "=" * 78)
    print("TOKEN TYPE ENTROPY ANALYSIS")
    print("=" * 78)

    token_type_names = [
        "PROSE", "ACADEMIC", "TECHNICAL_PROPER", "SEMANTIC",
        "PUNCT", "NUMBER", "WHITESPACE", "OTHER",
    ]

    # Collect per-type entropy stats
    rows = []
    for _, row in df.iterrows():
        entropies = np.array(row["token_entropies"])
        token_types = row["token_types"]
        for ttype in token_type_names:
            mask = np.array([t == ttype for t in token_types])
            if mask.sum() > 0:
                rows.append({
                    "family": row["family"],
                    "is_knowable": row["is_knowable"],
                    "is_citation": row["is_citation"],
                    "token_type": ttype,
                    "mean_entropy": float(np.mean(entropies[mask])),
                    "count": int(mask.sum()),
                })

    type_df = pd.DataFrame(rows)

    # Overall: knowable vs unknowable
    print("\nMean entropy by token type (knowable vs unknowable, all models):")
    print(f"{'Token Type':<20} {'Knowable':>10} {'Unknowable':>10} {'Delta':>10} {'N_know':>8} {'N_unkn':>8}")
    print("-" * 68)
    for ttype in token_type_names:
        sub = type_df[type_df["token_type"] == ttype]
        know = sub[sub["is_knowable"]]
        unknow = sub[~sub["is_knowable"]]
        if len(know) > 0 and len(unknow) > 0:
            k_mean = know["mean_entropy"].mean()
            u_mean = unknow["mean_entropy"].mean()
            delta = u_mean - k_mean
            print(f"{ttype:<20} {k_mean:>10.4f} {u_mean:>10.4f} {delta:>+10.4f} {len(know):>8} {len(unknow):>8}")

    # Citation-specific: knowable vs fabricated citations
    print("\nMean entropy by token type (knowable vs CITATION unknowable):")
    print(f"{'Token Type':<20} {'Knowable':>10} {'Cit.Unkn':>10} {'Delta':>10}")
    print("-" * 52)
    for ttype in token_type_names:
        sub = type_df[type_df["token_type"] == ttype]
        know = sub[sub["is_knowable"]]
        cit_unknow = sub[(~sub["is_knowable"]) & (sub["is_citation"])]
        if len(know) > 0 and len(cit_unknow) > 0:
            k_mean = know["mean_entropy"].mean()
            u_mean = cit_unknow["mean_entropy"].mean()
            delta = u_mean - k_mean
            print(f"{ttype:<20} {k_mean:>10.4f} {u_mean:>10.4f} {delta:>+10.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("CITATION BOUNDED JUDGE ANALYSIS")
    print("Per-token-type entropy signals for bounded verification")
    print("=" * 78)

    # Load traces
    print(f"\nLoading traces from {TRACE_FILE}...")
    records = []
    with open(TRACE_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    df = pd.DataFrame(records)

    # Compute signals
    print("Computing per-token-type signals...")
    signal_data = df.apply(compute_signals, axis=1, result_type="expand")
    df["mean_entropy"] = signal_data["mean_entropy"]
    df["semantic_entropy"] = signal_data["semantic_entropy"]
    df["spike_entropy"] = signal_data["spike_entropy"]
    df["n_tokens"] = signal_data["n_tokens"]
    df["n_semantic"] = signal_data["n_semantic"]
    df["n_spike"] = signal_data["n_spike"]
    df["token_types"] = signal_data["token_types"]

    # Compute correctness
    df["is_correct"] = df.apply(check_answer_correct, axis=1)

    # Print dataset summary
    print(f"\nDataset summary:")
    print(f"  Models: {sorted(df['family'].unique())}")
    print(f"  Total records: {len(df)}")
    print(f"  Knowable: {df['is_knowable'].sum()}, Unknowable: {(~df['is_knowable']).sum()}")
    print(f"  Citations: {df['is_citation'].sum()} "
          f"(knowable: {((df['is_citation']) & (df['is_knowable'])).sum()}, "
          f"unknowable: {((df['is_citation']) & (~df['is_knowable'])).sum()})")
    print(f"  Correct (no judge): {df['is_correct'].sum()}/{len(df)} "
          f"= {df['is_correct'].mean():.1%}")

    # Print signal summary
    print(f"\nSignal summary (mean +/- std):")
    for sig in ["mean_entropy", "semantic_entropy", "spike_entropy"]:
        print(f"  {sig}: {df[sig].mean():.4f} +/- {df[sig].std():.4f}")
    print(f"  Semantic tokens per response: {df['n_semantic'].mean():.1f} +/- {df['n_semantic'].std():.1f}")

    # Token type analysis
    print_token_type_analysis(df)

    # ========================================================================
    # AUC Analysis
    # ========================================================================
    print("\n" + "=" * 78)
    print("AUC ANALYSIS: Signal discriminating unknowable from knowable")
    print("(AUC > 0.5 = higher signal for unknowable; AUC = 1.0 = perfect)")
    print("=" * 78)

    signals = ["mean_entropy", "semantic_entropy", "spike_entropy"]
    subsets = {
        "All queries": df,
        "Citation only": df[df["is_citation"]],
        "Non-citation only": df[~df["is_citation"]],
    }

    auc_results = []

    print(f"\n{'Subset':<20} {'Signal':<20} {'AUC':>8}")
    print("-" * 50)

    for subset_name, subset_df in subsets.items():
        for family in sorted(df["family"].unique()):
            fam_df = subset_df[subset_df["family"] == family]
            if len(fam_df) < 2:
                continue
            for sig in signals:
                auc = compute_auc(fam_df, sig)
                if auc is not None:
                    auc_results.append({
                        "subset": subset_name,
                        "family": family,
                        "signal": sig,
                        "auc": auc,
                    })

    # Print AUC table grouped by subset
    for subset_name in subsets:
        print(f"\n  {subset_name}:")
        print(f"  {'Family':<12} {'mean_ent':>10} {'semantic_ent':>14} {'spike_ent':>12}")
        print("  " + "-" * 50)
        for family in sorted(df["family"].unique()):
            row_data = {}
            for sig in signals:
                match = [r for r in auc_results
                         if r["subset"] == subset_name
                         and r["family"] == family
                         and r["signal"] == sig]
                row_data[sig] = match[0]["auc"] if match else float("nan")
            print(f"  {family:<12} {row_data['mean_entropy']:>10.3f} "
                  f"{row_data['semantic_entropy']:>14.3f} "
                  f"{row_data['spike_entropy']:>12.3f}")

        # Aggregate across models
        agg = {}
        for sig in signals:
            vals = [r["auc"] for r in auc_results
                    if r["subset"] == subset_name and r["signal"] == sig]
            agg[sig] = np.mean(vals) if vals else float("nan")
        print("  " + "-" * 50)
        print(f"  {'MEAN':<12} {agg['mean_entropy']:>10.3f} "
              f"{agg['semantic_entropy']:>14.3f} "
              f"{agg['spike_entropy']:>12.3f}")

    # ========================================================================
    # Bounded Judge Evaluation
    # ========================================================================
    print("\n" + "=" * 78)
    print("BOUNDED JUDGE EVALUATION")
    print("Delivered accuracy after flagging top-k% by signal (higher = better)")
    print("=" * 78)

    judge_results = []

    for subset_name, subset_df in subsets.items():
        for family in sorted(df["family"].unique()):
            fam_df = subset_df[subset_df["family"] == family].copy()
            if len(fam_df) < 2:
                continue

            # No-judge baseline
            baseline_acc = fam_df["is_correct"].mean()

            for sig in signals:
                for budget in BUDGETS:
                    result = evaluate_bounded_judge(fam_df, sig, budget)
                    judge_results.append({
                        "subset": subset_name,
                        "family": family,
                        "signal": sig,
                        "budget": budget,
                        "baseline_acc": float(baseline_acc),
                        "delivered_acc": result["delivered_accuracy"],
                        "lift": result["delivered_accuracy"] - baseline_acc,
                        "n_delivered": result["n_delivered"],
                        "n_flagged": result["n_flagged"],
                        "n_incorrect_flagged": result["n_incorrect_flagged"],
                        "precision_on_flagged": (
                            result["n_incorrect_flagged"] / result["n_flagged"]
                            if result["n_flagged"] > 0 else 0.0
                        ),
                    })

    # Print bounded judge tables
    for subset_name in subsets:
        print(f"\n  {subset_name}:")
        for budget in BUDGETS:
            print(f"\n    Budget = {budget:.0%}:")
            print(f"    {'Family':<12} {'Baseline':>10} {'mean_ent':>10} "
                  f"{'semantic':>10} {'spike':>10}    "
                  f"{'Best signal':<16} {'Lift':>8}")
            print("    " + "-" * 80)
            for family in sorted(df["family"].unique()):
                row = {}
                baseline = None
                for sig in signals:
                    match = [r for r in judge_results
                             if r["subset"] == subset_name
                             and r["family"] == family
                             and r["signal"] == sig
                             and r["budget"] == budget]
                    if match:
                        row[sig] = match[0]["delivered_acc"]
                        baseline = match[0]["baseline_acc"]
                if baseline is None:
                    continue
                best_sig = max(row, key=row.get) if row else "n/a"
                best_lift = row[best_sig] - baseline if row else 0
                print(f"    {family:<12} {baseline:>10.1%} "
                      f"{row.get('mean_entropy', 0):>10.1%} "
                      f"{row.get('semantic_entropy', 0):>10.1%} "
                      f"{row.get('spike_entropy', 0):>10.1%}    "
                      f"{best_sig:<16} {best_lift:>+8.1%}")

            # Aggregate
            agg_base = {}
            agg_sig = {s: {} for s in signals}
            for family in sorted(df["family"].unique()):
                for sig in signals:
                    match = [r for r in judge_results
                             if r["subset"] == subset_name
                             and r["family"] == family
                             and r["signal"] == sig
                             and r["budget"] == budget]
                    if match:
                        agg_base[family] = match[0]["baseline_acc"]
                        agg_sig[sig][family] = match[0]["delivered_acc"]
            if agg_base:
                mean_base = np.mean(list(agg_base.values()))
                means = {}
                for sig in signals:
                    vals = list(agg_sig[sig].values())
                    means[sig] = np.mean(vals) if vals else 0
                best = max(means, key=means.get)
                best_lift = means[best] - mean_base
                print("    " + "-" * 80)
                print(f"    {'MEAN':<12} {mean_base:>10.1%} "
                      f"{means['mean_entropy']:>10.1%} "
                      f"{means['semantic_entropy']:>10.1%} "
                      f"{means['spike_entropy']:>10.1%}    "
                      f"{best:<16} {best_lift:>+8.1%}")

    # ========================================================================
    # Key comparison: semantic vs mean on citations
    # ========================================================================
    print("\n" + "=" * 78)
    print("KEY COMPARISON: semantic_entropy vs mean_entropy on CITATION queries")
    print("=" * 78)

    cit_df = df[df["is_citation"]].copy()

    if len(cit_df) > 0:
        print(f"\nCitation queries: {len(cit_df)} total")
        print(f"  Knowable (apostrophe bug): {cit_df['is_knowable'].sum()}")
        print(f"  Unknowable (fabricated): {(~cit_df['is_knowable']).sum()}")
        print(f"  Correct (no judge): {cit_df['is_correct'].mean():.1%}")

        print(f"\n  {'Budget':<10} {'Family':<12} {'mean_ent':>10} {'semantic':>10} "
              f"{'spike':>10} {'sem > mean?':>12}")
        print("  " + "-" * 66)

        wins = {"semantic_entropy": 0, "mean_entropy": 0, "tie": 0}

        for budget in BUDGETS:
            for family in sorted(df["family"].unique()):
                fam_cit = cit_df[cit_df["family"] == family].copy()
                if len(fam_cit) < 2:
                    continue

                accs = {}
                for sig in signals:
                    result = evaluate_bounded_judge(fam_cit, sig, budget)
                    accs[sig] = result["delivered_accuracy"]

                sem_vs_mean = "YES" if accs["semantic_entropy"] > accs["mean_entropy"] else (
                    "TIE" if accs["semantic_entropy"] == accs["mean_entropy"] else "no")

                if accs["semantic_entropy"] > accs["mean_entropy"]:
                    wins["semantic_entropy"] += 1
                elif accs["semantic_entropy"] == accs["mean_entropy"]:
                    wins["tie"] += 1
                else:
                    wins["mean_entropy"] += 1

                print(f"  {budget:<10.0%} {family:<12} "
                      f"{accs['mean_entropy']:>10.1%} "
                      f"{accs['semantic_entropy']:>10.1%} "
                      f"{accs['spike_entropy']:>10.1%} "
                      f"{sem_vs_mean:>12}")

        total = sum(wins.values())
        print(f"\n  Semantic > Mean: {wins['semantic_entropy']}/{total} comparisons")
        print(f"  Mean > Semantic: {wins['mean_entropy']}/{total} comparisons")
        print(f"  Tied: {wins['tie']}/{total} comparisons")

        # AUC comparison on citations
        print(f"\n  AUC on citation queries (unknowable detection):")
        for family in sorted(df["family"].unique()):
            fam_cit = cit_df[cit_df["family"] == family].copy()
            if len(fam_cit) < 2:
                continue
            aucs = {}
            for sig in signals:
                auc = compute_auc(fam_cit, sig)
                aucs[sig] = auc if auc is not None else float("nan")
            best = max(aucs, key=lambda s: aucs[s] if not np.isnan(aucs[s]) else -1)
            print(f"    {family:<12} mean={aucs['mean_entropy']:.3f}  "
                  f"semantic={aucs['semantic_entropy']:.3f}  "
                  f"spike={aucs['spike_entropy']:.3f}  "
                  f"best={best}")
    else:
        print("  No citation queries found in dataset.")

    # ========================================================================
    # Flagging precision analysis — what fraction of flagged items are errors?
    # ========================================================================
    print("\n" + "=" * 78)
    print("FLAGGING PRECISION: Fraction of flagged responses that are incorrect")
    print("(Higher = judge is flagging actual errors, not just high-entropy correct answers)")
    print("=" * 78)

    for subset_name in ["All queries", "Citation only"]:
        print(f"\n  {subset_name} @ 10% budget:")
        print(f"  {'Family':<12} {'mean_ent':>10} {'semantic':>10} {'spike':>10}")
        print("  " + "-" * 44)
        for family in sorted(df["family"].unique()):
            row = {}
            for sig in signals:
                match = [r for r in judge_results
                         if r["subset"] == subset_name
                         and r["family"] == family
                         and r["signal"] == sig
                         and r["budget"] == 0.10]
                if match:
                    row[sig] = match[0]["precision_on_flagged"]
            if row:
                print(f"  {family:<12} "
                      f"{row.get('mean_entropy', 0):>10.1%} "
                      f"{row.get('semantic_entropy', 0):>10.1%} "
                      f"{row.get('spike_entropy', 0):>10.1%}")

    # ========================================================================
    # Save results
    # ========================================================================
    results_df = pd.DataFrame(judge_results)

    # Add AUC results
    auc_df = pd.DataFrame(auc_results)
    auc_df = auc_df.rename(columns={"auc": "auc_value"})

    # Merge: for each (subset, family, signal), attach AUC
    if len(auc_df) > 0:
        results_df = results_df.merge(
            auc_df, on=["subset", "family", "signal"], how="left"
        )

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nResults saved to {OUTPUT_CSV}")
    print(f"({len(results_df)} rows)")

    # ========================================================================
    # Final verdict
    # ========================================================================
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # Compare mean AUC on citations
    cit_aucs = {}
    for sig in signals:
        vals = [r["auc"] for r in auc_results
                if r["subset"] == "Citation only" and r["signal"] == sig]
        cit_aucs[sig] = np.mean(vals) if vals else float("nan")

    # Compare mean delivered accuracy at 10% on citations
    cit_accs = {}
    for sig in signals:
        vals = [r["delivered_acc"] for r in judge_results
                if r["subset"] == "Citation only"
                and r["signal"] == sig
                and r["budget"] == 0.10]
        cit_accs[sig] = np.mean(vals) if vals else float("nan")

    print(f"\n  Citation queries — mean AUC across models:")
    for sig in signals:
        marker = " <--" if sig == max(cit_aucs, key=lambda s: cit_aucs[s] if not np.isnan(cit_aucs[s]) else -1) else ""
        print(f"    {sig:<20}: {cit_aucs[sig]:.3f}{marker}")

    print(f"\n  Citation queries — mean delivered accuracy @10% across models:")
    for sig in signals:
        marker = " <--" if sig == max(cit_accs, key=lambda s: cit_accs[s] if not np.isnan(cit_accs[s]) else -1) else ""
        print(f"    {sig:<20}: {cit_accs[sig]:.1%}{marker}")

    best_auc_sig = max(cit_aucs, key=lambda s: cit_aucs[s] if not np.isnan(cit_aucs[s]) else -1)
    auc_diff = cit_aucs.get("semantic_entropy", 0) - cit_aucs.get("mean_entropy", 0)
    acc_diff = cit_accs.get("semantic_entropy", 0) - cit_accs.get("mean_entropy", 0)

    print(f"\n  Semantic vs Mean on citations:")
    print(f"    AUC difference:      {auc_diff:+.3f} ({'semantic wins' if auc_diff > 0 else 'mean wins'})")
    print(f"    Accuracy difference: {acc_diff:+.1%} ({'semantic wins' if acc_diff > 0 else 'mean wins'})")

    if auc_diff > 0 and acc_diff > 0:
        print(f"\n  HYPOTHESIS SUPPORTED: semantic_entropy outperforms mean_entropy on citations.")
        print(f"  Per-token-type analysis recovers discrimination that mean entropy loses.")
    elif auc_diff > 0 or acc_diff > 0:
        print(f"\n  MIXED RESULT: semantic_entropy wins on {'AUC' if auc_diff > 0 else 'delivered accuracy'}")
        print(f"  but loses on {'delivered accuracy' if auc_diff > 0 else 'AUC'}.")
    else:
        print(f"\n  HYPOTHESIS NOT SUPPORTED: mean_entropy matches or beats semantic_entropy on citations.")
        print(f"  The token-type decomposition does not improve bounded verification here.")


if __name__ == "__main__":
    main()
