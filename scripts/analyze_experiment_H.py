#!/usr/bin/env python3
"""Analyze Experiment H: Paired comparison of coding styles on content token entropy.

For each of 15 algorithms, compares compact (Style A) vs documented (Style B)
traces. The key metric is CONTENT-ONLY entropy — entropy of tokens classified
as "content" (not syntactic or semantic scaffolding).

The scaffolding-coupling hypothesis predicts:
  Style B (documented) should have LOWER content token entropy than Style A (compact)
  because surrounding scaffolding context constrains the model's choices for content tokens.

Usage:
    python scripts/analyze_experiment_H.py [--file FILE]
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np
from scipy import stats


def find_latest_file(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_traces(filepath):
    traces = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


# Token classifier (same as other scripts)
SYNTAX = {
    "def", "return", "if", "elif", "else", "for", "while", "in", "not", "and",
    "or", "is", "class", "import", "from", "as", "with", "try", "except",
    "finally", "raise", "pass", "break", "continue", "yield", "lambda",
    "global", "nonlocal", "assert", "del", "True", "False", "None",
    "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "//", "%",
    "**", "&", "|", "^", "~", "<<", ">>", "+=", "-=", "*=", "/=",
    "(", ")", "[", "]", "{", "}", ":", ",", ".", ";", "@", "->",
    "```", "```python",
}

SEMANTIC = {
    "i", "j", "k", "n", "m", "x", "y", "z", "s", "t", "p", "q",
    "left", "right", "mid", "result", "res", "ans", "val", "key",
    "node", "root", "head", "tail", "prev", "next", "curr",
    "temp", "tmp", "count", "total", "sum", "max", "min", "len", "idx",
    "start", "end", "low", "high", "top", "bottom",
    "self", "cls", "args", "kwargs",
    "int", "str", "float", "bool", "list", "dict", "set", "tuple",
    "range", "enumerate", "zip", "map", "filter", "sorted",
    "append", "extend", "insert", "remove", "pop", "get",
    "join", "split", "strip", "replace", "lower", "upper",
    "isinstance", "type", "print",
}


def classify_token(token_text):
    stripped = token_text.strip()
    if not stripped:
        return "syntactic"
    if stripped.startswith("\n"):
        return "syntactic"
    if stripped.startswith("```"):
        return "syntactic"
    if stripped in SYNTAX:
        return "syntactic"
    if stripped in SEMANTIC:
        return "semantic_scaffolding"
    if stripped.isdigit():
        return "semantic_scaffolding"
    return "content"


def analyze_trace(trace):
    """Classify tokens and compute per-tier entropy stats."""
    texts = trace["token_texts"]
    ents = trace["token_entropies"]
    min_len = min(len(texts), len(ents))

    by_class = defaultdict(list)
    for i in range(min_len):
        cls = classify_token(texts[i])
        by_class[cls].append(ents[i])

    total = min_len
    n_syn = len(by_class["syntactic"])
    n_sem = len(by_class["semantic_scaffolding"])
    n_con = len(by_class["content"])
    scaffolding_pct = (n_syn + n_sem) / total * 100 if total > 0 else 0

    content_ent = statistics.mean(by_class["content"]) if by_class["content"] else 0
    scaffolding_ents = by_class["syntactic"] + by_class["semantic_scaffolding"]
    scaffolding_ent = statistics.mean(scaffolding_ents) if scaffolding_ents else 0
    overall_ent = statistics.mean(ents[:min_len]) if min_len > 0 else 0

    # Max entropy and entropy_std for content tokens only
    content_max = max(by_class["content"]) if by_class["content"] else 0
    content_std = statistics.stdev(by_class["content"]) if len(by_class["content"]) > 1 else 0

    return {
        "total_tokens": total,
        "n_content": n_con,
        "scaffolding_pct": scaffolding_pct,
        "content_mean_entropy": content_ent,
        "content_max_entropy": content_max,
        "content_std_entropy": content_std,
        "scaffolding_mean_entropy": scaffolding_ent,
        "overall_mean_entropy": overall_ent,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment H paired comparison analysis")
    parser.add_argument("--file", help="Experiment H traces JSONL file")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_file = args.file or find_latest_file(os.path.join(project_root, "experiment_H_coding_styles_*.jsonl"))

    if not exp_file:
        print("ERROR: No experiment H file found.")
        return

    print(f"Experiment H traces: {exp_file}")
    traces = load_traces(exp_file)
    print(f"Loaded {len(traces)} traces.")

    # Group by algorithm and style
    by_algo = defaultdict(dict)
    for trace in traces:
        algo = trace["name"]
        style = trace["style"]
        analysis = analyze_trace(trace)
        by_algo[algo][style] = {**analysis, **trace}

    # Paired comparison
    print(f"\n{'='*100}")
    print("Paired Comparison: Compact (A) vs Documented (B)")
    print(f"{'='*100}")
    print(f"  {'Algorithm':<20s} {'A Tokens':>8s} {'B Tokens':>8s} "
          f"{'A Scaff%':>8s} {'B Scaff%':>8s} "
          f"{'A ContEnt':>10s} {'B ContEnt':>10s} {'A/B':>5s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*5}")

    paired_content_a = []
    paired_content_b = []
    paired_scaff_a = []
    paired_scaff_b = []
    paired_overall_a = []
    paired_overall_b = []

    for algo in sorted(by_algo.keys()):
        styles = by_algo[algo]
        if "compact" not in styles or "documented" not in styles:
            continue

        a = styles["compact"]
        b = styles["documented"]

        ratio = a["content_mean_entropy"] / b["content_mean_entropy"] if b["content_mean_entropy"] > 0 else float("inf")
        print(f"  {algo:<20s} {a['total_tokens']:>8d} {b['total_tokens']:>8d} "
              f"{a['scaffolding_pct']:>7.1f}% {b['scaffolding_pct']:>7.1f}% "
              f"{a['content_mean_entropy']:>10.6f} {b['content_mean_entropy']:>10.6f} "
              f"{ratio:>4.2f}x")

        paired_content_a.append(a["content_mean_entropy"])
        paired_content_b.append(b["content_mean_entropy"])
        paired_scaff_a.append(a["scaffolding_pct"])
        paired_scaff_b.append(b["scaffolding_pct"])
        paired_overall_a.append(a["overall_mean_entropy"])
        paired_overall_b.append(b["overall_mean_entropy"])

    if len(paired_content_a) < 3:
        print("\n  Too few paired traces for statistical analysis.")
        return

    n_pairs = len(paired_content_a)
    mean_a = statistics.mean(paired_content_a)
    mean_b = statistics.mean(paired_content_b)

    print(f"\n{'='*100}")
    print("Aggregate Statistics")
    print(f"{'='*100}")
    print(f"  N pairs: {n_pairs}")
    print(f"  Mean content entropy (compact):    {mean_a:.6f}")
    print(f"  Mean content entropy (documented): {mean_b:.6f}")
    print(f"  Ratio (compact/documented): {mean_a/mean_b:.3f}x" if mean_b > 0 else "")

    # Paired t-test on content entropy
    t_stat, p_value = stats.ttest_rel(paired_content_a, paired_content_b)
    print(f"\n  Paired t-test (content entropy A vs B):")
    print(f"    t = {t_stat:.3f}, p = {p_value:.4f}")

    # Wilcoxon signed-rank (non-parametric)
    w_stat, w_p = stats.wilcoxon(paired_content_a, paired_content_b)
    print(f"  Wilcoxon signed-rank:")
    print(f"    W = {w_stat:.1f}, p = {w_p:.4f}")

    # Effect size (Cohen's d)
    diffs = np.array(paired_content_a) - np.array(paired_content_b)
    cohens_d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0
    print(f"  Cohen's d: {cohens_d:.3f}")

    # How many algorithms show A > B (supports hypothesis)?
    n_support = sum(1 for a, b in zip(paired_content_a, paired_content_b) if a > b)
    print(f"\n  Algorithms where compact > documented (supports hypothesis): {n_support}/{n_pairs}")

    # Scaffolding comparison
    print(f"\n  Mean scaffolding% (compact):    {statistics.mean(paired_scaff_a):.1f}%")
    print(f"  Mean scaffolding% (documented): {statistics.mean(paired_scaff_b):.1f}%")

    # Overall entropy comparison
    print(f"\n  Mean overall entropy (compact):    {statistics.mean(paired_overall_a):.6f}")
    print(f"  Mean overall entropy (documented): {statistics.mean(paired_overall_b):.6f}")

    # Interpretation
    print(f"\n{'='*100}")
    print("Interpretation")
    print(f"{'='*100}")
    mean_scaff_a = statistics.mean(paired_scaff_a)
    mean_scaff_b = statistics.mean(paired_scaff_b)

    # Check whether scaffolding direction matches the E analysis prediction
    # E analysis: higher scaffolding% → lower content entropy (rho = -0.614)
    scaff_a_higher = mean_scaff_a > mean_scaff_b
    content_a_lower = mean_a < mean_b
    direction_consistent = scaff_a_higher == content_a_lower

    if p_value < 0.05 and direction_consistent:
        higher_style = "compact" if scaff_a_higher else "documented"
        lower_ent_style = "compact" if content_a_lower else "documented"
        print(f"  E-CONSISTENT: {higher_style} has MORE scaffolding ({mean_scaff_a:.1f}% vs {mean_scaff_b:.1f}%)")
        print(f"  and {lower_ent_style} has LOWER content entropy ({min(mean_a, mean_b):.6f} vs {max(mean_a, mean_b):.6f}).")
        print(f"  Direction matches E analysis (more scaffolding → lower content entropy).")
        print(f"")
        print(f"  CAUTION: Confounded by content type. Documented style generates English prose")
        print(f"  in docstrings/comments (high entropy), while compact generates only algorithm")
        print(f"  logic (low entropy). The scaffolding-content coupling and the content-type")
        print(f"  effects cannot be separated in this experimental design.")
    elif p_value < 0.05:
        print(f"  INCONSISTENT with E analysis direction.")
        if scaff_a_higher and not content_a_lower:
            print(f"  Compact has more scaffolding but HIGHER content entropy.")
        else:
            print(f"  Documented has more scaffolding but HIGHER content entropy.")
    else:
        print(f"  INCONCLUSIVE: No significant difference (p = {p_value:.4f}).")

    # Save
    if args.output:
        output = {
            "n_pairs": n_pairs,
            "mean_content_entropy_compact": mean_a,
            "mean_content_entropy_documented": mean_b,
            "ratio": mean_a / mean_b if mean_b > 0 else None,
            "paired_t_stat": t_stat,
            "paired_p_value": p_value,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "cohens_d": cohens_d,
            "n_support": n_support,
            "per_algorithm": {
                algo: {
                    "compact_content_ent": by_algo[algo]["compact"]["content_mean_entropy"],
                    "documented_content_ent": by_algo[algo]["documented"]["content_mean_entropy"],
                }
                for algo in sorted(by_algo.keys())
                if "compact" in by_algo[algo] and "documented" in by_algo[algo]
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
