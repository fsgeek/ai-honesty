#!/usr/bin/env python3
"""Reproducible analysis of scaffolding-content entropy coupling.

Two analyses:
  F (Natural Variation): Compare comment entropy vs code entropy within
    each function to measure natural scaffolding effects.
  E (Simulated Stripping): Correlate scaffolding% with content-token-only
    entropy across functions, with partial correlation controlling for
    complexity (token count).

Key finding from in-context exploration:
  - Partial correlation rho = -0.700 (p = 0.004) after controlling for complexity
  - Scaffolding context genuinely helps the model be certain about content tokens
  - Implication: minification would INCREASE content entropy

Usage:
    python scripts/analyze_scaffolding_coupling.py [--code FILE]
"""

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict

import numpy as np
from scipy import stats


def find_latest_file(pattern):
    """Find the most recent file matching a glob pattern."""
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_traces(filepath):
    """Load traces from a JSONL file."""
    traces = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def is_comment_token(token_text):
    """Check if a token is part of a comment or docstring."""
    stripped = token_text.strip()
    # Python comment indicator
    if stripped.startswith("#"):
        return True
    # Docstring delimiters and their content (heuristic: tokens after """ until next """)
    if stripped in ('"""', "'''", '"\"\"\n', "'''\n"):
        return True
    return False


def is_code_token(token_text):
    """Check if a token is actual code (not comment, not whitespace-only)."""
    stripped = token_text.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    if stripped in ('"""', "'''"):
        return False
    return True


def classify_for_scaffolding(token_text):
    """Classify token for scaffolding analysis (simplified two-tier)."""
    # Python keywords and syntax
    SYNTAX = {
        "def", "return", "if", "elif", "else", "for", "while", "in", "not",
        "and", "or", "is", "class", "import", "from", "as", "with", "try",
        "except", "finally", "raise", "pass", "break", "continue", "yield",
        "lambda", "True", "False", "None",
        "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "//", "%",
        "**", "(", ")", "[", "]", "{", "}", ":", ",", ".", ";", "@",
        "```", "```python",
    }
    SEMANTIC = {
        "i", "j", "k", "n", "m", "x", "y", "z", "s", "t",
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

    stripped = token_text.strip()
    if not stripped or stripped.isspace():
        return "scaffolding"
    if stripped in ("\n", "\r\n"):
        return "scaffolding"
    if stripped.startswith("\n"):
        return "scaffolding"
    if stripped in SYNTAX:
        return "scaffolding"
    if stripped in SEMANTIC:
        return "scaffolding"
    if stripped.isdigit():
        return "scaffolding"
    return "content"


def analyze_f_natural_variation(traces):
    """Analysis F: Compare comment/docstring entropy vs code entropy per function."""
    print("\n" + "=" * 70)
    print("Analysis F: Natural Variation (Comment vs Code Entropy)")
    print("=" * 70)

    per_function = []

    for trace in traces:
        token_texts = trace["token_texts"]
        token_entropies = trace["token_entropies"]
        min_len = min(len(token_texts), len(token_entropies))

        # Segment into regions: detect docstring blocks and comment lines
        comment_entropies = []
        code_entropies = []
        in_docstring = False

        for i in range(min_len):
            text = token_texts[i]
            ent = token_entropies[i]
            stripped = text.strip()

            # Toggle docstring state
            if stripped in ('"""', "'''", '"""\n', "'''\n"):
                in_docstring = not in_docstring
                comment_entropies.append(ent)
                continue

            if in_docstring:
                comment_entropies.append(ent)
                continue

            if stripped.startswith("#"):
                comment_entropies.append(ent)
                continue

            # Skip whitespace-only and code fence
            if not stripped or stripped.startswith("```"):
                continue

            code_entropies.append(ent)

        if comment_entropies and code_entropies:
            comment_mean = statistics.mean(comment_entropies)
            code_mean = statistics.mean(code_entropies)
            ratio = comment_mean / code_mean if code_mean > 0 else float("inf")
            per_function.append({
                "name": trace["name"],
                "comment_tokens": len(comment_entropies),
                "code_tokens": len(code_entropies),
                "comment_mean_entropy": comment_mean,
                "code_mean_entropy": code_mean,
                "ratio": ratio,
            })

    if not per_function:
        print("  No functions with both comment and code tokens found.")
        return {}

    print(f"\n  {'Function':<25s} {'Comment Tokens':>14s} {'Code Tokens':>12s} "
          f"{'Comment Ent':>12s} {'Code Ent':>10s} {'Ratio':>7s}")
    print(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*12} {'-'*10} {'-'*7}")

    for f in per_function:
        print(f"  {f['name']:<25s} {f['comment_tokens']:>14d} {f['code_tokens']:>12d} "
              f"{f['comment_mean_entropy']:>12.6f} {f['code_mean_entropy']:>10.6f} "
              f"{f['ratio']:>6.1f}x")

    # Aggregate
    all_comment_ent = [f["comment_mean_entropy"] for f in per_function]
    all_code_ent = [f["code_mean_entropy"] for f in per_function]
    overall_comment = statistics.mean(all_comment_ent)
    overall_code = statistics.mean(all_code_ent)
    overall_ratio = overall_comment / overall_code if overall_code > 0 else float("inf")

    print(f"\n  Overall: comments {overall_comment:.6f} vs code {overall_code:.6f} = {overall_ratio:.1f}x")

    # Paired test
    if len(per_function) >= 3:
        t_stat, p_val = stats.ttest_rel(all_comment_ent, all_code_ent)
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    else:
        t_stat, p_val = None, None

    return {
        "per_function": per_function,
        "overall_comment_entropy": overall_comment,
        "overall_code_entropy": overall_code,
        "overall_ratio": overall_ratio,
        "paired_t_stat": t_stat,
        "paired_p_value": p_val,
    }


def analyze_e_simulated_stripping(traces):
    """Analysis E: Correlate scaffolding% with content-token entropy."""
    print("\n" + "=" * 70)
    print("Analysis E: Simulated Stripping (Scaffolding% vs Content Entropy)")
    print("=" * 70)

    per_function = []

    for trace in traces:
        token_texts = trace["token_texts"]
        token_entropies = trace["token_entropies"]
        min_len = min(len(token_texts), len(token_entropies))

        n_scaffolding = 0
        n_content = 0
        content_entropies = []

        for i in range(min_len):
            cls = classify_for_scaffolding(token_texts[i])
            if cls == "scaffolding":
                n_scaffolding += 1
            else:
                n_content += 1
                content_entropies.append(token_entropies[i])

        total = n_scaffolding + n_content
        if total == 0 or not content_entropies:
            continue

        scaffolding_pct = n_scaffolding / total * 100
        content_mean_entropy = statistics.mean(content_entropies)

        per_function.append({
            "name": trace["name"],
            "total_tokens": total,
            "scaffolding_pct": scaffolding_pct,
            "content_mean_entropy": content_mean_entropy,
            "n_content_tokens": n_content,
        })

    if len(per_function) < 5:
        print("  Too few traces for correlation analysis.")
        return {}

    print(f"\n  {'Function':<25s} {'Tokens':>6s} {'Scaff%':>7s} {'Content Ent':>12s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*12}")
    for f in per_function:
        print(f"  {f['name']:<25s} {f['total_tokens']:>6d} {f['scaffolding_pct']:>6.1f}% "
              f"{f['content_mean_entropy']:>12.6f}")

    # Raw correlation
    scaffolding_pcts = np.array([f["scaffolding_pct"] for f in per_function])
    content_ents = np.array([f["content_mean_entropy"] for f in per_function])
    token_counts = np.array([f["total_tokens"] for f in per_function])

    rho_raw, p_raw = stats.spearmanr(scaffolding_pcts, content_ents)
    print(f"\n  Raw Spearman correlation (scaffolding% vs content entropy):")
    print(f"    rho = {rho_raw:.3f}, p = {p_raw:.4f}")

    # Confound check: is scaffolding% correlated with token count?
    rho_confound, p_confound = stats.spearmanr(scaffolding_pcts, token_counts)
    print(f"\n  Confound check (scaffolding% vs token count):")
    print(f"    rho = {rho_confound:.3f}, p = {p_confound:.4f}")

    # Partial correlation: scaffolding% vs content entropy, controlling for token count
    # Using Pearson partial correlation
    from numpy.linalg import lstsq

    def partial_correlation(x, y, z):
        """Partial Spearman correlation of x,y controlling for z."""
        # Rank-transform for Spearman
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        rz = stats.rankdata(z)

        # Residualize x and y on z
        A = np.column_stack([rz, np.ones(len(rz))])
        res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
        res_y = ry - A @ lstsq(A, ry, rcond=None)[0]

        # Pearson on residuals
        r, p = stats.pearsonr(res_x, res_y)
        return r, p

    rho_partial, p_partial = partial_correlation(scaffolding_pcts, content_ents, token_counts)
    print(f"\n  Partial correlation (controlling for token count):")
    print(f"    rho = {rho_partial:.3f}, p = {p_partial:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if rho_partial < -0.3 and p_partial < 0.05:
        print(f"    CONFIRMED: More scaffolding -> lower content entropy (rho={rho_partial:.3f})")
        print(f"    Scaffolding context helps the model be more certain about content.")
        print(f"    Implication: Removing scaffolding (minification) would INCREASE content entropy.")
    elif abs(rho_partial) < 0.3:
        print(f"    No significant relationship after controlling for complexity.")
    else:
        print(f"    Relationship exists but direction unclear. Investigate further.")

    return {
        "per_function": per_function,
        "raw_spearman_rho": rho_raw,
        "raw_spearman_p": p_raw,
        "confound_rho": rho_confound,
        "confound_p": p_confound,
        "partial_rho": rho_partial,
        "partial_p": p_partial,
    }


def main():
    parser = argparse.ArgumentParser(description="Scaffolding-content entropy coupling analysis")
    parser.add_argument("--code", help="Code entropy traces JSONL file")
    parser.add_argument("--output", help="Output JSON file", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    code_file = args.code or find_latest_file(os.path.join(project_root, "code_entropy_traces_*.jsonl"))

    if not code_file:
        print("ERROR: No code trace file found. Specify with --code.")
        return

    print(f"Code traces: {code_file}")
    traces = load_traces(code_file)
    print(f"Loaded {len(traces)} traces.")

    # Run both analyses
    f_results = analyze_f_natural_variation(traces)
    e_results = analyze_e_simulated_stripping(traces)

    # Save
    if args.output:
        output = {
            "f_natural_variation": f_results,
            "e_simulated_stripping": e_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
