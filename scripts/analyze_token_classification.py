#!/usr/bin/env python3
"""Reproducible token classification analysis for code and math proof traces.

Classifies tokens into three tiers:
  1. Syntactic scaffolding — grammar-forced (keywords, operators, delimiters)
  2. Semantic scaffolding — convention-forced (standard variable names, common patterns)
  3. Content — genuine model decisions (algorithm-specific logic, comments, names)

Reads JSONL trace files (code_entropy_traces, math_proof_traces batches 1+2),
computes per-domain scaffolding ratios, entropy ratios, spike concentration,
and the cross-domain manifold comparison table.

Usage:
    python scripts/analyze_token_classification.py [--code FILE] [--math1 FILE] [--math2 FILE]

If no files specified, uses the most recent matching JSONL in the project root.
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from scipy.stats import binomtest, ttest_rel


# ─── Token Classification Rules ──────────────────────────────────────────────

# Python syntactic scaffolding: grammar-forced tokens
PYTHON_SYNTAX_TOKENS = {
    # Keywords
    "def", "return", "if", "elif", "else", "for", "while", "in", "not", "and",
    "or", "is", "class", "import", "from", "as", "with", "try", "except",
    "finally", "raise", "pass", "break", "continue", "yield", "lambda",
    "global", "nonlocal", "assert", "del", "True", "False", "None",
    # Operators and delimiters
    "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "//", "%",
    "**", "&", "|", "^", "~", "<<", ">>", "+=", "-=", "*=", "/=",
    "(", ")", "[", "]", "{", "}", ":", ",", ".", ";", "@", "->", "...",
    # Markdown/code fences
    "```", "```python", "```py",
}

# Python semantic scaffolding: convention-forced variable names
PYTHON_SEMANTIC_SCAFFOLDING = {
    # Common iterator/index names
    "i", "j", "k", "n", "m", "x", "y", "z", "s", "t", "p", "q",
    # Common variable names
    "left", "right", "mid", "result", "res", "ans", "ret", "val", "key",
    "node", "root", "head", "tail", "prev", "next", "curr", "current",
    "temp", "tmp", "count", "total", "sum", "max", "min", "len", "idx",
    "start", "end", "low", "high", "top", "bottom",
    # Common parameter patterns
    "self", "cls", "args", "kwargs",
    # Common type-related
    "int", "str", "float", "bool", "list", "dict", "set", "tuple",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    # Common method names
    "append", "extend", "insert", "remove", "pop", "get", "items",
    "keys", "values", "join", "split", "strip", "replace", "lower",
    "upper", "format", "isinstance", "type", "print",
}

# Math proof syntactic scaffolding
MATH_SYNTAX_TOKENS = {
    # LaTeX structural commands
    "\\", "$", "$$", "{", "}", "[", "]", "(", ")", ",", ".", ":", ";",
    "=", "<", ">", "+", "-", "*", "/", "^", "_",
    # Common LaTeX commands (structural, not semantic)
    "\\text", "\\mathbb", "\\in", "\\subset", "\\subseteq", "\\cup",
    "\\cap", "\\times", "\\to", "\\rightarrow", "\\Rightarrow",
    "\\implies", "\\iff", "\\forall", "\\exists", "\\neg", "\\land",
    "\\lor", "\\leq", "\\geq", "\\neq", "\\infty", "\\cdot", "\\ldots",
    "\\frac", "\\sum", "\\prod", "\\int", "\\lim",
    # Markdown formatting
    "**", "*", "#", "##", "###", "####", ">", "---", "```",
    # Structural words in proofs
    "Proof", "proof", "QED", "Q.E.D.", "\\qed", "\\blacksquare",
}

# Math proof semantic scaffolding: conventional proof language
MATH_SEMANTIC_SCAFFOLDING = {
    "let", "Let", "suppose", "Suppose", "assume", "Assume",
    "then", "Then", "therefore", "Therefore", "hence", "Hence",
    "thus", "Thus", "since", "Since", "because", "Because",
    "we", "We", "have", "show", "want", "need", "must",
    "consider", "Consider", "define", "Define", "note", "Note",
    "recall", "Recall", "observe", "Observe",
    "if", "If", "and", "or", "not", "but", "But",
    "where", "Where", "such", "that", "which", "with", "for", "by",
    "the", "The", "a", "an", "is", "are", "was", "be", "been",
    "this", "This", "it", "It", "there", "There", "any", "all", "every",
    "some", "no", "each", "given", "Given",
    "contradiction", "Contradiction", "case", "Case",
    "induction", "hypothesis", "base",
}


def classify_token_code(token_text):
    """Classify a single token from a code trace into three tiers."""
    text = token_text.strip()

    # Empty / whitespace-only
    if not text or text.isspace():
        return "syntactic"

    # Newlines
    if text in ("\n", "\r\n", "\r") or text.startswith("\n"):
        return "syntactic"

    # Code fence markers
    if text.startswith("```"):
        return "syntactic"

    # Pure punctuation / operators
    if text in PYTHON_SYNTAX_TOKENS:
        return "syntactic"

    # Python keywords (check stripped)
    if text in PYTHON_SYNTAX_TOKENS:
        return "syntactic"

    # Check if it's a keyword embedded in whitespace
    stripped = text.strip()
    if stripped in PYTHON_SYNTAX_TOKENS:
        return "syntactic"

    # Semantic scaffolding: conventional names
    if stripped in PYTHON_SEMANTIC_SCAFFOLDING:
        return "semantic_scaffolding"

    # Numbers (conventional in many contexts)
    if stripped.isdigit() or stripped.replace(".", "", 1).isdigit():
        return "semantic_scaffolding"

    # String delimiters / docstring markers
    if stripped in ('""', "''", '"""', "'''", '"', "'"):
        return "syntactic"

    # Content: everything else (function names, string content, comments, etc.)
    return "content"


def classify_token_math(token_text):
    """Classify a single token from a math proof trace into three tiers."""
    text = token_text.strip()

    # Empty / whitespace-only
    if not text or text.isspace():
        return "syntactic"

    # Newlines
    if text in ("\n", "\r\n", "\r") or text.startswith("\n"):
        return "syntactic"

    # LaTeX structural tokens
    if text.startswith("\\") and len(text) <= 15:
        # LaTeX commands — check if it's structural vs content
        if text in MATH_SYNTAX_TOKENS or text.startswith("\\text") or text.startswith("\\math"):
            return "syntactic"

    # Markdown formatting
    if text in ("**", "*", "#", "##", "###", "####", ">", "---", "```"):
        return "syntactic"

    # Pure punctuation
    stripped = text.strip()
    if stripped in {".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "=",
                    "<", ">", "+", "-", "*", "/", "^", "_", "$", "$$",
                    "≤", "≥", "≠", "→", "⇒", "∈", "⊂", "⊆", "∪", "∩",
                    "∀", "∃", "¬", "∧", "∨", "⟹", "⟺"}:
        return "syntactic"

    # Semantic scaffolding: conventional proof words
    if stripped in MATH_SEMANTIC_SCAFFOLDING:
        return "semantic_scaffolding"

    # Single-letter variables (conventional in math)
    if len(stripped) == 1 and stripped.isalpha():
        return "semantic_scaffolding"

    # Numbers
    if stripped.isdigit() or stripped.replace(".", "", 1).isdigit():
        return "semantic_scaffolding"

    # Common number words
    if stripped.lower() in {"zero", "one", "two", "first", "second"}:
        return "semantic_scaffolding"

    # Content: theorem-specific terms, definitions, novel logical steps
    return "content"


def analyze_traces(traces, classifier_fn, domain_name):
    """Analyze a set of traces using the given classifier function."""
    all_classifications = []
    all_entropies_by_class = defaultdict(list)
    per_trace_stats = []

    for trace in traces:
        token_texts = trace["token_texts"]
        token_entropies = trace["token_entropies"]

        if len(token_texts) != len(token_entropies):
            min_len = min(len(token_texts), len(token_entropies))
            token_texts = token_texts[:min_len]
            token_entropies = token_entropies[:min_len]

        classifications = []
        for text, ent in zip(token_texts, token_entropies):
            cls = classifier_fn(text)
            classifications.append(cls)
            all_entropies_by_class[cls].append(ent)
            all_classifications.append(cls)

        # Per-trace stats
        n_tokens = len(classifications)
        n_syntactic = classifications.count("syntactic")
        n_semantic = classifications.count("semantic_scaffolding")
        n_content = classifications.count("content")
        scaffolding_pct = (n_syntactic + n_semantic) / n_tokens * 100 if n_tokens > 0 else 0

        # Spike concentration: entropy > 90th percentile
        if token_entropies:
            threshold = sorted(token_entropies)[int(len(token_entropies) * 0.9)]
            spike_tokens = [(cls, ent) for cls, ent in zip(classifications, token_entropies) if ent > threshold]
            n_spikes = len(spike_tokens)
            n_spikes_content = sum(1 for cls, _ in spike_tokens if cls == "content")
            n_spikes_semantic = sum(1 for cls, _ in spike_tokens if cls in ("content", "semantic_scaffolding"))
            spike_content_pct = n_spikes_content / n_spikes * 100 if n_spikes > 0 else 0
        else:
            n_spikes = n_spikes_content = 0
            spike_content_pct = 0

        per_trace_stats.append({
            "name": trace.get("name", "unknown"),
            "category": trace.get("category", "knowable"),
            "n_tokens": n_tokens,
            "n_syntactic": n_syntactic,
            "n_semantic_scaffolding": n_semantic,
            "n_content": n_content,
            "scaffolding_pct": scaffolding_pct,
            "mean_entropy": trace.get("mean_entropy", statistics.mean(token_entropies) if token_entropies else 0),
            "n_spikes": n_spikes,
            "n_spikes_content": n_spikes_content,
            "spike_content_pct": spike_content_pct,
        })

    # Aggregate stats
    total = len(all_classifications)
    n_syn = all_classifications.count("syntactic")
    n_sem = all_classifications.count("semantic_scaffolding")
    n_con = all_classifications.count("content")

    mean_ent = {cls: statistics.mean(ents) if ents else 0
                for cls, ents in all_entropies_by_class.items()}

    # Entropy ratio: content / scaffolding
    scaffolding_ents = all_entropies_by_class["syntactic"] + all_entropies_by_class["semantic_scaffolding"]
    content_ents = all_entropies_by_class["content"]
    scaffolding_mean = statistics.mean(scaffolding_ents) if scaffolding_ents else 0
    content_mean = statistics.mean(content_ents) if content_ents else 0
    entropy_ratio = content_mean / scaffolding_mean if scaffolding_mean > 0 else float("inf")

    return {
        "domain": domain_name,
        "n_traces": len(traces),
        "total_tokens": total,
        "syntactic_pct": n_syn / total * 100 if total > 0 else 0,
        "semantic_scaffolding_pct": n_sem / total * 100 if total > 0 else 0,
        "content_pct": n_con / total * 100 if total > 0 else 0,
        "total_scaffolding_pct": (n_syn + n_sem) / total * 100 if total > 0 else 0,
        "mean_entropy_syntactic": mean_ent.get("syntactic", 0),
        "mean_entropy_semantic_scaffolding": mean_ent.get("semantic_scaffolding", 0),
        "mean_entropy_content": mean_ent.get("content", 0),
        "entropy_ratio": entropy_ratio,
        "per_trace": per_trace_stats,
    }


def spike_concentration_test(traces, classifier_fn):
    """Test whether entropy spikes concentrate on content tokens."""
    per_trace_content_pct = []  # fraction of spikes that are content, per trace
    per_trace_base_rate = []    # fraction of tokens that are content, per trace

    for trace in traces:
        token_texts = trace["token_texts"]
        token_entropies = trace["token_entropies"]
        min_len = min(len(token_texts), len(token_entropies))
        token_texts = token_texts[:min_len]
        token_entropies = token_entropies[:min_len]

        classifications = [classifier_fn(t) for t in token_texts]
        n_content = sum(1 for c in classifications if c == "content")
        base_rate = n_content / len(classifications) if classifications else 0

        # Spikes: top 10% by entropy
        if len(token_entropies) < 5:
            continue
        threshold = sorted(token_entropies)[int(len(token_entropies) * 0.9)]
        spike_indices = [i for i, e in enumerate(token_entropies) if e > threshold]
        if not spike_indices:
            continue

        n_spike_content = sum(1 for i in spike_indices if classifications[i] == "content")
        spike_pct = n_spike_content / len(spike_indices)

        per_trace_content_pct.append(spike_pct)
        per_trace_base_rate.append(base_rate)

    if len(per_trace_content_pct) < 3:
        return {"error": "Too few traces for statistical test"}

    # Overall spike concentration
    mean_spike_pct = statistics.mean(per_trace_content_pct)
    mean_base_rate = statistics.mean(per_trace_base_rate)

    # Paired t-test: spike_content_pct vs base_rate across traces
    t_stat, p_value = ttest_rel(per_trace_content_pct, per_trace_base_rate)

    # Binomial test on pooled counts
    total_spikes = sum(1 for _ in per_trace_content_pct for __ in range(1))  # placeholder
    # More precise: re-count
    total_spike_content = 0
    total_spike_all = 0
    for trace in traces:
        token_texts = trace["token_texts"]
        token_entropies = trace["token_entropies"]
        min_len = min(len(token_texts), len(token_entropies))
        token_texts = token_texts[:min_len]
        token_entropies = token_entropies[:min_len]

        classifications = [classifier_fn(t) for t in token_texts]
        if len(token_entropies) < 5:
            continue
        threshold = sorted(token_entropies)[int(len(token_entropies) * 0.9)]
        spike_indices = [i for i, e in enumerate(token_entropies) if e > threshold]
        total_spike_all += len(spike_indices)
        total_spike_content += sum(1 for i in spike_indices if classifications[i] == "content")

    binom_result = binomtest(total_spike_content, total_spike_all, mean_base_rate, alternative="greater")

    return {
        "n_traces": len(per_trace_content_pct),
        "mean_spike_content_pct": mean_spike_pct * 100,
        "mean_base_rate_pct": mean_base_rate * 100,
        "excess_pct": (mean_spike_pct - mean_base_rate) * 100,
        "paired_t_stat": t_stat,
        "paired_p_value": p_value,
        "binomial_k": total_spike_content,
        "binomial_n": total_spike_all,
        "binomial_p0": mean_base_rate,
        "binomial_p_value": binom_result.pvalue,
    }


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


def print_domain_summary(result):
    """Print summary for a single domain."""
    print(f"\n{'='*70}")
    print(f"Domain: {result['domain']}")
    print(f"  Traces: {result['n_traces']}, Total tokens: {result['total_tokens']}")
    print(f"  Syntactic scaffolding:  {result['syntactic_pct']:5.1f}%  "
          f"(mean entropy: {result['mean_entropy_syntactic']:.6f})")
    print(f"  Semantic scaffolding:   {result['semantic_scaffolding_pct']:5.1f}%  "
          f"(mean entropy: {result['mean_entropy_semantic_scaffolding']:.6f})")
    print(f"  Content:                {result['content_pct']:5.1f}%  "
          f"(mean entropy: {result['mean_entropy_content']:.6f})")
    print(f"  Total scaffolding:      {result['total_scaffolding_pct']:5.1f}%")
    print(f"  Entropy ratio (content/scaffolding): {result['entropy_ratio']:.2f}x")


def print_math_category_breakdown(traces):
    """Print entropy breakdown by proof category."""
    by_category = defaultdict(list)
    for trace in traces:
        cat = trace.get("category", "knowable")
        mean_ent = trace.get("mean_entropy")
        if mean_ent is None:
            mean_ent = statistics.mean(trace["token_entropies"]) if trace["token_entropies"] else 0
        by_category[cat].append(mean_ent)

    print(f"\n{'='*70}")
    print("Math Proof Entropy by Category:")
    baseline = statistics.mean(by_category.get("knowable", [0.159]))
    for cat in ["knowable", "independent", "open", "false", "neutrosophic", "erdos_solved"]:
        if cat in by_category:
            mean = statistics.mean(by_category[cat])
            ratio = mean / baseline if baseline > 0 else float("inf")
            print(f"  {cat:20s}: mean_entropy={mean:.3f}  ({ratio:.1f}x baseline)  n={len(by_category[cat])}")


def print_cross_domain_table(results):
    """Print the cross-domain manifold comparison table."""
    print(f"\n{'='*70}")
    print("Cross-Domain Manifold Comparison:")
    print(f"  {'Domain':<20s} {'Scaffolding%':>12s} {'Entropy Ratio':>14s} {'Content Entropy':>16s}")
    print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*16}")
    for r in sorted(results, key=lambda x: x["total_scaffolding_pct"]):
        print(f"  {r['domain']:<20s} {r['total_scaffolding_pct']:>11.1f}% "
              f"{r['entropy_ratio']:>13.2f}x {r['mean_entropy_content']:>15.6f}")


def main():
    parser = argparse.ArgumentParser(description="Token classification analysis for entropy traces")
    parser.add_argument("--code", help="Code entropy traces JSONL file")
    parser.add_argument("--math1", help="Math proof traces batch 1 JSONL file")
    parser.add_argument("--math2", help="Math proof traces batch 2 JSONL file")
    parser.add_argument("--output", help="Output JSON file for results", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find trace files
    code_file = args.code or find_latest_file(os.path.join(project_root, "code_entropy_traces_*.jsonl"))
    math1_file = args.math1 or find_latest_file(os.path.join(project_root, "math_proof_traces_2*.jsonl"))
    math2_file = args.math2 or find_latest_file(os.path.join(project_root, "math_proof_traces_batch2_*.jsonl"))

    all_results = []

    # ─── Code Traces ──────────────────────────────────────────────────────
    if code_file:
        print(f"Code traces: {code_file}")
        code_traces = load_traces(code_file)
        code_result = analyze_traces(code_traces, classify_token_code, "Code Generation")
        print_domain_summary(code_result)

        # Spike concentration test
        spike_test = spike_concentration_test(code_traces, classify_token_code)
        print(f"\n  Spike Concentration Test (code):")
        if "error" not in spike_test:
            print(f"    Spikes on content: {spike_test['mean_spike_content_pct']:.1f}% "
                  f"(base rate: {spike_test['mean_base_rate_pct']:.1f}%)")
            print(f"    Excess: +{spike_test['excess_pct']:.1f} percentage points")
            print(f"    Paired t-test: t={spike_test['paired_t_stat']:.2f}, p={spike_test['paired_p_value']:.2e}")
            print(f"    Binomial test: {spike_test['binomial_k']}/{spike_test['binomial_n']} "
                  f"vs p0={spike_test['binomial_p0']:.3f}, p={spike_test['binomial_p_value']:.2e}")
            code_result["spike_test"] = spike_test
        else:
            print(f"    {spike_test['error']}")

        all_results.append(code_result)
    else:
        print("No code trace file found.")

    # ─── Math Proof Traces ────────────────────────────────────────────────
    math_traces_knowable = []
    math_traces_all = []

    if math1_file:
        print(f"\nMath traces (batch 1 - knowable): {math1_file}")
        math1 = load_traces(math1_file)
        # Batch 1 has no category field — all are knowable
        for t in math1:
            t.setdefault("category", "knowable")
        math_traces_knowable = math1
        math_traces_all.extend(math1)

    if math2_file:
        print(f"Math traces (batch 2 - mixed): {math2_file}")
        math2 = load_traces(math2_file)
        math_traces_all.extend(math2)

    if math_traces_knowable:
        knowable_result = analyze_traces(math_traces_knowable, classify_token_math, "Math Proofs (knowable)")
        print_domain_summary(knowable_result)

        # Spike concentration test for math
        spike_test_math = spike_concentration_test(math_traces_knowable, classify_token_math)
        if "error" not in spike_test_math:
            print(f"\n  Spike Concentration Test (math proofs):")
            print(f"    Spikes on content: {spike_test_math['mean_spike_content_pct']:.1f}% "
                  f"(base rate: {spike_test_math['mean_base_rate_pct']:.1f}%)")
            print(f"    Excess: +{spike_test_math['excess_pct']:.1f} percentage points")
            print(f"    Paired t-test: t={spike_test_math['paired_t_stat']:.2f}, "
                  f"p={spike_test_math['paired_p_value']:.2e}")
            knowable_result["spike_test"] = spike_test_math

        all_results.append(knowable_result)

    if math_traces_all and len(math_traces_all) > len(math_traces_knowable):
        all_math_result = analyze_traces(math_traces_all, classify_token_math, "Math Proofs (all)")
        print_domain_summary(all_math_result)
        print_math_category_breakdown(math_traces_all)
        all_results.append(all_math_result)

    # ─── Cross-Domain Table ───────────────────────────────────────────────
    if len(all_results) >= 2:
        print_cross_domain_table(all_results)

    # ─── Per-Trace Detail Table ───────────────────────────────────────────
    if math_traces_all:
        print(f"\n{'='*70}")
        print("Per-Trace Detail (math proofs):")
        print(f"  {'Name':<35s} {'Category':<15s} {'Tokens':>6s} {'Scaff%':>7s} {'MeanEnt':>8s}")
        print(f"  {'-'*35} {'-'*15} {'-'*6} {'-'*7} {'-'*8}")
        for result in all_results:
            for pt in result["per_trace"]:
                print(f"  {pt['name']:<35s} {pt['category']:<15s} "
                      f"{pt['n_tokens']:>6d} {pt['scaffolding_pct']:>6.1f}% "
                      f"{pt['mean_entropy']:>8.3f}")

    # ─── Save JSON output ─────────────────────────────────────────────────
    if args.output:
        # Strip non-serializable items
        output_data = {
            "domains": [{k: v for k, v in r.items() if k != "per_trace"} for r in all_results],
            "per_trace": {r["domain"]: r["per_trace"] for r in all_results},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
