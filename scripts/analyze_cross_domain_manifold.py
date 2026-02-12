#!/usr/bin/env python3
"""Cross-domain manifold analysis: scaffolding density × entropy characteristics.

Combines four trace domains to map the format-constraint manifold:
  1. Natural text (exp27c traces) — minimal scaffolding (~15%)
  2. Citations (exp27c citation traces) — medium scaffolding (~45%)
  3. Code generation (code_entropy_traces) — high scaffolding (~58%)
  4. Math proofs (math_proof_traces batch 1) — highest scaffolding (~67%)

For each domain, measures:
  - Three-tier token classification (syntactic, semantic scaffolding, content)
  - Per-tier mean entropy
  - Entropy ratio (content/scaffolding)
  - Spike concentration on content tokens
  - Gini coefficient of entropy distribution

Usage:
    python scripts/analyze_cross_domain_manifold.py [--text FILE] [--code FILE] [--math FILE]
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


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


def gini_coefficient(values):
    """Compute Gini coefficient of a list of values."""
    if not values:
        return 0.0
    arr = np.array(sorted(values))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


# ─── Text Token Classifier ───────────────────────────────────────────────────

TEXT_SYNTACTIC = {
    ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", '"', "'",
    "-", "--", "—", "/", "\\", "@", "#", "$", "%", "&", "*", "+", "=",
    "\n", "\r\n",
}

TEXT_SEMANTIC = {
    # Determiners, prepositions, conjunctions, auxiliaries
    "the", "The", "a", "A", "an", "An",
    "of", "Of", "in", "In", "to", "To", "for", "For", "with", "With",
    "on", "On", "at", "At", "by", "By", "from", "From", "as", "As",
    "is", "Is", "are", "Are", "was", "Was", "were", "Were", "be", "Be",
    "been", "being", "have", "Have", "has", "Has", "had", "Had",
    "do", "does", "did", "will", "Will", "would", "Would", "could",
    "should", "can", "may", "might", "shall",
    "and", "And", "or", "Or", "but", "But", "not", "Not",
    "that", "That", "which", "Which", "this", "This", "it", "It",
    "its", "their", "they", "them",
}


def classify_token_text(token_text):
    """Classify a natural text token."""
    stripped = token_text.strip()
    if not stripped:
        return "syntactic"
    if stripped in TEXT_SYNTACTIC:
        return "syntactic"
    if stripped in TEXT_SEMANTIC:
        return "semantic_scaffolding"
    return "content"


# ─── Code Token Classifier (from analyze_token_classification.py) ────────────

PYTHON_SYNTAX = {
    "def", "return", "if", "elif", "else", "for", "while", "in", "not", "and",
    "or", "is", "class", "import", "from", "as", "with", "try", "except",
    "finally", "raise", "pass", "break", "continue", "yield", "lambda",
    "global", "nonlocal", "assert", "del", "True", "False", "None",
    "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "//", "%",
    "**", "&", "|", "^", "~", "<<", ">>", "+=", "-=", "*=", "/=",
    "(", ")", "[", "]", "{", "}", ":", ",", ".", ";", "@", "->",
    "```", "```python",
}

PYTHON_SEMANTIC = {
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


def classify_token_code(token_text):
    stripped = token_text.strip()
    if not stripped:
        return "syntactic"
    if stripped.startswith("\n"):
        return "syntactic"
    if stripped.startswith("```"):
        return "syntactic"
    if stripped in PYTHON_SYNTAX:
        return "syntactic"
    if stripped in PYTHON_SEMANTIC:
        return "semantic_scaffolding"
    if stripped.isdigit():
        return "semantic_scaffolding"
    return "content"


# ─── Math Proof Token Classifier ─────────────────────────────────────────────

MATH_SEMANTIC = {
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


def classify_token_math(token_text):
    stripped = token_text.strip()
    if not stripped:
        return "syntactic"
    if stripped.startswith("\n"):
        return "syntactic"
    if stripped in {".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "=",
                    "<", ">", "+", "-", "*", "/", "^", "_", "$", "$$",
                    "**", "#", "##", "###", "####", ">", "---", "```",
                    "≤", "≥", "≠", "→", "⇒", "∈", "⊂", "⊆", "∪", "∩"}:
        return "syntactic"
    if stripped.startswith("\\") and len(stripped) <= 15:
        return "syntactic"
    if stripped in MATH_SEMANTIC:
        return "semantic_scaffolding"
    if len(stripped) == 1 and stripped.isalpha():
        return "semantic_scaffolding"
    if stripped.isdigit():
        return "semantic_scaffolding"
    return "content"


def analyze_domain(traces, classifier_fn, domain_name, min_tokens=5):
    """Analyze one domain: token classification + entropy stats."""
    all_classes = []
    all_ents_by_class = defaultdict(list)
    all_ents = []

    for trace in traces:
        texts = trace.get("token_texts", [])
        ents = trace.get("token_entropies", [])
        min_len = min(len(texts), len(ents))
        if min_len < min_tokens:
            continue

        for i in range(min_len):
            cls = classifier_fn(texts[i])
            all_classes.append(cls)
            all_ents_by_class[cls].append(ents[i])
            all_ents.append(ents[i])

    if not all_classes:
        return None

    total = len(all_classes)
    n_syn = all_classes.count("syntactic")
    n_sem = all_classes.count("semantic_scaffolding")
    n_con = all_classes.count("content")

    syn_mean = statistics.mean(all_ents_by_class["syntactic"]) if all_ents_by_class["syntactic"] else 0
    sem_mean = statistics.mean(all_ents_by_class["semantic_scaffolding"]) if all_ents_by_class["semantic_scaffolding"] else 0
    con_mean = statistics.mean(all_ents_by_class["content"]) if all_ents_by_class["content"] else 0
    scaff_ents = all_ents_by_class["syntactic"] + all_ents_by_class["semantic_scaffolding"]
    scaff_mean = statistics.mean(scaff_ents) if scaff_ents else 0
    entropy_ratio = con_mean / scaff_mean if scaff_mean > 0 else float("inf")

    # Gini coefficient
    gini = gini_coefficient(all_ents)

    # Spike concentration
    sorted_ents = sorted(all_ents)
    threshold = sorted_ents[int(len(sorted_ents) * 0.9)]
    spike_total = sum(1 for e in all_ents if e > threshold)
    spike_content = 0
    idx = 0
    for trace in traces:
        texts = trace.get("token_texts", [])
        ents = trace.get("token_entropies", [])
        min_len = min(len(texts), len(ents))
        if min_len < min_tokens:
            continue
        for i in range(min_len):
            if ents[i] > threshold and classifier_fn(texts[i]) == "content":
                spike_content += 1
    content_base_rate = n_con / total if total > 0 else 0
    spike_content_pct = spike_content / spike_total * 100 if spike_total > 0 else 0

    return {
        "domain": domain_name,
        "n_traces": len(traces),
        "total_tokens": total,
        "syntactic_pct": n_syn / total * 100,
        "semantic_scaffolding_pct": n_sem / total * 100,
        "content_pct": n_con / total * 100,
        "total_scaffolding_pct": (n_syn + n_sem) / total * 100,
        "mean_entropy_syntactic": syn_mean,
        "mean_entropy_semantic": sem_mean,
        "mean_entropy_content": con_mean,
        "mean_entropy_scaffolding": scaff_mean,
        "entropy_ratio": entropy_ratio,
        "gini": gini,
        "spike_content_pct": spike_content_pct,
        "content_base_rate_pct": content_base_rate * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-domain manifold analysis")
    parser.add_argument("--text", help="Text traces JSONL (exp27c)")
    parser.add_argument("--code", help="Code entropy traces JSONL")
    parser.add_argument("--math", help="Math proof traces JSONL (batch 1, knowable)")
    parser.add_argument("--exp-h", help="Experiment H traces JSONL (if available)")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    text_file = args.text or find_latest_file(os.path.join(project_root, "exp27c_traces_*.jsonl"))
    code_file = args.code or find_latest_file(os.path.join(project_root, "code_entropy_traces_*.jsonl"))
    math_file = args.math or find_latest_file(os.path.join(project_root, "math_proof_traces_2*.jsonl"))
    exp_h_file = args.exp_h or find_latest_file(os.path.join(project_root, "experiment_H_coding_styles_*.jsonl"))

    results = []

    # ─── Natural Text ─────────────────────────────────────────────────────
    if text_file:
        print(f"Text traces: {text_file}")
        text_traces = load_traces(text_file)

        # Split by knowable vs unknowable
        knowable = [t for t in text_traces if t.get("is_knowable")]
        unknowable = [t for t in text_traces if not t.get("is_knowable")]

        text_result = analyze_domain(text_traces, classify_token_text, "Text (all)")
        if text_result:
            results.append(text_result)

        if knowable:
            k_result = analyze_domain(knowable, classify_token_text, "Text (knowable)")
            if k_result:
                results.append(k_result)
        if unknowable:
            u_result = analyze_domain(unknowable, classify_token_text, "Text (unknowable)")
            if u_result:
                results.append(u_result)

        # Citation subset
        citations = [t for t in text_traces if t.get("is_citation")]
        if citations:
            c_result = analyze_domain(citations, classify_token_text, "Text (citations)")
            if c_result:
                results.append(c_result)

    # ─── Code Generation ──────────────────────────────────────────────────
    if code_file:
        print(f"Code traces: {code_file}")
        code_traces = load_traces(code_file)
        code_result = analyze_domain(code_traces, classify_token_code, "Code Generation")
        if code_result:
            results.append(code_result)

    # ─── Math Proofs ──────────────────────────────────────────────────────
    if math_file:
        print(f"Math traces: {math_file}")
        math_traces = load_traces(math_file)
        math_result = analyze_domain(math_traces, classify_token_math, "Math Proofs")
        if math_result:
            results.append(math_result)

    # ─── Experiment H ─────────────────────────────────────────────────────
    if exp_h_file:
        print(f"Exp H traces: {exp_h_file}")
        h_traces = load_traces(exp_h_file)
        compact = [t for t in h_traces if t.get("style") == "compact"]
        documented = [t for t in h_traces if t.get("style") == "documented"]
        if compact:
            h_compact = analyze_domain(compact, classify_token_code, "Code (compact)")
            if h_compact:
                results.append(h_compact)
        if documented:
            h_documented = analyze_domain(documented, classify_token_code, "Code (documented)")
            if h_documented:
                results.append(h_documented)

    # ─── Output ───────────────────────────────────────────────────────────
    if not results:
        print("No data found.")
        return

    print(f"\n{'='*90}")
    print("Cross-Domain Format-Constraint Manifold")
    print(f"{'='*90}")
    print(f"  {'Domain':<22s} {'Scaff%':>7s} {'Syn%':>6s} {'SemSc%':>7s} "
          f"{'Cont%':>6s} {'E.Ratio':>8s} {'Gini':>6s} {'Spike%':>7s} {'Base%':>6s}")
    print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")

    for r in sorted(results, key=lambda x: x["total_scaffolding_pct"]):
        print(f"  {r['domain']:<22s} {r['total_scaffolding_pct']:>6.1f}% "
              f"{r['syntactic_pct']:>5.1f}% {r['semantic_scaffolding_pct']:>6.1f}% "
              f"{r['content_pct']:>5.1f}% {r['entropy_ratio']:>7.2f}x "
              f"{r['gini']:>5.3f} {r['spike_content_pct']:>6.1f}% "
              f"{r['content_base_rate_pct']:>5.1f}%")

    # Per-tier entropy detail
    print(f"\n{'='*90}")
    print("Per-Tier Mean Entropy")
    print(f"{'='*90}")
    print(f"  {'Domain':<22s} {'Syntactic':>12s} {'Sem.Scaff':>12s} {'Content':>12s} {'Overall':>12s}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for r in sorted(results, key=lambda x: x["total_scaffolding_pct"]):
        overall = (r["mean_entropy_syntactic"] * r["syntactic_pct"] +
                   r["mean_entropy_semantic"] * r["semantic_scaffolding_pct"] +
                   r["mean_entropy_content"] * r["content_pct"]) / 100
        print(f"  {r['domain']:<22s} {r['mean_entropy_syntactic']:>12.6f} "
              f"{r['mean_entropy_semantic']:>12.6f} {r['mean_entropy_content']:>12.6f} "
              f"{overall:>12.6f}")

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
