#!/usr/bin/env python3
"""A/B comparison framework for evaluating multiple versions of a paper section.

Uses LLM judges on OpenRouter to perform blind evaluation of 2-4 candidate
versions of a paper section. Each judge evaluates versions individually
(blind, one at a time), then performs a comparative ranking with all versions
side by side (still blind-labeled).

This extends the daily review pipeline's infrastructure (OpenRouter API calls,
JSONL output with provenance) but is standalone — no imports from other scripts.

Architecture:
  - Phase 1: Individual blind evaluation (each judge × each version)
  - Phase 2: Comparative ranking (each judge sees all versions)
  - Output: Incremental JSONL + human-readable summary table

Usage:
    # Full comparison of 4 intro versions
    python scripts/ab_comparison.py \\
      --versions papers/sosp/intro_paxos.tex papers/sosp/intro_restructured.tex \\
                 papers/sosp/intro_judge_generated.tex papers/sosp/intro.tex \\
      --transition-context papers/sosp/background.tex \\
      --output-dir reviews/

    # Dry run to see configuration without API calls
    python scripts/ab_comparison.py \\
      --versions papers/sosp/intro.tex papers/sosp/intro_restructured.tex \\
      --dry-run

    # Custom models and perspectives
    python scripts/ab_comparison.py \\
      --versions papers/sosp/intro.tex papers/sosp/intro_restructured.tex \\
      --models google/gemini-2.5-pro-preview deepseek/deepseek-chat-v3-0324 \\
      --perspectives systems_reviewer narrative_reviewer

Environment:
    OPENROUTER_API_KEY: Required. Set in environment or in .env file
    in the project root.

Output:
    reviews/
      ab_comparison_YYYYMMDD_HHMMSS.jsonl   # All records (provenance + evaluations)
"""

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration: default judge panel
# ---------------------------------------------------------------------------

# Models with fixed temperatures for controlled comparison.
# Each tuple: (model_id, temperature)
DEFAULT_MODELS = [
    ("google/gemini-2.5-pro-preview", 0.3),
    ("deepseek/deepseek-chat-v3-0324", 0.7),
    ("qwen/qwen-2.5-72b-instruct", 1.0),
]

# Perspectives (system prompts for each judge role).
# Keys are perspective names; values are system prompts.
DEFAULT_PERSPECTIVES = {
    "systems_reviewer": (
        "You are a senior systems researcher reviewing a paper section for "
        "SOSP 2026. You care about: Is the problem well-motivated for systems? "
        "Is the contribution architectural? Would a distributed systems "
        "researcher follow this argument?"
    ),
    "ml_reviewer": (
        "You are an ML researcher reviewing a systems paper that makes claims "
        "about language model internals. You care about: Are the ML claims "
        "correct? Are the evaluation metrics appropriate? Are there missing "
        "baselines or confounds?"
    ),
    "narrative_reviewer": (
        "You are an editor evaluating the narrative quality of a paper section. "
        "You care about: Does the argument flow linearly? Does each paragraph "
        "motivate the next? Is the reader ever confused about why something "
        "appears? Are claims supported before being referenced?"
    ),
    "junior_systems_phd": (
        "You are a first-year PhD student in operating systems. You have strong "
        "background in distributed systems, consensus protocols, and OS internals. "
        "You have NEVER taken a machine learning course. You do not know what "
        "entropy means in an ML context, what RLHF is, or how transformers work. "
        "You are reading this paper to decide if it's relevant to your research. "
        "Flag anything you cannot follow. Be honest about where you get lost."
    ),
    "senior_ml_no_systems": (
        "You are a senior ML researcher who has published at NeurIPS, ICML, and "
        "ICLR. You know transformers, RLHF, and language model internals deeply. "
        "You have NEVER attended a systems conference (SOSP, OSDI, EuroSys). You "
        "do not think in terms of interfaces, architectural boundaries, or system "
        "design patterns. You care about: Are the ML claims technically correct? "
        "Is the evaluation rigorous by ML standards? Does the systems framing add "
        "insight or just jargon?"
    ),
}

# Maximum retries for empty or failed responses
MAX_RETRIES = 1

# Blind labels assigned to versions (in order, before shuffling)
BLIND_LABELS = ["Version A", "Version B", "Version C", "Version D"]


# ---------------------------------------------------------------------------
# OpenRouter API (standalone copy from daily_review_pipeline.py)
# ---------------------------------------------------------------------------

def call_openrouter(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> dict:
    """Call OpenRouter API and return full response dict.

    Matches the daily_review_pipeline.py call pattern: same headers, same
    error handling, same response structure.
    """
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fsgeek/ai-honesty",
        "X-Title": "AI Honesty A/B Comparison Pipeline",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    start_ms = time.monotonic_ns() // 1_000_000
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        if resp.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
                "latency_ms": latency_ms,
                "response_text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "success": True,
            "error": None,
            "latency_ms": latency_ms,
            "response_text": message.get("content", ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish_reason": choice.get("finish_reason", ""),
            "model_id_returned": data.get("model", model),
        }
    except Exception as e:
        latency_ms = (time.monotonic_ns() // 1_000_000) - start_ms
        return {
            "success": False,
            "error": str(e),
            "latency_ms": latency_ms,
            "response_text": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def call_with_retry(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> dict:
    """Call OpenRouter with retry on failure or empty response."""
    result = call_openrouter(model, temperature, system_prompt, user_prompt, api_key)

    retries = 0
    while retries < MAX_RETRIES and (
        not result["success"] or len(result.get("response_text", "")) == 0
    ):
        retries += 1
        reason = "empty response" if result["success"] else result.get("error", "unknown")
        print(f"    RETRY {retries}/{MAX_RETRIES}: {reason}...")
        time.sleep(3)
        result = call_openrouter(model, temperature, system_prompt, user_prompt, api_key)

    return result


def load_env_file(project_root: Path):
    """Load .env file if it exists (same logic as daily pipeline)."""
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# File and version handling
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    """SHA256 hash of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


def read_version_file(path: str) -> str:
    """Read a .tex file and return its contents."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Version file not found: {path}")
        sys.exit(1)
    return p.read_text()


def read_transition_context(path: str, max_lines: int = 30) -> str:
    """Read the first N lines of a transition context file."""
    p = Path(path)
    if not p.exists():
        print(f"WARNING: Transition context file not found: {path}")
        return ""
    lines = p.read_text().splitlines()
    return "\n".join(lines[:max_lines])


def create_blind_mapping(version_paths: list[str]) -> dict:
    """Create a randomized mapping from blind labels to file paths.

    Returns:
        {
            "label_to_file": {"Version A": "path/to/file1.tex", ...},
            "file_to_label": {"path/to/file1.tex": "Version A", ...},
            "shuffle_seed": <int>,
        }
    """
    n = len(version_paths)
    if n < 2 or n > 4:
        print(f"ERROR: Need 2-4 versions, got {n}")
        sys.exit(1)

    labels = BLIND_LABELS[:n]
    # Shuffle version paths (not labels) so labels stay A, B, C, D
    # but which file gets which label is random.
    shuffled_paths = list(version_paths)
    shuffle_seed = random.randint(0, 2**31)
    rng = random.Random(shuffle_seed)
    rng.shuffle(shuffled_paths)

    label_to_file = {}
    file_to_label = {}
    for label, path in zip(labels, shuffled_paths):
        label_to_file[label] = path
        file_to_label[path] = label

    return {
        "label_to_file": label_to_file,
        "file_to_label": file_to_label,
        "shuffle_seed": shuffle_seed,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_individual_eval_prompt(
    blind_label: str,
    section_text: str,
    transition_context: Optional[str] = None,
) -> str:
    """Build the prompt for a single blind evaluation of one version."""
    parts = [
        f"You are evaluating a paper section labeled **{blind_label}**.\n",
        "Do NOT speculate about the origin of this text. Evaluate it purely "
        "on its merits.\n",
        f"--- BEGIN {blind_label} ---\n",
        section_text,
        f"\n--- END {blind_label} ---\n",
    ]

    if transition_context:
        parts.append(
            "\nFor context, here are the first 30 lines of the NEXT section "
            "in the paper (the section that follows the one you are evaluating). "
            "Use this to assess how well the evaluated section transitions into "
            "what comes next.\n"
        )
        parts.append("--- BEGIN NEXT SECTION (context only) ---\n")
        parts.append(transition_context)
        parts.append("\n--- END NEXT SECTION ---\n")

    parts.append(
        "\nPlease evaluate this section on the following criteria. "
        "For each score, use a 1-10 scale where 1 is worst and 10 is best.\n\n"
        "1. **Clarity** (1-10): Is the writing clear? Can a reader follow "
        "the argument without re-reading?\n"
        "2. **Motivation** (1-10): Does this section make you want to read "
        "the rest of the paper? Does it establish urgency?\n"
        "3. **Venue Fit** (1-10): Does this belong at SOSP? Is it framed "
        "as a systems contribution?\n"
        "4. **Narrative Flow** (1-10): Does the argument progress linearly? "
        "Does each paragraph motivate the next?"
    )

    if transition_context:
        parts.append(
            " Include in this score how well the section transitions "
            "into the next section shown above."
        )

    parts.append(
        "\n5. **Strongest Element**: In one sentence, what is the single "
        "strongest aspect of this section?\n"
        "6. **Weakest Element**: In one sentence, what is the single "
        "weakest aspect of this section?\n"
        "\nProvide your response in the following format:\n"
        "```\n"
        "Clarity: <score>/10\n"
        "Motivation: <score>/10\n"
        "Venue Fit: <score>/10\n"
        "Narrative Flow: <score>/10\n"
        "Strongest: <one sentence>\n"
        "Weakest: <one sentence>\n"
        "```\n"
        "\nAfter the structured scores, you may add a brief paragraph of "
        "additional commentary if you have observations that do not fit "
        "the above categories."
    )

    return "\n".join(parts)


def build_comparison_prompt(
    version_texts: dict[str, str],
    transition_context: Optional[str] = None,
) -> str:
    """Build the prompt for comparative ranking of all versions.

    Args:
        version_texts: {blind_label: section_text} for all versions
        transition_context: optional next-section context
    """
    labels = sorted(version_texts.keys())  # Alphabetical: A, B, C, D

    parts = [
        "You are now comparing multiple versions of the same paper section. "
        "Each version is labeled with a blind label. Do NOT speculate about "
        "the origin of any version. Evaluate purely on merits.\n",
    ]

    for label in labels:
        parts.append(f"\n--- BEGIN {label} ---\n")
        parts.append(version_texts[label])
        parts.append(f"\n--- END {label} ---\n")

    if transition_context:
        parts.append(
            "\nFor context, the next section in the paper begins:\n"
            "--- BEGIN NEXT SECTION (context only) ---\n"
        )
        parts.append(transition_context)
        parts.append("\n--- END NEXT SECTION ---\n")

    label_list = ", ".join(labels)
    parts.append(
        f"\nYou have now read all versions: {label_list}.\n\n"
        "Please provide:\n"
        "1. **Ranking**: Order the versions from best to worst. For each, "
        "give a one-sentence justification.\n"
        "2. **Recommendation**: Which single version would you recommend for "
        "SOSP submission, and why (one sentence)?\n"
        "3. **Key Differentiator**: What is the most important difference "
        "between the best and worst versions (one sentence)?\n"
        "\nProvide your response in the following format:\n"
        "```\n"
        "Ranking:\n"
        "1. <label> - <justification>\n"
        "2. <label> - <justification>\n"
    )

    # Add placeholder lines for remaining versions
    for i in range(2, len(labels)):
        parts.append(f"{i+1}. <label> - <justification>\n")

    parts.append(
        "Recommendation: <label> - <reason>\n"
        "Key Differentiator: <one sentence>\n"
        "```\n"
        "\nAfter the structured response, you may add a brief paragraph of "
        "additional commentary."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def parse_individual_scores(response_text: str) -> dict:
    """Extract structured scores from an individual evaluation response.

    Returns a dict with keys: clarity, motivation, venue_fit, narrative_flow,
    strongest, weakest. Missing values are None.
    """
    scores = {
        "clarity": None,
        "motivation": None,
        "venue_fit": None,
        "narrative_flow": None,
        "strongest": None,
        "weakest": None,
    }

    # Score patterns: "Clarity: 7/10" or "Clarity: 7" or "**Clarity**: 7/10"
    score_patterns = {
        "clarity": r"[Cc]larity[:\s*]+(\d+)\s*/?\s*10?",
        "motivation": r"[Mm]otivation[:\s*]+(\d+)\s*/?\s*10?",
        "venue_fit": r"[Vv]enue\s*[Ff]it[:\s*]+(\d+)\s*/?\s*10?",
        "narrative_flow": r"[Nn]arrative\s*[Ff]low[:\s*]+(\d+)\s*/?\s*10?",
    }

    for key, pattern in score_patterns.items():
        match = re.search(pattern, response_text)
        if match:
            val = int(match.group(1))
            if 1 <= val <= 10:
                scores[key] = val

    # Text fields
    strongest_match = re.search(
        r"[Ss]trongest[:\s*]+(.+?)(?:\n|$)", response_text
    )
    if strongest_match:
        scores["strongest"] = strongest_match.group(1).strip()

    weakest_match = re.search(
        r"[Ww]eakest[:\s*]+(.+?)(?:\n|$)", response_text
    )
    if weakest_match:
        scores["weakest"] = weakest_match.group(1).strip()

    return scores


def parse_comparison_ranking(response_text: str, labels: list[str]) -> dict:
    """Extract ranking and recommendation from a comparison response.

    Returns:
        {
            "ranking": ["Version A", "Version B", ...],  # best to worst
            "recommendation": "Version A",
            "key_differentiator": "...",
        }
    """
    result = {
        "ranking": [],
        "recommendation": None,
        "key_differentiator": None,
    }

    # Parse ranking lines: "1. Version A - justification"
    # or "1. **Version A** - justification"
    ranking_pattern = r"(\d+)\.\s*\*{0,2}(Version [A-D])\*{0,2}\s*[-:]\s*(.+?)(?:\n|$)"
    matches = re.findall(ranking_pattern, response_text)
    if matches:
        # Sort by rank number, extract label
        sorted_matches = sorted(matches, key=lambda m: int(m[0]))
        result["ranking"] = [m[1] for m in sorted_matches]

    # Parse recommendation
    rec_match = re.search(
        r"[Rr]ecommendation[:\s*]+\*{0,2}(Version [A-D])\*{0,2}\s*[-:]\s*(.+?)(?:\n|$)",
        response_text,
    )
    if rec_match:
        result["recommendation"] = rec_match.group(1)

    # Parse key differentiator
    diff_match = re.search(
        r"[Kk]ey\s*[Dd]ifferentiator[:\s*]+(.+?)(?:\n|$)", response_text
    )
    if diff_match:
        result["key_differentiator"] = diff_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def append_jsonl(output_file: Path, record: dict):
    """Append a single JSON record to the output file."""
    with open(output_file, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def print_summary(
    records: list[dict],
    blind_mapping: dict,
    labels: list[str],
):
    """Print a human-readable summary table to stdout."""
    # Separate individual and comparison records
    individual = [r for r in records if r.get("record_type") == "individual_eval"]
    comparisons = [r for r in records if r.get("record_type") == "comparison"]

    criteria = ["clarity", "motivation", "venue_fit", "narrative_flow"]

    print("\n" + "=" * 78)
    print("A/B COMPARISON SUMMARY")
    print("=" * 78)

    # Blind label mapping
    print("\nBlind Label Mapping:")
    for label in labels:
        filepath = blind_mapping["label_to_file"][label]
        print(f"  {label}: {Path(filepath).name}")

    # Per-version score table
    print(f"\n{'':>20s}", end="")
    for label in labels:
        print(f"  {label:>12s}", end="")
    print()
    print("-" * (20 + 14 * len(labels)))

    for criterion in criteria:
        print(f"  {criterion:>18s}", end="")
        for label in labels:
            version_scores = [
                r["parsed_scores"][criterion]
                for r in individual
                if r["blind_label"] == label
                and r["parsed_scores"].get(criterion) is not None
            ]
            if version_scores:
                mean = statistics.mean(version_scores)
                print(f"  {mean:>10.1f}/10", end="")
            else:
                print(f"  {'n/a':>12s}", end="")
        print()

    # Composite score (mean of all four criteria)
    print("-" * (20 + 14 * len(labels)))
    print(f"  {'COMPOSITE':>18s}", end="")
    for label in labels:
        all_scores = []
        for criterion in criteria:
            version_scores = [
                r["parsed_scores"][criterion]
                for r in individual
                if r["blind_label"] == label
                and r["parsed_scores"].get(criterion) is not None
            ]
            all_scores.extend(version_scores)
        if all_scores:
            mean = statistics.mean(all_scores)
            print(f"  {mean:>10.1f}/10", end="")
        else:
            print(f"  {'n/a':>12s}", end="")
    print()

    # Individual judge scores (detailed breakdown)
    print(f"\nDetailed scores by judge:")
    print(f"  {'Judge':>35s}  {'Label':>10s}  {'Clar':>4s}  {'Motv':>4s}  "
          f"{'Vnue':>4s}  {'Narr':>4s}")
    print("  " + "-" * 75)
    for r in individual:
        judge_name = f"{r['perspective']}@{r['model_id'].split('/')[-1]}"
        label = r["blind_label"]
        s = r["parsed_scores"]
        c = s.get("clarity") or "-"
        m = s.get("motivation") or "-"
        v = s.get("venue_fit") or "-"
        n = s.get("narrative_flow") or "-"
        print(f"  {judge_name:>35s}  {label:>10s}  {c:>4}  {m:>4}  {v:>4}  {n:>4}")

    # Comparison rankings
    if comparisons:
        print(f"\nComparative Rankings:")
        print(f"  {'Judge':>35s}  {'Ranking (best -> worst)':40s}  {'Recommend':>10s}")
        print("  " + "-" * 90)
        for r in comparisons:
            judge_name = f"{r['perspective']}@{r['model_id'].split('/')[-1]}"
            ranking_str = " > ".join(r["parsed_ranking"].get("ranking", ["?"])) or "(unparsed)"
            rec = r["parsed_ranking"].get("recommendation") or "?"
            print(f"  {judge_name:>35s}  {ranking_str:40s}  {rec:>10s}")

        # Aggregate: count first-place finishes
        print(f"\n  First-place votes:")
        first_place_counts = {label: 0 for label in labels}
        for r in comparisons:
            ranking = r["parsed_ranking"].get("ranking", [])
            if ranking:
                first = ranking[0]
                if first in first_place_counts:
                    first_place_counts[first] += 1
        for label in labels:
            filename = Path(blind_mapping["label_to_file"][label]).name
            count = first_place_counts[label]
            bar = "#" * count
            print(f"    {label} ({filename:>30s}): {count:2d} {bar}")

        # Aggregate: recommendation counts
        print(f"\n  Recommendation votes:")
        rec_counts = {label: 0 for label in labels}
        for r in comparisons:
            rec = r["parsed_ranking"].get("recommendation")
            if rec and rec in rec_counts:
                rec_counts[rec] += 1
        for label in labels:
            filename = Path(blind_mapping["label_to_file"][label]).name
            count = rec_counts[label]
            bar = "#" * count
            print(f"    {label} ({filename:>30s}): {count:2d} {bar}")

    # Strongest/weakest per version
    print(f"\nStrongest elements (by version):")
    for label in labels:
        print(f"  {label}:")
        for r in individual:
            if r["blind_label"] == label and r["parsed_scores"].get("strongest"):
                judge = f"{r['perspective']}@{r['model_id'].split('/')[-1]}"
                print(f"    [{judge}] {r['parsed_scores']['strongest']}")

    print(f"\nWeakest elements (by version):")
    for label in labels:
        print(f"  {label}:")
        for r in individual:
            if r["blind_label"] == label and r["parsed_scores"].get("weakest"):
                judge = f"{r['perspective']}@{r['model_id'].split('/')[-1]}"
                print(f"    [{judge}] {r['parsed_scores']['weakest']}")

    # Token/cost summary
    total_tokens = sum(
        r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
        for r in records
        if r.get("record_type") in ("individual_eval", "comparison")
    )
    total_latency = sum(
        r.get("latency_ms", 0)
        for r in records
        if r.get("record_type") in ("individual_eval", "comparison")
    )
    n_calls = sum(
        1 for r in records
        if r.get("record_type") in ("individual_eval", "comparison")
    )
    n_success = sum(
        1 for r in records
        if r.get("record_type") in ("individual_eval", "comparison")
        and r.get("success")
    )

    print(f"\nAPI Usage:")
    print(f"  Total calls: {n_calls} ({n_success} successful)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total latency: {total_latency / 1000:.1f}s")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_comparison(args):
    """Execute the A/B comparison pipeline."""
    project_root = Path(__file__).parent.parent
    load_env_file(project_root)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not found.")
        print("Set it in .env or as an environment variable.")
        print("Get a key at https://openrouter.ai/keys")
        sys.exit(1)

    # --- Resolve version files ---
    version_paths = []
    for v in args.versions:
        p = Path(v)
        if not p.is_absolute():
            p = project_root / p
        version_paths.append(str(p))

    # --- Resolve transition context ---
    transition_context = None
    if args.transition_context:
        tc_path = Path(args.transition_context)
        if not tc_path.is_absolute():
            tc_path = project_root / tc_path
        transition_context = read_transition_context(str(tc_path))
        if not transition_context:
            print("WARNING: Transition context is empty, proceeding without it.")
            transition_context = None

    # --- Read version texts ---
    version_texts = {}
    for path in version_paths:
        version_texts[path] = read_version_file(path)

    # --- Create blind mapping ---
    blind_mapping = create_blind_mapping(version_paths)
    labels = sorted(blind_mapping["label_to_file"].keys())

    # --- Configure judge panel ---
    # Models: use --models override or defaults
    if args.models:
        # User-provided models get default temperatures distributed evenly
        n_models = len(args.models)
        temps = [round(0.3 + (0.7 * i / max(n_models - 1, 1)), 2) for i in range(n_models)]
        models = list(zip(args.models, temps))
    else:
        models = list(DEFAULT_MODELS)

    # Perspectives: use --perspectives override or defaults
    if args.perspectives:
        perspectives = {}
        for name in args.perspectives:
            if name in DEFAULT_PERSPECTIVES:
                perspectives[name] = DEFAULT_PERSPECTIVES[name]
            else:
                print(f"WARNING: Unknown perspective '{name}', skipping.")
                print(f"  Available: {', '.join(DEFAULT_PERSPECTIVES.keys())}")
        if not perspectives:
            print("ERROR: No valid perspectives specified.")
            sys.exit(1)
    else:
        perspectives = dict(DEFAULT_PERSPECTIVES)

    # Build judge list: (perspective_name, model_id, temperature)
    judges = []
    for perspective_name in sorted(perspectives.keys()):
        for model_id, temp in models:
            judges.append((perspective_name, model_id, temp))

    n_versions = len(labels)
    n_individual = len(judges) * n_versions
    n_comparison = len(judges)
    n_total_calls = n_individual + n_comparison

    # --- Output setup ---
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "reviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ab_comparison_{date_str}.jsonl"

    # --- Print configuration ---
    print("=" * 70)
    print("A/B COMPARISON CONFIGURATION")
    print("=" * 70)
    print(f"\nVersions ({n_versions}):")
    for label in labels:
        filepath = blind_mapping["label_to_file"][label]
        text = version_texts[filepath]
        sha = sha256_text(text)[:12]
        print(f"  {label}: {Path(filepath).name} ({len(text)} chars, SHA256: {sha}...)")

    if transition_context:
        print(f"\nTransition context: {args.transition_context} (first 30 lines)")
    else:
        print(f"\nTransition context: none")

    print(f"\nJudge panel ({len(judges)} judges):")
    for perspective_name, model_id, temp in judges:
        print(f"  {perspective_name:>20s} x {model_id:<40s} temp={temp:.1f}")

    print(f"\nEvaluation plan:")
    print(f"  Individual evaluations: {n_individual} ({len(judges)} judges x {n_versions} versions)")
    print(f"  Comparison evaluations: {n_comparison} ({len(judges)} judges)")
    print(f"  Total API calls: {n_total_calls}")
    print(f"\nOutput: {output_file}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without making API calls.")

        # In dry run, print what the first prompt would look like
        first_label = labels[0]
        first_file = blind_mapping["label_to_file"][first_label]
        sample_prompt = build_individual_eval_prompt(
            first_label,
            version_texts[first_file][:500] + "\n[... truncated for dry run ...]",
            transition_context[:200] + "\n[... truncated ...]" if transition_context else None,
        )
        print(f"\n--- SAMPLE PROMPT (first {len(sample_prompt)} chars) ---")
        print(sample_prompt[:2000])
        if len(sample_prompt) > 2000:
            print("[... truncated ...]")
        print("--- END SAMPLE PROMPT ---")
        return

    # --- Write provenance record ---
    provenance = {
        "record_type": "provenance",
        "timestamp": now.isoformat(),
        "run_id": date_str,
        "blind_mapping": blind_mapping,
        "version_hashes": {
            label: sha256_text(version_texts[blind_mapping["label_to_file"][label]])
            for label in labels
        },
        "version_char_counts": {
            label: len(version_texts[blind_mapping["label_to_file"][label]])
            for label in labels
        },
        "transition_context_file": args.transition_context,
        "transition_context_lines": len(transition_context.splitlines()) if transition_context else 0,
        "judges": [
            {"perspective": p, "model": m, "temperature": t}
            for p, m, t in judges
        ],
        "n_individual_evals": n_individual,
        "n_comparison_evals": n_comparison,
    }
    append_jsonl(output_file, provenance)

    # --- Phase 1: Individual blind evaluations ---
    print(f"\n{'='*70}")
    print("PHASE 1: Individual Blind Evaluations")
    print(f"{'='*70}")

    all_records = [provenance]
    eval_count = 0

    for perspective_name, model_id, temp in judges:
        for label in labels:
            eval_count += 1
            filepath = blind_mapping["label_to_file"][label]
            section_text = version_texts[filepath]

            print(f"\n[{eval_count}/{n_individual}] "
                  f"{perspective_name} x {model_id.split('/')[-1]} "
                  f"(temp={temp:.1f}) -> {label}")

            user_prompt = build_individual_eval_prompt(
                label, section_text, transition_context
            )
            system_prompt = perspectives[perspective_name]

            result = call_with_retry(
                model=model_id,
                temperature=temp,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=api_key,
            )

            parsed_scores = parse_individual_scores(result["response_text"])

            record = {
                "record_type": "individual_eval",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": date_str,
                "eval_index": eval_count - 1,
                "phase": "individual",
                "blind_label": label,
                "version_file": filepath,
                "perspective": perspective_name,
                "model_id": model_id,
                "model_id_returned": result.get("model_id_returned", model_id),
                "temperature": temp,
                "system_prompt": system_prompt,
                "response_text": result["response_text"],
                "parsed_scores": parsed_scores,
                "success": result["success"],
                "error": result["error"],
                "latency_ms": result["latency_ms"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "finish_reason": result.get("finish_reason", ""),
            }
            all_records.append(record)
            append_jsonl(output_file, record)

            if result["success"]:
                scores_str = ", ".join(
                    f"{k}={v}" for k, v in parsed_scores.items()
                    if v is not None and isinstance(v, int)
                )
                print(f"  OK: {scores_str}")
                print(f"  Tokens: {result['prompt_tokens']}+{result['completion_tokens']}, "
                      f"Latency: {result['latency_ms']}ms")
            else:
                print(f"  ERROR: {result['error']}")

            # Polite delay between API calls
            time.sleep(2)

    # --- Phase 2: Comparative ranking ---
    print(f"\n{'='*70}")
    print("PHASE 2: Comparative Rankings")
    print(f"{'='*70}")

    # Build version_texts dict keyed by blind label
    labeled_texts = {}
    for label in labels:
        filepath = blind_mapping["label_to_file"][label]
        labeled_texts[label] = version_texts[filepath]

    comparison_count = 0
    for perspective_name, model_id, temp in judges:
        comparison_count += 1
        print(f"\n[{comparison_count}/{n_comparison}] "
              f"{perspective_name} x {model_id.split('/')[-1]} "
              f"(temp={temp:.1f}) -> comparing all versions")

        user_prompt = build_comparison_prompt(labeled_texts, transition_context)
        system_prompt = perspectives[perspective_name]

        result = call_with_retry(
            model=model_id,
            temperature=temp,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
        )

        parsed_ranking = parse_comparison_ranking(result["response_text"], labels)

        record = {
            "record_type": "comparison",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": date_str,
            "comparison_index": comparison_count - 1,
            "phase": "comparison",
            "perspective": perspective_name,
            "model_id": model_id,
            "model_id_returned": result.get("model_id_returned", model_id),
            "temperature": temp,
            "system_prompt": system_prompt,
            "response_text": result["response_text"],
            "parsed_ranking": parsed_ranking,
            "success": result["success"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "finish_reason": result.get("finish_reason", ""),
        }
        all_records.append(record)
        append_jsonl(output_file, record)

        if result["success"]:
            ranking_str = " > ".join(parsed_ranking.get("ranking", ["?"]))
            rec = parsed_ranking.get("recommendation", "?")
            print(f"  OK: Ranking: {ranking_str}")
            print(f"  Recommendation: {rec}")
            print(f"  Tokens: {result['prompt_tokens']}+{result['completion_tokens']}, "
                  f"Latency: {result['latency_ms']}ms")
        else:
            print(f"  ERROR: {result['error']}")

        # Polite delay
        if comparison_count < n_comparison:
            time.sleep(2)

    # --- Write summary record ---
    summary = {
        "record_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": date_str,
        "blind_mapping": blind_mapping,
        "total_api_calls": n_total_calls,
        "successful_calls": sum(
            1 for r in all_records
            if r.get("record_type") in ("individual_eval", "comparison")
            and r.get("success")
        ),
        "failed_calls": sum(
            1 for r in all_records
            if r.get("record_type") in ("individual_eval", "comparison")
            and not r.get("success")
        ),
        "total_tokens": sum(
            r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
            for r in all_records
            if r.get("record_type") in ("individual_eval", "comparison")
        ),
        "total_latency_ms": sum(
            r.get("latency_ms", 0)
            for r in all_records
            if r.get("record_type") in ("individual_eval", "comparison")
        ),
    }

    # Compute per-version mean scores
    criteria = ["clarity", "motivation", "venue_fit", "narrative_flow"]
    version_stats = {}
    for label in labels:
        version_stats[label] = {}
        for criterion in criteria:
            scores = [
                r["parsed_scores"][criterion]
                for r in all_records
                if r.get("record_type") == "individual_eval"
                and r["blind_label"] == label
                and r["parsed_scores"].get(criterion) is not None
            ]
            if scores:
                version_stats[label][criterion] = {
                    "mean": round(statistics.mean(scores), 2),
                    "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "n": len(scores),
                }
            else:
                version_stats[label][criterion] = None

        # Composite
        all_scores = []
        for criterion in criteria:
            c_scores = [
                r["parsed_scores"][criterion]
                for r in all_records
                if r.get("record_type") == "individual_eval"
                and r["blind_label"] == label
                and r["parsed_scores"].get(criterion) is not None
            ]
            all_scores.extend(c_scores)
        if all_scores:
            version_stats[label]["composite"] = {
                "mean": round(statistics.mean(all_scores), 2),
                "stdev": round(statistics.stdev(all_scores), 2) if len(all_scores) > 1 else 0.0,
                "n": len(all_scores),
            }

    summary["version_stats"] = version_stats

    # Ranking aggregation
    first_place_counts = {label: 0 for label in labels}
    recommendation_counts = {label: 0 for label in labels}
    for r in all_records:
        if r.get("record_type") == "comparison":
            ranking = r["parsed_ranking"].get("ranking", [])
            if ranking:
                first = ranking[0]
                if first in first_place_counts:
                    first_place_counts[first] += 1
            rec = r["parsed_ranking"].get("recommendation")
            if rec and rec in recommendation_counts:
                recommendation_counts[rec] += 1

    summary["first_place_counts"] = first_place_counts
    summary["recommendation_counts"] = recommendation_counts

    append_jsonl(output_file, summary)
    all_records.append(summary)

    # --- Print summary ---
    print_summary(all_records, blind_mapping, labels)
    print(f"\nResults written to: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "A/B comparison of paper section versions using LLM judges on "
            "OpenRouter. Performs blind evaluation with controlled temperatures."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Compare two intro versions with transition context\n"
            "  python scripts/ab_comparison.py \\\n"
            "    --versions papers/sosp/intro.tex papers/sosp/intro_restructured.tex \\\n"
            "    --transition-context papers/sosp/background.tex\n"
            "\n"
            "  # Dry run to preview configuration\n"
            "  python scripts/ab_comparison.py \\\n"
            "    --versions papers/sosp/intro.tex papers/sosp/intro_restructured.tex \\\n"
            "    --dry-run\n"
            "\n"
            "  # Custom models and perspectives\n"
            "  python scripts/ab_comparison.py \\\n"
            "    --versions papers/sosp/intro.tex papers/sosp/intro_restructured.tex \\\n"
            "    --models google/gemini-2.5-pro-preview deepseek/deepseek-chat-v3-0324 \\\n"
            "    --perspectives systems_reviewer narrative_reviewer\n"
        ),
    )
    parser.add_argument(
        "--versions", nargs="+", required=True,
        help="2-4 .tex files to compare (paths relative to project root or absolute)",
    )
    parser.add_argument(
        "--transition-context", type=str, default=None,
        help="Path to the next section file (first 30 lines used for transition eval)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: reviews/)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Override default model list. Provide model IDs as on OpenRouter. "
            "Temperatures will be distributed evenly from 0.3 to 1.0."
        ),
    )
    parser.add_argument(
        "--perspectives", nargs="+", default=None,
        help=(
            "Override default perspectives. Choose from: "
            f"{', '.join(DEFAULT_PERSPECTIVES.keys())}"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and sample prompt without making API calls",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for blind label assignment reproducibility",
    )
    args = parser.parse_args()

    # Validate version count
    if len(args.versions) < 2:
        parser.error("At least 2 versions are required.")
    if len(args.versions) > 4:
        parser.error("At most 4 versions are supported.")

    if args.seed is not None:
        random.seed(args.seed)

    run_comparison(args)


if __name__ == "__main__":
    main()
