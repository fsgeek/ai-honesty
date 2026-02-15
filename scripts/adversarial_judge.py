#!/usr/bin/env python3
"""Adversarial reviewer judge for academic papers.

Simulates three hostile-but-competent reviewers who try to find reasons to
reject the paper. Each reviewer persona attacks from a different angle:

  Reviewer A (Methods Skeptic): Challenges experimental methodology, confounds,
    overfitting to evaluation, circular reasoning, and claims exceeding evidence.

  Reviewer B (Theory Skeptic): Questions whether formal results actually connect
    to empirical claims, looks for gaps between theorems and conclusions,
    hand-waving in proofs, unjustified assumptions, and alternative explanations.

  Reviewer C (So What? Reviewer): Challenges significance, novelty, and practical
    relevance. Compares to existing work and asks what this adds. Questions
    whether the evaluation scale is meaningful.

Each persona makes a separate API call, generating 5-10 specific, actionable
attacks with severity ratings (FATAL, MAJOR, MINOR).

Architecture:
  - Reuses strip_latex_to_text from redundancy_judge.py
  - Loads all .tex files from a paper directory (same pattern as provenance_judge.py)
  - OpenRouter API with retry logic (project standard)
  - JSONL output with SHA256 provenance records
  - Three passes (one per reviewer persona), run in parallel

Usage:
    # All three reviewers
    python scripts/adversarial_judge.py --paper-dir papers/sosp/

    # Single reviewer persona
    python scripts/adversarial_judge.py --paper-dir papers/sosp/ --reviewer methods
    python scripts/adversarial_judge.py --paper-dir papers/sosp/ --reviewer theory
    python scripts/adversarial_judge.py --paper-dir papers/sosp/ --reviewer significance

    # Dry run to see what would be sent
    python scripts/adversarial_judge.py --paper-dir papers/sosp/ --dry-run

    # Use a different model
    python scripts/adversarial_judge.py --paper-dir papers/sosp/ \\
      --model deepseek/deepseek-chat-v3-0324

Environment:
    OPENROUTER_API_KEY: Required. Set in environment or in .env file
    in the project root.

Output:
    reviews/
      adversarial_YYYYMMDD_HHMMSS.jsonl   # All records (provenance + findings)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Import strip_latex_to_text from the redundancy judge
sys.path.insert(0, str(Path(__file__).parent))
from redundancy_judge import strip_latex_to_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemini-2.5-pro-preview"
DEFAULT_TEMPERATURE = 0.4  # Slightly higher for creative adversarial thinking
MAX_RETRIES = 1
MAX_PARALLEL_CALLS = 3

# Reviewer persona keys and their human-readable names
REVIEWER_KEYS = {
    "methods": "methods_skeptic",
    "theory": "theory_skeptic",
    "significance": "so_what",
}

REVIEWER_NAMES = {
    "methods_skeptic": "Reviewer A: Methods Skeptic",
    "theory_skeptic": "Reviewer B: Theory Skeptic",
    "so_what": "Reviewer C: The 'So What?' Reviewer",
}

ALL_REVIEWERS = list(REVIEWER_KEYS.values())


# ---------------------------------------------------------------------------
# LaTeX file loading (same pattern as provenance_judge.py)
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tex_files(paper_dir: Path) -> list[dict]:
    """Load all .tex files from the paper directory.

    Returns a list of dicts with keys: filepath, filename, raw, plain, sha256,
    line_count, char_count. Only includes files that are part of the actual
    paper (heuristic: has a \\section or is a known paper file).
    """
    # Known paper section files (in presentation order)
    preferred_order = [
        "intro.tex", "background.tex", "formal_proof.tex", "design.tex",
        "eval.tex", "discussion.tex", "related.tex", "conclusion.tex",
        "abstract.tex",
    ]

    tex_files = sorted(paper_dir.glob("*.tex"))
    if not tex_files:
        return []

    sections = []
    for tf in tex_files:
        raw = tf.read_text()
        # Skip hidden/internal files
        if tf.name.startswith("_") or tf.name.startswith("."):
            continue
        # Include if it has a section heading or is a known paper file
        has_section = bool(re.search(r'\\(?:section|subsection|begin\{abstract\})', raw))
        is_known = tf.name in preferred_order
        # Also include the main .tex file (contains \documentclass)
        is_main = bool(re.search(r'\\documentclass', raw))

        # Skip alternate drafts (intro_paxos.tex, intro_composed.tex, etc.)
        if tf.name.startswith("intro_") or tf.name.startswith("cut_"):
            continue

        if has_section or is_known or is_main:
            plain = strip_latex_to_text(raw)
            sections.append({
                "filepath": str(tf),
                "filename": tf.name,
                "raw": raw,
                "plain": plain,
                "sha256": sha256_of_file(tf),
                "line_count": len(plain.splitlines()),
                "char_count": len(plain),
            })

    # Sort by preferred order, then alphabetically
    order_map = {name: i for i, name in enumerate(preferred_order)}

    def sort_key(s):
        return (order_map.get(s["filename"], 999), s["filename"])

    sections.sort(key=sort_key)
    return sections


def build_full_paper_text(sections: list[dict]) -> str:
    """Combine all sections into a single plain-text document for the LLM."""
    separator = "=" * 72
    parts = []
    for i, section in enumerate(sections):
        parts.append(f"\n{separator}")
        parts.append(f"SECTION {i + 1}: {section['filename']}")
        parts.append(f"{separator}\n")
        parts.append(section["plain"])
        parts.append("")

    parts.append(f"\n{separator}")
    parts.append("END OF PAPER")
    parts.append(f"{separator}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenRouter API (matches project patterns)
# ---------------------------------------------------------------------------

def call_openrouter(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int = 8192,
) -> dict:
    """Call OpenRouter API and return full response dict."""
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fsgeek/ai-honesty",
        "X-Title": "AI Honesty Adversarial Judge",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start_ms = time.monotonic_ns() // 1_000_000
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=600)
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
    max_tokens: int = 8192,
) -> dict:
    """Call OpenRouter with retry on failure or empty response."""
    result = call_openrouter(
        model, temperature, system_prompt, user_prompt, api_key, max_tokens
    )

    retries = 0
    while retries < MAX_RETRIES and (
        not result["success"] or len(result.get("response_text", "")) == 0
    ):
        retries += 1
        reason = (
            "empty response" if result["success"]
            else result.get("error", "unknown")
        )
        print(f"    RETRY {retries}/{MAX_RETRIES}: {reason}...")
        time.sleep(3)
        result = call_openrouter(
            model, temperature, system_prompt, user_prompt, api_key, max_tokens
        )

    return result


def load_env_file(project_root: Path):
    """Load .env file if it exists (same logic as other pipeline scripts)."""
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# Reviewer persona prompts
# ---------------------------------------------------------------------------

PERSONA_SYSTEM_PROMPTS = {
    "methods_skeptic": """\
You are Reviewer A: The Methods Skeptic.

Your philosophy: "I trust no experimental methodology unless it's airtight. \
I look for confounds, undisclosed variables, overfitting to the evaluation, \
circular reasoning, and claims that exceed what the evidence supports."

You are reviewing a paper submitted to a top-tier systems conference (SOSP). \
You are hostile but competent: you want to find every legitimate reason to \
reject this paper. You do NOT fabricate problems or nitpick formatting. Every \
attack you make must be a real methodological concern that a knowledgeable \
reviewer would raise.

Your areas of focus:
- CONFOUNDS: Are there uncontrolled variables that could explain the results?
- EVALUATION OVERFITTING: Is the evaluation designed in a way that guarantees \
the claimed result? Does the metric actually measure what the paper claims?
- CIRCULAR REASONING: Does the paper assume what it's trying to prove?
- CLAIMS EXCEEDING EVIDENCE: Where does the paper claim more than the data \
supports? Look for absolute language ("proves", "demonstrates") where the \
evidence only "suggests" or "is consistent with".
- STATISTICAL VALIDITY: Are sample sizes adequate? Are comparisons fair? \
Are baselines appropriate? Are error bars reported?
- REPRODUCIBILITY: Could someone reproduce these experiments from the paper \
alone? What information is missing?

For each attack, you MUST:
1. Quote the EXACT text from the paper that you are challenging
2. Explain the specific methodological weakness
3. Suggest what the authors could do to address it

Rate each attack:
- FATAL: This issue, if not addressed, is sufficient reason to reject the paper. \
The claimed contribution is invalidated or unsubstantiated.
- MAJOR: This requires substantial revision. The paper's contribution survives \
but the current presentation/evidence is inadequate.
- MINOR: This should be acknowledged or fixed with a small change. It does not \
undermine the core contribution.

You MUST output a JSON array with 5-10 attack objects. Each object has:
  - "finding_id": string, format "ADV-A-NNN" (e.g., "ADV-A-001")
  - "reviewer": "methods_skeptic"
  - "severity": "FATAL" | "MAJOR" | "MINOR"
  - "paper_claim": string - exact text from the paper being attacked (quote it)
  - "paper_location": string - section name or description of where in the paper
  - "attack": string - the specific argument a reviewer would make (2-5 sentences)
  - "suggested_defense": string - what the authors could do to address this (1-3 sentences)

Output format:
```json
[
  { ... },
  { ... }
]
```

After the JSON array, provide a 2-3 sentence overall assessment: If you were \
writing the "Reasons for Rejection" section of your review, what would it say?""",

    "theory_skeptic": """\
You are Reviewer B: The Theory Skeptic.

Your philosophy: "I question whether the formal results actually connect to the \
empirical claims. I look for gaps between what the theorems prove and what the \
paper claims they mean, hand-waving in proofs, unjustified assumptions, and \
alternative explanations the authors haven't considered."

You are reviewing a paper submitted to a top-tier systems conference (SOSP). \
You are hostile but competent: you want to find every legitimate gap between \
the paper's formal apparatus and its real-world claims. You do NOT fabricate \
problems. Every attack must be a real concern that a theoretically sophisticated \
reviewer would raise.

Your areas of focus:
- THEOREM-CLAIM GAP: Does the paper prove X but then claim Y? Do the formal \
results actually entail the conclusions drawn from them?
- ASSUMPTION SMUGGLING: What assumptions do the theorems require that the \
paper does not explicitly state? Are these assumptions realistic?
- PROOF HAND-WAVING: Where does the formal argument skip steps, appeal to \
intuition, or leave gaps that a careful reader would notice?
- ALTERNATIVE EXPLANATIONS: Are there simpler explanations for the empirical \
results that don't require the paper's theoretical framework?
- ANALOGY ABUSE: If the paper draws analogies to established results (e.g., \
FLP impossibility), does the analogy actually hold? What are the disanalogies?
- FORMALISM FIT: Is the formalism appropriate for the domain? Does it capture \
the relevant aspects of the problem, or does it formalize something adjacent \
to the actual question?

For each attack, you MUST:
1. Quote the EXACT text from the paper that you are challenging
2. Explain the specific theoretical weakness or gap
3. Suggest what the authors could do to address it

Rate each attack:
- FATAL: The theoretical contribution is invalid or vacuous. The theorems do \
not support the conclusions.
- MAJOR: There is a significant gap that requires substantial new argument or \
qualification.
- MINOR: A small clarification or qualification would address this.

You MUST output a JSON array with 5-10 attack objects. Each object has:
  - "finding_id": string, format "ADV-B-NNN" (e.g., "ADV-B-001")
  - "reviewer": "theory_skeptic"
  - "severity": "FATAL" | "MAJOR" | "MINOR"
  - "paper_claim": string - exact text from the paper being attacked (quote it)
  - "paper_location": string - section name or description of where in the paper
  - "attack": string - the specific argument a reviewer would make (2-5 sentences)
  - "suggested_defense": string - what the authors could do to address this (1-3 sentences)

Output format:
```json
[
  { ... },
  { ... }
]
```

After the JSON array, provide a 2-3 sentence overall assessment: If you were \
writing the "Reasons for Rejection" section of your review, what would it say?""",

    "so_what": """\
You are Reviewer C: The "So What?" Reviewer.

Your philosophy: "I ask whether this matters. I challenge significance, novelty, \
and practical relevance. I compare to existing work and ask what this adds. I \
challenge whether the evaluation scale is meaningful."

You are reviewing a paper submitted to a top-tier systems conference (SOSP). \
You are hostile but competent: you want to determine whether this paper makes \
a contribution that justifies publication at a top venue. You do NOT dismiss \
work unfairly. Every attack must be a real concern about impact, novelty, or \
relevance that a senior reviewer would raise.

Your areas of focus:
- NOVELTY: What exactly is new here? Is the core insight something the community \
already knows? Has this (or something very similar) been published before?
- SIGNIFICANCE: Even if the results are correct, do they matter? Who would \
change their behavior based on these findings?
- SCALE OF EVALUATION: Is the evaluation large enough to support the generality \
of the claims? If the paper tests 4 models, can it claim architectural universality?
- PRACTICAL RELEVANCE: Does this lead to actionable improvements, or is it \
purely diagnostic? Can practitioners USE these results?
- VENUE FIT: Is this a systems paper? Does it belong at SOSP, or would it be \
better suited to a different venue (ML conference, philosophy workshop, etc.)?
- COMPARISON TO EXISTING WORK: How does this compare to the large body of \
work on LLM hallucination, uncertainty quantification, and calibration? What \
does this add that CalibratedMath, Kadavath et al., or Lin et al. do not already provide?
- FRAMING: Is the paper overselling its results? Is the narrative honest about \
limitations?

For each attack, you MUST:
1. Quote the EXACT text from the paper that you are challenging
2. Explain why this claim is insufficiently significant, novel, or practical
3. Suggest what the authors could do to address it

Rate each attack:
- FATAL: The paper fails to demonstrate sufficient contribution for the venue. \
The core claim is either known, trivial, or unsupported.
- MAJOR: The contribution exists but is oversold, undertested, or poorly \
contextualized against prior work.
- MINOR: A reframing or additional paragraph would address this.

You MUST output a JSON array with 5-10 attack objects. Each object has:
  - "finding_id": string, format "ADV-C-NNN" (e.g., "ADV-C-001")
  - "reviewer": "so_what"
  - "severity": "FATAL" | "MAJOR" | "MINOR"
  - "paper_claim": string - exact text from the paper being attacked (quote it)
  - "paper_location": string - section name or description of where in the paper
  - "attack": string - the specific argument a reviewer would make (2-5 sentences)
  - "suggested_defense": string - what the authors could do to address this (1-3 sentences)

Output format:
```json
[
  { ... },
  { ... }
]
```

After the JSON array, provide a 2-3 sentence overall assessment: If you were \
writing the "Reasons for Rejection" section of your review, what would it say?""",
}


def build_reviewer_prompt(paper_text: str) -> str:
    """Build the user prompt containing the full stripped paper."""
    return (
        "Below is an academic paper submitted to SOSP (a top-tier systems "
        "conference), stripped of LaTeX formatting for readability. Each section "
        "is delimited by a header line.\n\n"
        "Read the paper carefully. Then generate your attacks.\n\n"
        f"{paper_text}"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_adversarial_response(response_text: str, reviewer_key: str) -> dict:
    """Parse the LLM response into structured adversarial findings.

    Returns:
        {
            "findings": [...],       # list of finding dicts
            "assessment_text": str,  # the overall assessment
            "parse_success": bool,
        }
    """
    result = {
        "findings": [],
        "assessment_text": "",
        "parse_success": False,
    }

    if not response_text:
        return result

    # Try to extract JSON array from the response
    json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find a bare JSON array
        json_match = re.search(r'(\[\s*\{.*?\}\s*\])', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            result["assessment_text"] = response_text
            return result

    try:
        findings = json.loads(json_str)
        if isinstance(findings, list):
            valid_findings = []
            for i, f in enumerate(findings):
                if isinstance(f, dict) and ("attack" in f or "paper_claim" in f):
                    # Normalize severity
                    severity = str(f.get("severity", "MAJOR")).upper()
                    if severity not in ("FATAL", "MAJOR", "MINOR"):
                        severity = "MAJOR"
                    f["severity"] = severity

                    # Ensure reviewer field
                    f["reviewer"] = f.get("reviewer", reviewer_key)

                    # Ensure finding_id
                    prefix_map = {
                        "methods_skeptic": "ADV-A",
                        "theory_skeptic": "ADV-B",
                        "so_what": "ADV-C",
                    }
                    prefix = prefix_map.get(reviewer_key, "ADV-X")
                    if "finding_id" not in f or not f["finding_id"]:
                        f["finding_id"] = f"{prefix}-{i + 1:03d}"

                    # Ensure required string fields
                    f.setdefault("paper_claim", "")
                    f.setdefault("paper_location", "")
                    f.setdefault("attack", "")
                    f.setdefault("suggested_defense", "")

                    valid_findings.append(f)

            result["findings"] = valid_findings
            result["parse_success"] = True
    except json.JSONDecodeError:
        pass

    # Extract assessment text (everything after the JSON block)
    # Look for text after the closing ``` of the JSON block
    assessment_match = re.search(
        r'```\s*\n(.*?)$',
        response_text[json_match.end():] if json_match else response_text,
        re.DOTALL,
    )
    if assessment_match:
        result["assessment_text"] = assessment_match.group(1).strip()
    else:
        # Try text after the JSON array
        after_json = response_text[json_match.end():].strip() if json_match else ""
        if after_json:
            result["assessment_text"] = after_json
        elif not result["parse_success"]:
            result["assessment_text"] = response_text

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

def print_findings_summary(all_findings: list[dict], reviewer_results: list[dict]):
    """Print a human-readable summary of adversarial findings."""
    print(f"\n{'=' * 72}")
    print("ADVERSARIAL REVIEW SUMMARY")
    print(f"{'=' * 72}")

    if not all_findings:
        print("\nNo attacks generated.")
        return

    # Count by severity
    severity_counts = {"FATAL": 0, "MAJOR": 0, "MINOR": 0}
    for f in all_findings:
        severity = f.get("severity", "MAJOR")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    print(f"\nTotal attacks: {len(all_findings)}")
    print(f"  FATAL: {severity_counts.get('FATAL', 0)}")
    print(f"  MAJOR: {severity_counts.get('MAJOR', 0)}")
    print(f"  MINOR: {severity_counts.get('MINOR', 0)}")

    # Per-reviewer breakdown
    for rr in reviewer_results:
        reviewer_key = rr.get("reviewer_key", "?")
        reviewer_name = REVIEWER_NAMES.get(reviewer_key, reviewer_key)
        findings = rr.get("findings", [])
        n_fatal = sum(1 for f in findings if f.get("severity") == "FATAL")
        n_major = sum(1 for f in findings if f.get("severity") == "MAJOR")
        n_minor = sum(1 for f in findings if f.get("severity") == "MINOR")

        print(f"\n  {reviewer_name}:")
        print(f"    Attacks: {len(findings)} "
              f"(F={n_fatal}, M={n_major}, m={n_minor})")

        assessment = rr.get("assessment_text", "")
        if assessment:
            # Truncate long assessments
            if len(assessment) > 300:
                assessment = assessment[:300] + "..."
            print(f"    Assessment: {assessment}")

    # FATAL findings detail
    fatal_findings = [f for f in all_findings if f.get("severity") == "FATAL"]
    if fatal_findings:
        print(f"\n{'=' * 72}")
        print("FATAL ATTACKS (detail)")
        print(f"{'=' * 72}")
        for i, f in enumerate(fatal_findings, 1):
            finding_id = f.get("finding_id", f"?-{i}")
            reviewer_key = f.get("reviewer", "?")
            reviewer_name = REVIEWER_NAMES.get(reviewer_key, reviewer_key)
            print(f"\n--- {finding_id} ({reviewer_name}) ---")

            claim = f.get("paper_claim", "(no claim quoted)")
            if len(claim) > 250:
                claim = claim[:250] + "..."
            print(f"  Claim:    \"{claim}\"")
            print(f"  Location: {f.get('paper_location', '?')}")

            attack = f.get("attack", "(no attack)")
            if len(attack) > 400:
                attack = attack[:400] + "..."
            print(f"  Attack:   {attack}")

            defense = f.get("suggested_defense", "(none)")
            if len(defense) > 200:
                defense = defense[:200] + "..."
            print(f"  Defense:  {defense}")

    # MAJOR findings (abbreviated)
    major_findings = [f for f in all_findings if f.get("severity") == "MAJOR"]
    if major_findings:
        print(f"\n{'=' * 72}")
        print("MAJOR ATTACKS (abbreviated)")
        print(f"{'=' * 72}")
        for f in major_findings:
            finding_id = f.get("finding_id", "?")
            location = f.get("paper_location", "?")
            attack = f.get("attack", "")
            # First sentence only
            first_sentence = attack.split(". ")[0] + "." if attack else "(no attack)"
            if len(first_sentence) > 150:
                first_sentence = first_sentence[:150] + "..."
            print(f"  {finding_id:12s} [{location}] {first_sentence}")

    # API usage
    total_tokens = sum(
        rr.get("prompt_tokens", 0) + rr.get("completion_tokens", 0)
        for rr in reviewer_results
    )
    total_latency = sum(rr.get("latency_ms", 0) for rr in reviewer_results)
    n_success = sum(1 for rr in reviewer_results if rr.get("success"))

    print(f"\nAPI Usage:")
    print(f"  Total calls: {len(reviewer_results)} ({n_success} successful)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total latency: {total_latency / 1000:.1f}s")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Per-reviewer processing
# ---------------------------------------------------------------------------

def process_reviewer(
    reviewer_key: str,
    paper_text: str,
    model_id: str,
    temperature: float,
    api_key: str,
) -> dict:
    """Run one reviewer persona against the paper. Designed for thread pool."""
    system_prompt = PERSONA_SYSTEM_PROMPTS[reviewer_key]
    user_prompt = build_reviewer_prompt(paper_text)

    result = call_with_retry(
        model=model_id,
        temperature=temperature,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key=api_key,
        max_tokens=8192,
    )

    parsed = parse_adversarial_response(result["response_text"], reviewer_key)

    return {
        "reviewer_key": reviewer_key,
        "reviewer_name": REVIEWER_NAMES.get(reviewer_key, reviewer_key),
        "model_id": model_id,
        "temperature": temperature,
        "success": result["success"],
        "error": result["error"],
        "latency_ms": result["latency_ms"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "finish_reason": result.get("finish_reason", ""),
        "model_id_returned": result.get("model_id_returned", model_id),
        "response_text": result["response_text"],
        "findings": parsed["findings"],
        "assessment_text": parsed["assessment_text"],
        "parse_success": parsed["parse_success"],
        "n_findings": len(parsed["findings"]),
        "n_fatal": sum(1 for f in parsed["findings"] if f.get("severity") == "FATAL"),
        "n_major": sum(1 for f in parsed["findings"] if f.get("severity") == "MAJOR"),
        "n_minor": sum(1 for f in parsed["findings"] if f.get("severity") == "MINOR"),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_adversarial_judge(args) -> Optional[Path]:
    """Execute the adversarial judge pipeline.

    Returns the output file path, or None if dry run.
    """
    project_root = Path(__file__).parent.parent
    load_env_file(project_root)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not found.")
        print("Set it in .env or as an environment variable.")
        print("Get a key at https://openrouter.ai/keys")
        sys.exit(1)

    # --- Load paper ---
    paper_dir = Path(args.paper_dir)
    if not paper_dir.is_absolute():
        paper_dir = project_root / paper_dir
    if not paper_dir.is_dir():
        print(f"ERROR: Paper directory not found: {paper_dir}")
        sys.exit(1)

    tex_sections = load_tex_files(paper_dir)
    if not tex_sections:
        print(f"ERROR: No .tex files found in {paper_dir}")
        sys.exit(1)

    paper_text = build_full_paper_text(tex_sections)

    # --- Determine which reviewers to run ---
    if args.reviewer:
        key = args.reviewer.lower()
        if key not in REVIEWER_KEYS:
            print(f"ERROR: Unknown reviewer '{args.reviewer}'.")
            print(f"  Valid options: {', '.join(REVIEWER_KEYS.keys())}")
            sys.exit(1)
        reviewers = [REVIEWER_KEYS[key]]
    else:
        reviewers = ALL_REVIEWERS

    # --- Configure model ---
    model_id = args.model if args.model else DEFAULT_MODEL
    temperature = getattr(args, "temperature", DEFAULT_TEMPERATURE)

    # --- Output setup ---
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else project_root / "reviews"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"adversarial_{date_str}.jsonl"

    # --- Print configuration ---
    print("=" * 72)
    print("ADVERSARIAL JUDGE CONFIGURATION")
    print("=" * 72)

    total_chars = sum(s["char_count"] for s in tex_sections)
    total_lines = sum(s["line_count"] for s in tex_sections)

    print(f"\nPaper directory: {paper_dir}")
    print(f"\nSections ({len(tex_sections)}):")
    for i, s in enumerate(tex_sections, 1):
        sha_short = s["sha256"][:12]
        print(f"  {i}. {s['filename']:30s} ({s['line_count']} lines, "
              f"{s['char_count']} chars, SHA256: {sha_short}...)")

    print(f"\nTotal: {total_lines} lines, {total_chars} chars of plain text")
    print(f"Paper text size: ~{len(paper_text)} chars")

    print(f"\nModel: {model_id} (temp={temperature})")

    print(f"\nReviewers ({len(reviewers)}):")
    for rk in reviewers:
        print(f"  {REVIEWER_NAMES.get(rk, rk)}")

    print(f"\nAPI calls planned: {len(reviewers)}")
    print(f"Output: {output_file}")

    if args.dry_run:
        print(f"\n{'=' * 72}")
        print("[DRY RUN] Showing what would be sent. No API calls made.")
        print(f"{'=' * 72}")

        # Show each reviewer's system prompt (truncated)
        for rk in reviewers:
            sp = PERSONA_SYSTEM_PROMPTS[rk]
            rname = REVIEWER_NAMES.get(rk, rk)
            print(f"\n--- SYSTEM PROMPT: {rname} ({len(sp)} chars) ---")
            print(sp[:1500])
            if len(sp) > 1500:
                print("[... truncated ...]")
            print(f"--- END ---")

        # Show user prompt (truncated)
        user_prompt = build_reviewer_prompt(paper_text)
        print(f"\n--- USER PROMPT ({len(user_prompt)} chars) ---")
        print(user_prompt[:3000])
        if len(user_prompt) > 3000:
            print(f"\n[... {len(user_prompt) - 3000} more chars ...]")
        print("--- END USER PROMPT ---")

        return None

    # --- Write provenance record ---
    provenance = {
        "record_type": "provenance",
        "timestamp": now.isoformat(),
        "run_id": date_str,
        "tool": "adversarial_judge",
        "paper_dir": str(paper_dir),
        "sections": [
            {
                "index": i,
                "filename": s["filename"],
                "filepath": s["filepath"],
                "line_count": s["line_count"],
                "char_count": s["char_count"],
                "sha256": s["sha256"],
            }
            for i, s in enumerate(tex_sections)
        ],
        "total_plain_chars": total_chars,
        "total_plain_lines": total_lines,
        "paper_text_chars": len(paper_text),
        "model": model_id,
        "temperature": temperature,
        "reviewers": reviewers,
        "n_api_calls": len(reviewers),
    }
    append_jsonl(output_file, provenance)

    # --- Run reviewer personas ---
    print(f"\n{'=' * 72}")
    print("RUNNING ADVERSARIAL REVIEW")
    print(f"{'=' * 72}")

    reviewer_results = []
    all_findings = []

    # Run reviewers in parallel (they're independent)
    futures_map = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as executor:
        for rk in reviewers:
            future = executor.submit(
                process_reviewer, rk, paper_text, model_id, temperature, api_key
            )
            futures_map[future] = rk

        for future in as_completed(futures_map):
            rk = futures_map[future]
            rname = REVIEWER_NAMES.get(rk, rk)

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "reviewer_key": rk,
                    "reviewer_name": rname,
                    "model_id": model_id,
                    "success": False,
                    "error": str(e),
                    "findings": [],
                    "assessment_text": "",
                    "parse_success": False,
                    "n_findings": 0,
                    "n_fatal": 0,
                    "n_major": 0,
                    "n_minor": 0,
                    "latency_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

            reviewer_results.append(result)

            # Write to JSONL incrementally
            record = {
                "record_type": "reviewer_analysis",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": date_str,
                **result,
            }
            append_jsonl(output_file, record)

            # Print progress
            if result.get("success"):
                n = result.get("n_findings", 0)
                n_fatal = result.get("n_fatal", 0)
                n_major = result.get("n_major", 0)
                n_minor = result.get("n_minor", 0)
                print(f"\n  {rname}:")
                print(f"    Attacks: {n} (F={n_fatal}, M={n_major}, m={n_minor})")
                ptok = result.get("prompt_tokens", 0)
                ctok = result.get("completion_tokens", 0)
                lat = result.get("latency_ms", 0)
                print(f"    Tokens: {ptok}+{ctok}, Latency: {lat}ms")
                if not result.get("parse_success"):
                    print(f"    WARNING: Could not parse JSON from response; "
                          f"raw text captured.")
            else:
                print(f"\n  {rname}: ERROR: {result.get('error', 'unknown')}")

            # Collect findings
            all_findings.extend(result.get("findings", []))

    # --- Write individual finding records for pipeline consumption ---
    for f in all_findings:
        finding_record = {
            "record_type": "finding",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": date_str,
            **f,
        }
        append_jsonl(output_file, finding_record)

    # --- Write summary record ---
    summary_record = {
        "record_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": date_str,
        "n_reviewers": len(reviewers),
        "n_successful": sum(1 for rr in reviewer_results if rr.get("success")),
        "total_findings": len(all_findings),
        "total_fatal": sum(1 for f in all_findings if f.get("severity") == "FATAL"),
        "total_major": sum(1 for f in all_findings if f.get("severity") == "MAJOR"),
        "total_minor": sum(1 for f in all_findings if f.get("severity") == "MINOR"),
        "total_tokens": sum(
            rr.get("prompt_tokens", 0) + rr.get("completion_tokens", 0)
            for rr in reviewer_results
        ),
        "total_latency_ms": sum(
            rr.get("latency_ms", 0) for rr in reviewer_results
        ),
    }
    append_jsonl(output_file, summary_record)

    # --- Print summary ---
    print_findings_summary(all_findings, reviewer_results)
    print(f"\nResults written to: {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial reviewer judge for academic papers. Simulates three "
            "hostile-but-competent reviewer personas who try to find reasons "
            "to reject the paper. Each persona attacks from a different angle: "
            "methodology, theory, and significance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # All three reviewers\n"
            "  python scripts/adversarial_judge.py --paper-dir papers/sosp/\n"
            "\n"
            "  # Single reviewer persona\n"
            "  python scripts/adversarial_judge.py "
            "--paper-dir papers/sosp/ --reviewer methods\n"
            "  python scripts/adversarial_judge.py "
            "--paper-dir papers/sosp/ --reviewer theory\n"
            "  python scripts/adversarial_judge.py "
            "--paper-dir papers/sosp/ --reviewer significance\n"
            "\n"
            "  # Dry run to preview what would be sent\n"
            "  python scripts/adversarial_judge.py "
            "--paper-dir papers/sosp/ --dry-run\n"
            "\n"
            "  # Use a different model\n"
            "  python scripts/adversarial_judge.py --paper-dir papers/sosp/ \\\n"
            "    --model deepseek/deepseek-chat-v3-0324\n"
        ),
    )
    parser.add_argument(
        "--paper-dir", type=str, required=True,
        help="Directory containing .tex files for the paper.",
    )
    parser.add_argument(
        "--reviewer", type=str, default=None,
        choices=["methods", "theory", "significance"],
        help=(
            "Run a single reviewer persona instead of all three. "
            "Options: methods (Reviewer A), theory (Reviewer B), "
            "significance (Reviewer C)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: reviews/)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            f"Override default model. Provide an OpenRouter model ID. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Temperature for LLM calls (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration and sample prompts without making API calls",
    )

    args = parser.parse_args()
    run_adversarial_judge(args)


if __name__ == "__main__":
    main()
