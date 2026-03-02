#!/usr/bin/env python3
"""Redundancy detection judge for academic papers.

Identifies semantic redundancy ACROSS sections of a LaTeX paper. Distinguishes
between anxious restatement (same claim/result/number repeated because the
author doesn't trust the reader followed along) and intentional reference
(brief forward/backward reference like "As shown in section 3").

Architecture:
  - Parse LaTeX files in order, strip commands to plain text, preserve
    section/subsection structure
  - For each section, send its claims to an LLM along with all other sections
  - LLM identifies specific passages that are semantically redundant
  - Output: structured JSONL report with severity, saveable lines, primary
    vs echo designation, and recommended action

Usage:
    # Analyze the full paper (files in order)
    python scripts/redundancy_judge.py \\
      --files papers/sosp/intro.tex papers/sosp/background.tex \\
             papers/sosp/formal_proof.tex papers/sosp/design.tex \\
             papers/sosp/eval.tex papers/sosp/discussion.tex \\
             papers/sosp/related.tex papers/sosp/conclusion.tex

    # Dry run to see what would be sent without API calls
    python scripts/redundancy_judge.py \\
      --files papers/sosp/intro.tex papers/sosp/background.tex \\
      --dry-run

    # Use multiple models for consensus
    python scripts/redundancy_judge.py \\
      --files papers/sosp/intro.tex papers/sosp/background.tex \\
      --models google/gemini-2.5-pro-preview deepseek/deepseek-chat-v3-0324

Environment:
    OPENROUTER_API_KEY: Required. Set in environment or in .env file
    in the project root.

Output:
    reviews/
      redundancy_YYYYMMDD_HHMMSS.jsonl   # All records (provenance + findings)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemini-2.5-pro-preview"
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 1


# ---------------------------------------------------------------------------
# LaTeX stripping
# ---------------------------------------------------------------------------

def strip_latex_to_text(latex: str) -> str:
    """Strip LaTeX commands to get plain text, preserving section structure.

    Keeps section/subsection headings as markdown-style headers for the LLM
    to orient itself. Removes commands, environments, comments, and math
    markup while preserving the readable text content.
    """
    text = latex

    # Remove LaTeX comments (lines starting with %)
    text = re.sub(r'(?m)^%.*$', '', text)
    # Remove inline comments (% not preceded by \)
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Convert section commands to markdown-style headers
    text = re.sub(r'\\section\*?\{([^}]*)\}', r'\n# \1\n', text)
    text = re.sub(r'\\subsection\*?\{([^}]*)\}', r'\n## \1\n', text)
    text = re.sub(r'\\subsubsection\*?\{([^}]*)\}', r'\n### \1\n', text)

    # Convert \fakepara to bold paragraph headers (project-specific command)
    text = re.sub(r'\\fakepara\{([^}]*)\}', r'\n**\1**', text)

    # Remove label commands
    text = re.sub(r'\\label\{[^}]*\}', '', text)

    # Remove \ref, \cite, \eqref but keep surrounding context
    text = re.sub(r'\\(?:ref|eqref)\{[^}]*\}', '[ref]', text)
    text = re.sub(r'\\cite[tp]?\{[^}]*\}', '[citation]', text)
    text = re.sub(r'~\\cite[tp]?\{[^}]*\}', ' [citation]', text)

    # Remove common formatting commands, keep content
    text = re.sub(r'\\(?:emph|textbf|textit|texttt|textsc)\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:small|large|Large|LARGE|footnotesize|scriptsize|tiny|normalsize)\b', '', text)
    text = re.sub(r'\{\\(?:bf|it|em|tt|sc|sl)\s+([^}]*)\}', r'\1', text)

    # Handle math environments: inline math ($...$) -> keep content roughly
    text = re.sub(r'\$([^$]+)\$', r' \1 ', text)
    # Display math (\[...\] and $$...$$)
    text = re.sub(r'\\\[.*?\\\]', ' [equation] ', text, flags=re.DOTALL)
    text = re.sub(r'\$\$.*?\$\$', ' [equation] ', text, flags=re.DOTALL)

    # Remove begin/end environments but keep content between them
    # Special case: remove figure/table environments entirely (they're not text)
    text = re.sub(
        r'\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}',
        '\n[figure/table omitted]\n', text, flags=re.DOTALL
    )
    # For other environments, just strip the begin/end markers
    text = re.sub(r'\\begin\{[^}]*\}(?:\[[^\]]*\])?', '', text)
    text = re.sub(r'\\end\{[^}]*\}', '', text)

    # Remove common standalone commands
    text = re.sub(r'\\(?:balance|newpage|clearpage|pagebreak|noindent|vspace|hspace|xspace)\b(?:\{[^}]*\})?', '', text)
    text = re.sub(r'\\(?:vfill|hfill|centering|raggedright|raggedleft)\b', '', text)

    # Remove \item markers, keep the content
    text = re.sub(r'\\item\s*(?:\[[^\]]*\])?\s*', '\n  - ', text)

    # Remove remaining simple commands (e.g., \eg, \ie, \cf)
    text = re.sub(r'\\(?:eg|ie|cf)\b', lambda m: m.group(0)[1:] + '.', text)

    # Remove leftover backslash commands that we haven't handled
    # Be conservative: only remove \command without braces (to avoid destroying content)
    text = re.sub(r'\\[a-zA-Z]+(?:\*)?(?=\s|$|[^{a-zA-Z])', '', text)

    # Remove remaining \command{...} patterns we missed, keeping the content
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)

    # Clean up braces
    text = text.replace('{', '').replace('}', '')

    # Handle LaTeX escaped special characters
    text = text.replace('\\&', '&')
    text = text.replace('\\%', '%')
    text = text.replace('\\$', '$')
    text = text.replace('\\#', '#')
    text = text.replace('\\_', '_')
    text = text.replace('\\textbackslash', '\\')
    # Em-dash and en-dash
    text = text.replace('---', '\u2014')
    text = text.replace('--', '\u2013')
    # Quotes
    text = text.replace('``', '\u201c')
    text = text.replace("''", '\u201d')

    # Clean up tildes (LaTeX non-breaking space)
    text = text.replace('~', ' ')

    # Clean up multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up leading/trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    text = '\n'.join(lines)

    return text.strip()


def extract_section_name(latex: str, filename: str) -> str:
    """Extract the section name from LaTeX source, falling back to filename."""
    match = re.search(r'\\section\*?\{([^}]*)\}', latex)
    if match:
        name = match.group(1)
        # Clean up LaTeX escapes in the name
        name = name.replace('\\&', '&')
        name = name.replace('\\%', '%')
        name = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', name)  # \emph{X} -> X
        name = re.sub(r'\\[a-zA-Z]+', '', name)  # remaining commands
        return name.strip()
    return Path(filename).stem.replace('_', ' ').title()


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

class Section:
    """A section of the paper with both raw and stripped text."""

    def __init__(self, filepath: str, raw_latex: str):
        self.filepath = filepath
        self.filename = Path(filepath).name
        self.raw_latex = raw_latex
        self.plain_text = strip_latex_to_text(raw_latex)
        self.name = extract_section_name(raw_latex, filepath)
        self.line_count = len(self.plain_text.splitlines())
        self.sha256 = hashlib.sha256(raw_latex.encode()).hexdigest()

    def __repr__(self):
        return f"Section({self.name!r}, {self.line_count} lines)"


def load_sections(file_paths: list[str]) -> list[Section]:
    """Load and parse all section files in order."""
    sections = []
    project_root = Path(__file__).parent.parent

    for path_str in file_paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        raw = path.read_text()
        sections.append(Section(str(path), raw))

    return sections


# ---------------------------------------------------------------------------
# OpenRouter API (matches project patterns from ab_comparison.py)
# ---------------------------------------------------------------------------

def call_openrouter(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int | None = None,
    app_label: str = "SOSP/redundancy",
) -> dict:
    """Call OpenRouter API and return full response dict."""
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fsgeek/ai-honesty",
        "X-Title": app_label,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

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
    max_tokens: int | None = None,
    app_label: str = "SOSP/redundancy",
) -> dict:
    """Call OpenRouter with retry on failure or empty response."""
    result = call_openrouter(model, temperature, system_prompt, user_prompt, api_key, max_tokens, app_label=app_label)

    retries = 0
    while retries < MAX_RETRIES and (
        not result["success"] or len(result.get("response_text", "")) == 0
    ):
        retries += 1
        reason = "empty response" if result["success"] else result.get("error", "unknown")
        print(f"    RETRY {retries}/{MAX_RETRIES}: {reason}...")
        time.sleep(3)
        result = call_openrouter(model, temperature, system_prompt, user_prompt, api_key, max_tokens, app_label=app_label)

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
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are reviewing an academic paper for semantic redundancy across sections.

Your task is to find places where the same claim, result, number, or argument
is restated in multiple sections. You must distinguish between two types of
repetition:

1. ANXIOUS RESTATEMENT (FLAG THIS): The same claim or result appears in full
   in multiple sections because the author does not trust the reader to
   remember it. The echo does not add new information, context, or nuance
   that the primary statement lacks. This wastes page budget.

2. INTENTIONAL REFERENCE (DO NOT FLAG): A brief forward or backward reference
   like "As shown in Section 3" or "recall that X holds". These are necessary
   connective tissue and should NOT be flagged. Similarly, an abstract or
   conclusion that briefly summarizes key results is conventional and not
   redundancy.

For each redundancy you find, identify:
  (1) The PRIMARY statement: where the idea first appears in full, with the
      exact quoted text
  (2) The ECHO: where the idea is restated, with the exact quoted text
  (3) Whether the echo adds anything the primary does not (new context,
      different framing for a different audience, additional detail)
  (4) Severity: HIGH (verbatim or near-verbatim restatement of multiple
      sentences), MEDIUM (same claim restated in different words), LOW
      (borderline — could be intentional reinforcement)
  (5) Recommended action: one of:
      - DELETE_ECHO: Remove the echo entirely; the primary is sufficient
      - MERGE_INTO_PRIMARY: The echo has useful additions; merge them into
        the primary location and delete the echo
      - SHORTEN_TO_REFERENCE: Replace the echo with a brief cross-reference
        ("As established in Section N, ...")
      - KEEP: This is actually intentional reference or conventional
        summarization; leave it alone

You MUST output your findings as a JSON array. Each element should be an object
with these fields:
  - "primary_section": section name where the idea first appears
  - "primary_quote": exact quoted text (30-150 words)
  - "echo_section": section name where the restatement appears
  - "echo_quote": exact quoted text (30-150 words)
  - "adds_anything": boolean — does the echo add information the primary lacks?
  - "what_it_adds": string or null — if adds_anything is true, what does it add?
  - "severity": "HIGH" | "MEDIUM" | "LOW"
  - "estimated_lines_saveable": integer — how many lines could be saved by
    the recommended action
  - "action": "DELETE_ECHO" | "MERGE_INTO_PRIMARY" | "SHORTEN_TO_REFERENCE" | "KEEP"
  - "explanation": 1-2 sentence justification for the severity and action

After the JSON array, provide a brief SUMMARY with:
  - Total redundancies found (by severity)
  - Estimated total lines saveable
  - The single worst offender (most egregious redundancy)
  - Any sections that are particularly clean (no redundancy with other sections)

Output format:
```json
[
  { ... },
  { ... }
]
```

SUMMARY:
..."""


def build_full_paper_prompt(sections: list[Section]) -> str:
    """Build the user prompt containing the full paper, section by section."""
    separator = "=" * 72
    parts = [
        "Below is an academic paper, presented section by section in order. "
        "Each section is delimited by a header line. The text has been stripped "
        "of LaTeX formatting for readability.\n\n"
        "Analyze the paper for semantic redundancy ACROSS sections. Do NOT "
        "flag redundancy within a single section — only across different sections.\n\n"
        "Pay special attention to:\n"
        "- Numbers or empirical results that appear in multiple sections\n"
        "- The same conceptual claim stated in both introduction and later sections\n"
        "- Definitions or explanations repeated in multiple places\n"
        "- The same analogy or metaphor used in more than one section\n\n"
        "Do NOT flag:\n"
        "- Brief cross-references (\"As shown in Section 3\")\n"
        "- The abstract summarizing results that appear in detail later\n"
        "- The conclusion summarizing the paper's contributions\n"
        "- A related work section comparing to ideas introduced earlier\n\n"
        "Here is the paper:",
        separator,
    ]

    for i, section in enumerate(sections):
        parts.append(f"\n{separator}")
        parts.append(f"SECTION {i + 1}: {section.name}")
        parts.append(f"(from: {section.filename})")
        parts.append(f"{separator}\n")
        parts.append(section.plain_text)
        parts.append("")

    parts.append(f"\n{separator}")
    parts.append("END OF PAPER")
    parts.append(f"{separator}\n")

    return "\n".join(parts)


def build_pairwise_prompt(focus_section: Section, other_sections: list[Section]) -> str:
    """Build a prompt comparing one section against all others.

    This is an alternative to the full-paper prompt for very long papers
    where context limits might be an issue.
    """
    parts = [
        f"You are analyzing the section \"{focus_section.name}\" for semantic "
        f"redundancy with other sections of the same paper.\n\n"
        f"FOCUS SECTION ({focus_section.name}, from {focus_section.filename}):\n"
        f"{'-' * 60}\n"
        f"{focus_section.plain_text}\n"
        f"{'-' * 60}\n\n"
        f"OTHER SECTIONS for comparison:\n"
    ]

    for other in other_sections:
        parts.append(f"\n{'=' * 60}")
        parts.append(f"SECTION: {other.name} (from {other.filename})")
        parts.append(f"{'=' * 60}\n")
        parts.append(other.plain_text)
        parts.append("")

    parts.append(
        f"\nIdentify any claims, results, or arguments in \"{focus_section.name}\" "
        f"that are semantically redundant with content in the other sections. "
        f"For each redundancy, determine whether the focus section contains "
        f"the PRIMARY statement or the ECHO.\n"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_redundancy_response(response_text: str) -> dict:
    """Parse the LLM response into structured redundancy findings.

    Returns:
        {
            "findings": [...],  # list of redundancy dicts
            "summary_text": str,  # the summary section
            "parse_success": bool,
        }
    """
    result = {
        "findings": [],
        "summary_text": "",
        "parse_success": False,
    }

    if not response_text:
        return result

    # Try to extract JSON array from the response
    # Look for ```json ... ``` blocks first
    json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find a bare JSON array
        json_match = re.search(r'(\[\s*\{.*?\}\s*\])', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # No JSON found — return raw text as summary
            result["summary_text"] = response_text
            return result

    try:
        findings = json.loads(json_str)
        if isinstance(findings, list):
            # Validate and normalize each finding
            valid_findings = []
            for f in findings:
                if isinstance(f, dict) and "primary_section" in f and "echo_section" in f:
                    # Normalize severity
                    severity = str(f.get("severity", "MEDIUM")).upper()
                    if severity not in ("HIGH", "MEDIUM", "LOW"):
                        severity = "MEDIUM"
                    f["severity"] = severity

                    # Normalize action
                    action = str(f.get("action", "SHORTEN_TO_REFERENCE")).upper()
                    valid_actions = {
                        "DELETE_ECHO", "MERGE_INTO_PRIMARY",
                        "SHORTEN_TO_REFERENCE", "KEEP"
                    }
                    if action not in valid_actions:
                        action = "SHORTEN_TO_REFERENCE"
                    f["action"] = action

                    # Ensure estimated_lines_saveable is int
                    try:
                        f["estimated_lines_saveable"] = int(f.get("estimated_lines_saveable", 0))
                    except (ValueError, TypeError):
                        f["estimated_lines_saveable"] = 0

                    # Ensure adds_anything is bool
                    f["adds_anything"] = bool(f.get("adds_anything", False))

                    valid_findings.append(f)

            result["findings"] = valid_findings
            result["parse_success"] = True
    except json.JSONDecodeError:
        # JSON parse failed — include raw text
        pass

    # Extract summary text (everything after the JSON block)
    summary_match = re.search(
        r'(?:```\s*\n|}\s*\]\s*\n)\s*(SUMMARY:.*)',
        response_text, re.DOTALL | re.IGNORECASE
    )
    if summary_match:
        result["summary_text"] = summary_match.group(1).strip()
    elif not result["parse_success"]:
        # If parsing failed, use the whole response as summary
        result["summary_text"] = response_text

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

def print_findings_summary(all_findings: list[dict], model_results: list[dict]):
    """Print a human-readable summary of redundancy findings."""
    print(f"\n{'=' * 72}")
    print("REDUNDANCY ANALYSIS SUMMARY")
    print(f"{'=' * 72}")

    if not all_findings:
        print("\nNo redundancies found across any model.")
        return

    # Count by severity
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    total_lines_saveable = 0
    action_counts = {}
    section_pair_counts = {}

    for f in all_findings:
        severity = f.get("severity", "MEDIUM")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        total_lines_saveable += f.get("estimated_lines_saveable", 0)

        action = f.get("action", "UNKNOWN")
        action_counts[action] = action_counts.get(action, 0) + 1

        pair = (f.get("primary_section", "?"), f.get("echo_section", "?"))
        section_pair_counts[pair] = section_pair_counts.get(pair, 0) + 1

    print(f"\nTotal redundancies found: {len(all_findings)}")
    print(f"  HIGH:   {severity_counts.get('HIGH', 0)}")
    print(f"  MEDIUM: {severity_counts.get('MEDIUM', 0)}")
    print(f"  LOW:    {severity_counts.get('LOW', 0)}")
    print(f"\nEstimated total lines saveable: {total_lines_saveable}")

    print(f"\nRecommended actions:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")

    # Most redundant section pairs
    if section_pair_counts:
        print(f"\nMost redundant section pairs:")
        sorted_pairs = sorted(section_pair_counts.items(), key=lambda x: -x[1])
        for (primary, echo), count in sorted_pairs[:5]:
            print(f"  {primary} -> {echo}: {count} redundancies")

    # High-severity findings detail
    high_findings = [f for f in all_findings if f.get("severity") == "HIGH"]
    if high_findings:
        print(f"\n{'=' * 72}")
        print("HIGH-SEVERITY REDUNDANCIES (detail)")
        print(f"{'=' * 72}")
        for i, f in enumerate(high_findings, 1):
            print(f"\n--- Redundancy {i} ---")
            print(f"Primary: [{f.get('primary_section', '?')}]")
            primary_quote = f.get('primary_quote', '(no quote)')
            if len(primary_quote) > 200:
                primary_quote = primary_quote[:200] + "..."
            print(f"  \"{primary_quote}\"")
            print(f"Echo:    [{f.get('echo_section', '?')}]")
            echo_quote = f.get('echo_quote', '(no quote)')
            if len(echo_quote) > 200:
                echo_quote = echo_quote[:200] + "..."
            print(f"  \"{echo_quote}\"")
            print(f"Action:  {f.get('action', '?')}")
            print(f"Lines saveable: {f.get('estimated_lines_saveable', '?')}")
            if f.get("explanation"):
                print(f"Reason:  {f['explanation']}")

    # Per-model breakdown
    if len(model_results) > 1:
        print(f"\n{'=' * 72}")
        print("PER-MODEL BREAKDOWN")
        print(f"{'=' * 72}")
        for mr in model_results:
            model = mr.get("model_id", "?")
            findings = mr.get("parsed", {}).get("findings", [])
            n_high = sum(1 for f in findings if f.get("severity") == "HIGH")
            n_med = sum(1 for f in findings if f.get("severity") == "MEDIUM")
            n_low = sum(1 for f in findings if f.get("severity") == "LOW")
            lines = sum(f.get("estimated_lines_saveable", 0) for f in findings)
            print(f"\n  {model}:")
            print(f"    Findings: {len(findings)} (H={n_high}, M={n_med}, L={n_low})")
            print(f"    Lines saveable: {lines}")

    # Consensus findings (if multiple models)
    if len(model_results) > 1:
        print(f"\n{'=' * 72}")
        print("CONSENSUS ANALYSIS")
        print(f"{'=' * 72}")
        # Rough consensus: group findings by (primary_section, echo_section, severity)
        # and count how many models flagged each
        from collections import Counter
        triplets = Counter()
        for mr in model_results:
            findings = mr.get("parsed", {}).get("findings", [])
            # Deduplicate within a single model by section pair
            seen_pairs = set()
            for f in findings:
                key = (f.get("primary_section", "?"), f.get("echo_section", "?"))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    triplets[key] += 1

        n_models = len(model_results)
        unanimous = [(k, v) for k, v in triplets.items() if v == n_models]
        majority = [(k, v) for k, v in triplets.items() if v > n_models / 2]

        print(f"\n  Models: {n_models}")
        print(f"  Section pairs flagged by ALL models: {len(unanimous)}")
        for (primary, echo), count in unanimous:
            print(f"    {primary} -> {echo}")
        print(f"  Section pairs flagged by majority: {len(majority)}")
        for (primary, echo), count in sorted(majority, key=lambda x: -x[1]):
            if count < n_models:  # Don't repeat unanimous ones
                print(f"    {primary} -> {echo} ({count}/{n_models} models)")

    # API usage
    total_tokens = sum(
        mr.get("prompt_tokens", 0) + mr.get("completion_tokens", 0)
        for mr in model_results
    )
    total_latency = sum(mr.get("latency_ms", 0) for mr in model_results)
    n_success = sum(1 for mr in model_results if mr.get("success"))

    print(f"\nAPI Usage:")
    print(f"  Total calls: {len(model_results)} ({n_success} successful)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total latency: {total_latency / 1000:.1f}s")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_redundancy_judge(args) -> Optional[Path]:
    """Execute the redundancy judge pipeline.

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

    # --- Load sections ---
    sections = load_sections(args.files)

    if not sections:
        print("ERROR: No sections loaded.")
        sys.exit(1)

    # --- Configure models ---
    temperature = getattr(args, 'temperature', DEFAULT_TEMPERATURE)
    if args.models:
        models = [(m, temperature) for m in args.models]
    else:
        models = [(DEFAULT_MODEL, temperature)]

    # --- Build the full paper prompt ---
    user_prompt = build_full_paper_prompt(sections)

    # --- Output setup ---
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "reviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"redundancy_{date_str}.jsonl"

    # --- Print configuration ---
    print("=" * 72)
    print("REDUNDANCY JUDGE CONFIGURATION")
    print("=" * 72)

    total_chars = sum(len(s.plain_text) for s in sections)
    total_lines = sum(s.line_count for s in sections)

    print(f"\nSections ({len(sections)}):")
    for i, s in enumerate(sections, 1):
        sha_short = s.sha256[:12]
        print(f"  {i}. {s.name:30s} ({s.filename}, {s.line_count} lines, "
              f"{len(s.plain_text)} chars, SHA256: {sha_short}...)")

    print(f"\nTotal: {total_lines} lines, {total_chars} chars of plain text")
    print(f"Prompt size: ~{len(user_prompt)} chars")

    print(f"\nModels ({len(models)}):")
    for model_id, temp in models:
        print(f"  {model_id} (temp={temp})")

    print(f"\nAPI calls planned: {len(models)}")
    print(f"Output: {output_file}")

    if args.dry_run:
        print(f"\n{'=' * 72}")
        print("[DRY RUN] Showing what would be sent. No API calls made.")
        print(f"{'=' * 72}")

        # Show system prompt
        print(f"\n--- SYSTEM PROMPT ({len(SYSTEM_PROMPT)} chars) ---")
        print(SYSTEM_PROMPT[:1500])
        if len(SYSTEM_PROMPT) > 1500:
            print("[... truncated ...]")
        print("--- END SYSTEM PROMPT ---")

        # Show user prompt (truncated)
        print(f"\n--- USER PROMPT ({len(user_prompt)} chars) ---")
        print(user_prompt[:3000])
        if len(user_prompt) > 3000:
            print(f"\n[... {len(user_prompt) - 3000} more chars ...]")
        print("--- END USER PROMPT ---")

        # Show a sample section's stripped text
        if sections:
            sample = sections[0]
            print(f"\n--- SAMPLE STRIPPED TEXT: {sample.name} ---")
            print(sample.plain_text[:1500])
            if len(sample.plain_text) > 1500:
                print(f"\n[... {len(sample.plain_text) - 1500} more chars ...]")
            print("--- END SAMPLE ---")

        return None

    # --- Write provenance record ---
    provenance = {
        "record_type": "provenance",
        "timestamp": now.isoformat(),
        "run_id": date_str,
        "tool": "redundancy_judge",
        "sections": [
            {
                "index": i,
                "name": s.name,
                "filename": s.filename,
                "filepath": s.filepath,
                "line_count": s.line_count,
                "char_count": len(s.plain_text),
                "sha256": s.sha256,
            }
            for i, s in enumerate(sections)
        ],
        "total_plain_chars": total_chars,
        "total_plain_lines": total_lines,
        "prompt_chars": len(user_prompt),
        "models": [{"model": m, "temperature": t} for m, t in models],
        "n_api_calls": len(models),
    }
    append_jsonl(output_file, provenance)

    # --- Call each model ---
    print(f"\n{'=' * 72}")
    print("RUNNING REDUNDANCY ANALYSIS")
    print(f"{'=' * 72}")

    model_results = []
    all_findings = []

    for idx, (model_id, temp) in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] {model_id} (temp={temp})...")

        result = call_with_retry(
            model=model_id,
            temperature=temp,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            api_key=api_key,
            app_label="SOSP/redundancy",
        )

        parsed = parse_redundancy_response(result["response_text"])

        record = {
            "record_type": "redundancy_analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": date_str,
            "model_index": idx - 1,
            "model_id": model_id,
            "model_id_returned": result.get("model_id_returned", model_id),
            "temperature": temp,
            "success": result["success"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "finish_reason": result.get("finish_reason", ""),
            "response_text": result["response_text"],
            "parsed": {
                "findings": parsed["findings"],
                "summary_text": parsed["summary_text"],
                "parse_success": parsed["parse_success"],
                "n_findings": len(parsed["findings"]),
                "n_high": sum(1 for f in parsed["findings"] if f.get("severity") == "HIGH"),
                "n_medium": sum(1 for f in parsed["findings"] if f.get("severity") == "MEDIUM"),
                "n_low": sum(1 for f in parsed["findings"] if f.get("severity") == "LOW"),
                "total_lines_saveable": sum(
                    f.get("estimated_lines_saveable", 0) for f in parsed["findings"]
                ),
            },
        }

        model_results.append(record)
        append_jsonl(output_file, record)

        if result["success"]:
            n = len(parsed["findings"])
            print(f"  OK: {n} redundancies found")
            if n > 0:
                n_high = sum(1 for f in parsed["findings"] if f.get("severity") == "HIGH")
                n_med = sum(1 for f in parsed["findings"] if f.get("severity") == "MEDIUM")
                n_low = sum(1 for f in parsed["findings"] if f.get("severity") == "LOW")
                lines = sum(f.get("estimated_lines_saveable", 0) for f in parsed["findings"])
                print(f"  Severity: H={n_high}, M={n_med}, L={n_low}")
                print(f"  Estimated lines saveable: {lines}")
            if not parsed["parse_success"]:
                print(f"  WARNING: Could not parse JSON from response; raw text captured.")
            print(f"  Tokens: {result['prompt_tokens']}+{result['completion_tokens']}, "
                  f"Latency: {result['latency_ms']}ms")
        else:
            print(f"  ERROR: {result['error']}")

        # Collect all findings for the aggregate summary
        for f in parsed["findings"]:
            f_with_model = dict(f)
            f_with_model["_model"] = model_id
            all_findings.append(f_with_model)

        # Polite delay between API calls
        if idx < len(models):
            time.sleep(2)

    # --- Write summary record ---
    summary_record = {
        "record_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": date_str,
        "n_models": len(models),
        "n_successful": sum(1 for mr in model_results if mr.get("success")),
        "total_findings": len(all_findings),
        "total_high": sum(1 for f in all_findings if f.get("severity") == "HIGH"),
        "total_medium": sum(1 for f in all_findings if f.get("severity") == "MEDIUM"),
        "total_low": sum(1 for f in all_findings if f.get("severity") == "LOW"),
        "total_lines_saveable": sum(f.get("estimated_lines_saveable", 0) for f in all_findings),
        "total_tokens": sum(
            mr.get("prompt_tokens", 0) + mr.get("completion_tokens", 0)
            for mr in model_results
        ),
        "total_latency_ms": sum(mr.get("latency_ms", 0) for mr in model_results),
    }
    append_jsonl(output_file, summary_record)

    # --- Print summary ---
    print_findings_summary(all_findings, model_results)
    print(f"\nResults written to: {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Redundancy detection judge for academic papers. Identifies "
            "semantic redundancy across sections using LLM judges on OpenRouter."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Analyze the full paper\n"
            "  python scripts/redundancy_judge.py \\\n"
            "    --files papers/sosp/intro.tex papers/sosp/background.tex \\\n"
            "           papers/sosp/formal_proof.tex papers/sosp/design.tex \\\n"
            "           papers/sosp/eval.tex papers/sosp/discussion.tex \\\n"
            "           papers/sosp/related.tex papers/sosp/conclusion.tex\n"
            "\n"
            "  # Dry run to preview what would be sent\n"
            "  python scripts/redundancy_judge.py \\\n"
            "    --files papers/sosp/intro.tex papers/sosp/background.tex \\\n"
            "    --dry-run\n"
            "\n"
            "  # Use multiple models for consensus\n"
            "  python scripts/redundancy_judge.py \\\n"
            "    --files papers/sosp/intro.tex papers/sosp/background.tex \\\n"
            "    --models google/gemini-2.5-pro-preview deepseek/deepseek-chat-v3-0324\n"
        ),
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help=(
            "Ordered list of .tex files composing the paper. "
            "Paths relative to project root or absolute."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: reviews/)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Override default model. Provide one or more OpenRouter model IDs. "
            "When multiple models are specified, the tool runs all and provides "
            "a consensus analysis. Default: google/gemini-2.5-pro-preview"
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
    parser.add_argument(
        "--pairwise", action="store_true",
        help=(
            "Use pairwise analysis mode: send each section individually "
            "compared against all others (more API calls, but works within "
            "smaller context windows)"
        ),
    )

    args = parser.parse_args()
    run_redundancy_judge(args)


if __name__ == "__main__":
    main()
