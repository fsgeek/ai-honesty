#!/usr/bin/env python3
"""Conciseness judge for academic papers under page pressure.

Sends each section of a LaTeX paper to an LLM copy editor that suggests
specific prose-tightening edits. Each section is processed independently
(one API call per section), enabling parallel execution and avoiding wasted
tokens on sections that don't need tightening.

Architecture:
  - Parse LaTeX files in order, strip to plain text preserving section structure
    (reuses strip_latex_to_text from redundancy_judge.py)
  - For each section, send to an LLM with copy-editing instructions
  - LLM returns per-paragraph suggestions as structured JSON
  - Output: JSONL report with provenance, per-section suggestions, and summary

Usage:
    # Analyze the full paper
    python scripts/conciseness_judge.py \\
      --files papers/sosp/intro.tex papers/sosp/background.tex \\
             papers/sosp/formal_proof.tex papers/sosp/design.tex \\
             papers/sosp/eval.tex papers/sosp/discussion.tex \\
             papers/sosp/related.tex papers/sosp/conclusion.tex

    # Target specific sections only
    python scripts/conciseness_judge.py \\
      --files papers/sosp/intro.tex papers/sosp/background.tex \\
             papers/sosp/formal_proof.tex papers/sosp/design.tex \\
      --sections Introduction Background

    # Dry run to see what would be sent without API calls
    python scripts/conciseness_judge.py \\
      --files papers/sosp/intro.tex papers/sosp/background.tex \\
      --dry-run

    # Use multiple models for consensus
    python scripts/conciseness_judge.py \\
      --files papers/sosp/intro.tex papers/sosp/background.tex \\
      --models google/gemini-2.5-pro-preview deepseek/deepseek-chat-v3-0324

Environment:
    OPENROUTER_API_KEY: Required. Set in environment or in .env file
    in the project root.

Output:
    reviews/
      conciseness_YYYYMMDD_HHMMSS.jsonl   # All records (provenance + suggestions)
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
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 1
MAX_PARALLEL_CALLS = 4


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
        self.name = self._extract_section_name()
        self.word_count = len(self.plain_text.split())
        self.line_count = len(self.plain_text.splitlines())
        self.sha256 = hashlib.sha256(raw_latex.encode()).hexdigest()

    def _extract_section_name(self) -> str:
        """Extract the section name from LaTeX source, falling back to filename."""
        match = re.search(r'\\section\*?\{([^}]*)\}', self.raw_latex)
        if match:
            name = match.group(1)
            name = name.replace('\\&', '&').replace('\\%', '%')
            name = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', name)
            name = re.sub(r'\\[a-zA-Z]+', '', name)
            return name.strip()
        return Path(self.filepath).stem.replace('_', ' ').title()

    def __repr__(self):
        return f"Section({self.name!r}, {self.word_count} words)"


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


def filter_sections(
    sections: list[Section], names: Optional[list[str]]
) -> list[Section]:
    """Filter sections by name (case-insensitive substring match).

    If names is None, return all sections.
    """
    if not names:
        return sections

    filtered = []
    name_lower = [n.lower() for n in names]
    for s in sections:
        section_lower = s.name.lower()
        if any(n in section_lower or section_lower in n for n in name_lower):
            filtered.append(s)

    if not filtered:
        available = [s.name for s in sections]
        print(f"WARNING: No sections matched {names}.")
        print(f"  Available sections: {available}")
        print("  Using all sections instead.")
        return sections

    return filtered


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
        "X-Title": "AI Honesty Conciseness Judge",
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
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a copy editor for a top-tier systems conference paper under severe page \
pressure. The paper must lose approximately 2 pages. For each paragraph, suggest \
specific edits that preserve meaning while reducing word count. Focus on:
- Wordy phrases that can be shortened (e.g., 'in order to' -> 'to', \
'the fact that' -> 'that', 'it is worth noting that' -> cut entirely)
- Passive voice that can become active (shorter)
- Redundant modifiers ('completely eliminate' -> 'eliminate')
- Sentences that can be merged
- Clauses that add no information

For each suggestion, output JSON with: section, original_text (exact quote, \
1-3 sentences), suggested_text (tightened version), words_saved (integer), \
explanation (brief). Only suggest changes that preserve the technical meaning \
exactly.

You MUST output your suggestions as a JSON array. Each element should be an \
object with these fields:
  - "section": string - the section or subsection name this edit applies to
  - "original_text": string - exact quoted text from the input (1-3 sentences)
  - "suggested_text": string - your tightened replacement
  - "words_saved": integer - number of words saved by this edit
  - "explanation": string - brief explanation of what was changed and why

After the JSON array, provide a brief SUMMARY with:
  - Total suggestions
  - Total words saveable
  - The single highest-impact edit (most words saved)
  - Any paragraphs that are already tight (no suggestions needed)

Output format:
```json
[
  { ... },
  { ... }
]
```

SUMMARY:
..."""


def build_section_prompt(section: Section) -> str:
    """Build the user prompt for analyzing a single section."""
    parts = [
        f"Below is the section \"{section.name}\" from an academic paper, "
        f"stripped of LaTeX formatting for readability. "
        f"The section is {section.word_count} words long.\n\n"
        f"Analyze every paragraph and suggest specific edits to tighten the prose. "
        f"Focus on cuts that preserve technical meaning exactly. Do NOT suggest "
        f"removing technical content, definitions, or evidence \u2014 only tighten "
        f"the language used to express them.\n\n"
        f"Be aggressive but precise. Every word saved is valuable.\n\n",
        "=" * 72,
        f"\nSECTION: {section.name}",
        f"(from: {section.filename}, {section.word_count} words)",
        "=" * 72,
        "",
        section.plain_text,
        "",
        "=" * 72,
        "END OF SECTION",
        "=" * 72,
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_conciseness_response(response_text: str) -> dict:
    """Parse the LLM response into structured conciseness suggestions.

    Returns:
        {
            "suggestions": [...],   # list of suggestion dicts
            "summary_text": str,    # the summary section
            "parse_success": bool,
        }
    """
    result = {
        "suggestions": [],
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
            # No JSON found -- return raw text as summary
            result["summary_text"] = response_text
            return result

    try:
        suggestions = json.loads(json_str)
        if isinstance(suggestions, list):
            valid = []
            for s in suggestions:
                if isinstance(s, dict) and "original_text" in s and "suggested_text" in s:
                    # Normalize words_saved to int
                    try:
                        s["words_saved"] = int(s.get("words_saved", 0))
                    except (ValueError, TypeError):
                        s["words_saved"] = 0

                    # Ensure section field exists
                    s.setdefault("section", "")
                    s.setdefault("explanation", "")

                    valid.append(s)

            result["suggestions"] = valid
            result["parse_success"] = True
    except json.JSONDecodeError:
        pass

    # Extract summary text (everything after the JSON block)
    summary_match = re.search(
        r'(?:```\s*\n|}\s*\]\s*\n)\s*(SUMMARY:.*)',
        response_text, re.DOTALL | re.IGNORECASE,
    )
    if summary_match:
        result["summary_text"] = summary_match.group(1).strip()
    elif not result["parse_success"]:
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

def print_findings_summary(
    section_results: list[dict],
    model_results: list[dict],
):
    """Print a human-readable summary of conciseness suggestions."""
    print(f"\n{'=' * 72}")
    print("CONCISENESS ANALYSIS SUMMARY")
    print(f"{'=' * 72}")

    if not section_results:
        print("\nNo suggestions generated.")
        return

    total_suggestions = 0
    total_words_saved = 0
    best_edit = None
    best_words = 0

    # Per-section breakdown
    print(f"\n{'Section':40s}  {'Suggestions':>11s}  {'Words Saved':>11s}")
    print("-" * 66)

    for sr in section_results:
        section_name = sr.get("section_name", "?")
        suggestions = sr.get("suggestions", [])
        n = len(suggestions)
        words = sum(s.get("words_saved", 0) for s in suggestions)
        total_suggestions += n
        total_words_saved += words

        print(f"  {section_name:38s}  {n:>11d}  {words:>11d}")

        # Track best single edit
        for s in suggestions:
            ws = s.get("words_saved", 0)
            if ws > best_words:
                best_words = ws
                best_edit = {**s, "_section_name": section_name}

    print("-" * 66)
    print(f"  {'TOTAL':38s}  {total_suggestions:>11d}  {total_words_saved:>11d}")

    # Rough page estimate (assume ~300 words per column, ~600 per page)
    if total_words_saved > 0:
        est_pages = total_words_saved / 600.0
        print(f"\n  Estimated page savings: ~{est_pages:.1f} pages "
              f"(at ~600 words/page)")

    # Highest-impact edit
    if best_edit:
        print(f"\n{'=' * 72}")
        print("HIGHEST-IMPACT EDIT")
        print(f"{'=' * 72}")
        print(f"  Section: {best_edit.get('_section_name', '?')}")
        orig = best_edit.get("original_text", "")
        if len(orig) > 200:
            orig = orig[:200] + "..."
        print(f"  Original: \"{orig}\"")
        sugg = best_edit.get("suggested_text", "")
        if len(sugg) > 200:
            sugg = sugg[:200] + "..."
        print(f"  Suggested: \"{sugg}\"")
        print(f"  Words saved: {best_edit.get('words_saved', 0)}")
        if best_edit.get("explanation"):
            print(f"  Explanation: {best_edit['explanation']}")

    # Per-model breakdown (if multiple models)
    if len(model_results) > 1:
        print(f"\n{'=' * 72}")
        print("PER-MODEL BREAKDOWN")
        print(f"{'=' * 72}")
        for model_id, results in _group_by_model(model_results):
            n_sugg = sum(
                len(r.get("suggestions", []))
                for r in results
            )
            words = sum(
                sum(s.get("words_saved", 0) for s in r.get("suggestions", []))
                for r in results
            )
            print(f"\n  {model_id}:")
            print(f"    Suggestions: {n_sugg}")
            print(f"    Words saveable: {words}")

    # API usage
    total_tokens = sum(
        r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
        for r in model_results
    )
    total_latency = sum(r.get("latency_ms", 0) for r in model_results)
    n_success = sum(1 for r in model_results if r.get("success"))

    print(f"\nAPI Usage:")
    print(f"  Total calls: {len(model_results)} ({n_success} successful)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total latency: {total_latency / 1000:.1f}s")
    print(f"{'=' * 72}")


def _group_by_model(results: list[dict]) -> list[tuple]:
    """Group results by model_id, returning (model_id, [results])."""
    groups = {}
    for r in results:
        mid = r.get("model_id", "unknown")
        groups.setdefault(mid, []).append(r)
    return list(groups.items())


# ---------------------------------------------------------------------------
# Parallel section processing
# ---------------------------------------------------------------------------

def process_section(
    section: Section,
    model_id: str,
    temperature: float,
    api_key: str,
) -> dict:
    """Process a single section through the LLM and return results.

    This function is designed to be called in a thread pool.
    """
    user_prompt = build_section_prompt(section)
    result = call_with_retry(
        model=model_id,
        temperature=temperature,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        api_key=api_key,
        max_tokens=8192,
    )

    parsed = parse_conciseness_response(result["response_text"])

    return {
        "section_name": section.name,
        "section_filename": section.filename,
        "section_filepath": section.filepath,
        "section_word_count": section.word_count,
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
        "suggestions": parsed["suggestions"],
        "summary_text": parsed["summary_text"],
        "parse_success": parsed["parse_success"],
        "n_suggestions": len(parsed["suggestions"]),
        "total_words_saved": sum(
            s.get("words_saved", 0) for s in parsed["suggestions"]
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_conciseness_judge(args) -> Optional[Path]:
    """Execute the conciseness judge pipeline.

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

    # --- Load and filter sections ---
    all_sections = load_sections(args.files)
    if not all_sections:
        print("ERROR: No sections loaded.")
        sys.exit(1)

    sections = filter_sections(
        all_sections,
        getattr(args, "sections", None),
    )

    # --- Configure models ---
    temperature = getattr(args, "temperature", DEFAULT_TEMPERATURE)
    if args.models:
        models = [(m, temperature) for m in args.models]
    else:
        models = [(DEFAULT_MODEL, temperature)]

    # --- Output setup ---
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else project_root / "reviews"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"conciseness_{date_str}.jsonl"

    # Total API calls: one per section per model
    n_calls = len(sections) * len(models)

    # --- Print configuration ---
    print("=" * 72)
    print("CONCISENESS JUDGE CONFIGURATION")
    print("=" * 72)

    total_words = sum(s.word_count for s in sections)
    total_lines = sum(s.line_count for s in sections)

    print(f"\nSections to analyze ({len(sections)}):")
    for i, s in enumerate(sections, 1):
        sha_short = s.sha256[:12]
        print(
            f"  {i}. {s.name:30s} ({s.filename}, {s.word_count} words, "
            f"{s.line_count} lines, SHA256: {sha_short}...)"
        )

    if len(all_sections) > len(sections):
        skipped = [s.name for s in all_sections if s not in sections]
        print(f"\n  Skipped sections: {skipped}")

    print(f"\nTotal: {total_words} words, {total_lines} lines")

    print(f"\nModels ({len(models)}):")
    for model_id, temp in models:
        print(f"  {model_id} (temp={temp})")

    print(f"\nAPI calls planned: {n_calls} "
          f"({len(sections)} sections x {len(models)} models)")
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

        # Show a sample section prompt
        if sections:
            sample = sections[0]
            sample_prompt = build_section_prompt(sample)
            print(f"\n--- SAMPLE USER PROMPT: {sample.name} "
                  f"({len(sample_prompt)} chars) ---")
            print(sample_prompt[:3000])
            if len(sample_prompt) > 3000:
                print(f"\n[... {len(sample_prompt) - 3000} more chars ...]")
            print("--- END SAMPLE PROMPT ---")

        return None

    # --- Write provenance record ---
    provenance = {
        "record_type": "provenance",
        "timestamp": now.isoformat(),
        "run_id": date_str,
        "tool": "conciseness_judge",
        "sections": [
            {
                "index": i,
                "name": s.name,
                "filename": s.filename,
                "filepath": s.filepath,
                "word_count": s.word_count,
                "line_count": s.line_count,
                "char_count": len(s.plain_text),
                "sha256": s.sha256,
            }
            for i, s in enumerate(sections)
        ],
        "total_words": total_words,
        "total_lines": total_lines,
        "models": [{"model": m, "temperature": t} for m, t in models],
        "n_api_calls": n_calls,
    }
    append_jsonl(output_file, provenance)

    # --- Process sections ---
    print(f"\n{'=' * 72}")
    print("RUNNING CONCISENESS ANALYSIS")
    print(f"{'=' * 72}")

    all_results = []
    call_index = 0

    for model_id, temp in models:
        print(f"\nModel: {model_id} (temp={temp})")
        print("-" * 60)

        # Build list of tasks for this model
        tasks = [(section, model_id, temp, api_key) for section in sections]

        # Process sections in parallel
        futures_map = {}
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as executor:
            for section, mid, t, key in tasks:
                future = executor.submit(process_section, section, mid, t, key)
                futures_map[future] = section.name

            for future in as_completed(futures_map):
                section_name = futures_map[future]
                call_index += 1

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "section_name": section_name,
                        "model_id": model_id,
                        "success": False,
                        "error": str(e),
                        "suggestions": [],
                        "n_suggestions": 0,
                        "total_words_saved": 0,
                    }

                all_results.append(result)

                # Write to JSONL incrementally
                record = {
                    "record_type": "section_analysis",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": date_str,
                    "call_index": call_index - 1,
                    **result,
                }
                append_jsonl(output_file, record)

                # Print progress
                if result.get("success"):
                    n = result.get("n_suggestions", 0)
                    words = result.get("total_words_saved", 0)
                    print(
                        f"  [{call_index}/{n_calls}] {section_name}: "
                        f"{n} suggestions, {words} words saveable"
                    )
                    ptok = result.get("prompt_tokens", 0)
                    ctok = result.get("completion_tokens", 0)
                    lat = result.get("latency_ms", 0)
                    print(f"    Tokens: {ptok}+{ctok}, Latency: {lat}ms")
                    if not result.get("parse_success"):
                        print(
                            f"    WARNING: Could not parse JSON from response; "
                            f"raw text captured."
                        )
                else:
                    print(
                        f"  [{call_index}/{n_calls}] {section_name}: "
                        f"ERROR: {result.get('error', 'unknown')}"
                    )

    # --- Build per-section aggregated results for summary ---
    section_summary = {}
    for r in all_results:
        sname = r.get("section_name", "?")
        if sname not in section_summary:
            section_summary[sname] = {
                "section_name": sname,
                "suggestions": [],
            }
        section_summary[sname]["suggestions"].extend(
            r.get("suggestions", [])
        )
    section_results = list(section_summary.values())

    # --- Write summary record ---
    total_suggestions = sum(r.get("n_suggestions", 0) for r in all_results)
    total_words_saved = sum(r.get("total_words_saved", 0) for r in all_results)

    summary_record = {
        "record_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": date_str,
        "n_models": len(models),
        "n_sections": len(sections),
        "n_successful": sum(1 for r in all_results if r.get("success")),
        "total_suggestions": total_suggestions,
        "total_words_saved": total_words_saved,
        "estimated_page_savings": round(total_words_saved / 600.0, 2),
        "total_tokens": sum(
            r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
            for r in all_results
        ),
        "total_latency_ms": sum(
            r.get("latency_ms", 0) for r in all_results
        ),
    }
    append_jsonl(output_file, summary_record)

    # --- Print summary ---
    print_findings_summary(section_results, all_results)
    print(f"\nResults written to: {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Conciseness judge for academic papers. Sends each section to an "
            "LLM copy editor that suggests specific prose-tightening edits "
            "to reduce page count while preserving technical meaning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Analyze the full paper\n"
            "  python scripts/conciseness_judge.py \\\n"
            "    --files papers/sosp/intro.tex papers/sosp/background.tex \\\n"
            "           papers/sosp/formal_proof.tex papers/sosp/design.tex \\\n"
            "           papers/sosp/eval.tex papers/sosp/discussion.tex \\\n"
            "           papers/sosp/related.tex papers/sosp/conclusion.tex\n"
            "\n"
            "  # Target specific sections by name\n"
            "  python scripts/conciseness_judge.py \\\n"
            "    --files papers/sosp/intro.tex papers/sosp/background.tex \\\n"
            "           papers/sosp/formal_proof.tex \\\n"
            "    --sections Introduction Background\n"
            "\n"
            "  # Dry run to preview what would be sent\n"
            "  python scripts/conciseness_judge.py \\\n"
            "    --files papers/sosp/intro.tex papers/sosp/background.tex \\\n"
            "    --dry-run\n"
            "\n"
            "  # Use multiple models for consensus\n"
            "  python scripts/conciseness_judge.py \\\n"
            "    --files papers/sosp/intro.tex papers/sosp/background.tex \\\n"
            "    --models google/gemini-2.5-pro-preview "
            "deepseek/deepseek-chat-v3-0324\n"
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
        "--sections", nargs="+", default=None,
        help=(
            "Target specific sections by name (case-insensitive substring "
            "match). If not specified, all sections are analyzed. "
            "Example: --sections Introduction Background"
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
            "a combined analysis. Default: google/gemini-2.5-pro-preview"
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
    run_conciseness_judge(args)


if __name__ == "__main__":
    main()
