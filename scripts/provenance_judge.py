#!/usr/bin/env python3
"""Provenance judge for academic papers: traces claims backward to code and data.

Performs TWO non-commutative passes:

  Pass 1 (Paper -> Data): Extracts every empirical claim from the paper LaTeX
  files, then attempts to verify each against CSV data files. Reports whether
  numbers match, cannot be found, or contradict the data.

  Pass 2 (Code -> Paper): Reads experiment scripts, extracts what they actually
  do (model IDs, evaluation logic, data transformations), and checks whether
  the paper accurately describes them.

The composition is non-commutative: Pass 1 asks "does the paper match the data?"
while Pass 2 asks "does the paper describe what the code does?" These yield
different findings even when applied to the same paper.

Architecture:
  - Reuses strip_latex_to_text from redundancy_judge.py
  - OpenRouter API with retry logic (same as redundancy_judge, conciseness_judge)
  - Parallel execution for independent API calls
  - JSONL output with SHA256 provenance records
  - Programmatic CSV verification where possible, LLM for qualitative claims

Usage:
    # Both passes (default)
    python scripts/provenance_judge.py --paper-dir papers/sosp/

    # Paper-to-data only
    python scripts/provenance_judge.py --paper-dir papers/sosp/ --pass paper-to-data

    # Code-to-paper only
    python scripts/provenance_judge.py --paper-dir papers/sosp/ --pass code-to-paper

    # Custom directories
    python scripts/provenance_judge.py \\
      --paper-dir papers/sosp/ --data-dir . --scripts-dir scripts/

Environment:
    OPENROUTER_API_KEY: Required. Set in environment or in .env file
    in the project root.

Output:
    reviews/
      provenance_YYYYMMDD_HHMMSS.jsonl   # All records (provenance + findings)
"""

import argparse
import csv
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
DEFAULT_TEMPERATURE = 0.2
MAX_RETRIES = 2
MAX_PARALLEL_CALLS = 3


# ---------------------------------------------------------------------------
# OpenRouter API (matches project patterns from redundancy_judge.py)
# ---------------------------------------------------------------------------

def call_openrouter(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int = 16384,
) -> dict:
    """Call OpenRouter API and return full response dict."""
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fsgeek/ai-honesty",
        "X-Title": "AI Honesty Provenance Judge",
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
    max_tokens: int = 16384,
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
        time.sleep(3 * retries)  # Exponential-ish backoff
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
# File loading and hashing
# ---------------------------------------------------------------------------

def sha256_of_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tex_files(paper_dir: Path) -> list[dict]:
    """Load all .tex files from the paper directory.

    Returns a list of dicts with keys: filepath, filename, raw, plain, sha256, line_count.
    Only includes files that are part of the actual paper (heuristic: has a \\section
    or is the main .tex file).
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

    # Filter to files that look like paper sections (have \section or \begin{abstract})
    sections = []
    for tf in tex_files:
        raw = tf.read_text()
        # Skip files that are clearly not paper sections
        if tf.name.startswith("_") or tf.name.startswith("."):
            continue
        # Include if it has a section heading or is a known paper file
        has_section = bool(re.search(r'\\(?:section|subsection|begin\{abstract\})', raw))
        is_known = tf.name in preferred_order
        # Also include the main .tex file (contains \documentclass)
        is_main = bool(re.search(r'\\documentclass', raw))

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


def load_csv_files(data_dir: Path) -> list[dict]:
    """Load CSV data files from the data directory.

    Focuses on experiment output CSVs (exp* pattern), plus other project CSVs.
    Returns list of dicts with keys: filepath, filename, sha256, headers, rows, preview.
    """
    csv_patterns = ["exp*.csv", "*_results.csv", "*_summary.csv",
                    "calibration_*.csv", "signal_*.csv", "benchmark_*.csv"]

    csv_files = set()
    for pattern in csv_patterns:
        csv_files.update(data_dir.glob(pattern))

    results = []
    for cf in sorted(csv_files):
        # Skip files in .venv or other non-project directories
        if ".venv" in str(cf) or "site-packages" in str(cf):
            continue
        try:
            with open(cf, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                continue
            headers = rows[0]
            data_rows = rows[1:]

            # Build a preview: first 10 rows as text
            preview_lines = [",".join(headers)]
            for row in data_rows[:10]:
                preview_lines.append(",".join(row))
            preview = "\n".join(preview_lines)

            results.append({
                "filepath": str(cf),
                "filename": cf.name,
                "sha256": sha256_of_file(cf),
                "headers": headers,
                "rows": data_rows,
                "n_rows": len(data_rows),
                "preview": preview,
            })
        except Exception as e:
            print(f"  WARNING: Could not read {cf}: {e}")

    return results


def load_script_files(scripts_dir: Path) -> list[dict]:
    """Load experiment scripts from the scripts directory.

    Returns list of dicts with keys: filepath, filename, sha256, content, line_count.
    """
    script_files = sorted(scripts_dir.glob("experiment*.py"))
    results = []
    for sf in script_files:
        try:
            content = sf.read_text()
            results.append({
                "filepath": str(sf),
                "filename": sf.name,
                "sha256": sha256_of_file(sf),
                "content": content,
                "line_count": len(content.splitlines()),
            })
        except Exception as e:
            print(f"  WARNING: Could not read {sf}: {e}")
    return results


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def append_jsonl(output_file: Path, record: dict):
    """Append a single JSON record to the output file."""
    with open(output_file, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Pass 1: Paper -> Data
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_SYSTEM = """\
You are a meticulous research auditor extracting empirical claims from an \
academic paper. Your task is to find every claim that could be verified \
against data or code.

For each empirical claim, extract:
  - "claim_text": the exact text from the paper (1-3 sentences, verbatim)
  - "claim_type": one of "numeric" (contains specific numbers), \
"comparative" (A > B, A outperforms B), "qualitative" (universal claims \
like "all models", "every architecture", "worse than random")
  - "numbers": list of specific numbers mentioned (percentages, AUC values, \
counts, etc.) as strings preserving formatting, or empty list if qualitative
  - "section": which section/subsection this appears in
  - "context": brief description of what the claim is about (1 sentence)

Focus on:
  - Accuracy percentages and AUC values
  - Claims about number of queries, models, or experimental conditions
  - Comparative claims (X outperforms Y, X is better than Y at budget Z)
  - Universal claims (all models, every architecture, universal)
  - Effect sizes (Cohen's d values, correlation coefficients)
  - Ground truth validation numbers (agreement rates)
  - Model names and counts

Do NOT extract:
  - Pure definitions or conceptual statements
  - Claims attributed to other papers (citations)
  - Hypothetical statements or future work
  - Formatting descriptions (e.g., "Table 1 shows...")

Output as a JSON array:
```json
[
  {
    "claim_text": "exact text from paper",
    "claim_type": "numeric",
    "numbers": ["82.1%", "78.5%"],
    "section": "Evaluation",
    "context": "tensor-guided vs text-guided accuracy at specific budgets"
  },
  ...
]
```"""


def build_claim_extraction_prompt(tex_sections: list[dict]) -> str:
    """Build prompt for extracting empirical claims from paper sections."""
    separator = "=" * 72
    parts = [
        "Below is an academic paper, stripped of LaTeX formatting. "
        "Extract every empirical claim that could be verified against "
        "experimental data or code. Be exhaustive.\n\n",
        separator,
    ]

    for section in tex_sections:
        parts.append(f"\n{separator}")
        parts.append(f"FILE: {section['filename']}")
        parts.append(f"{separator}\n")
        parts.append(section["plain"])
        parts.append("")

    parts.append(f"\n{separator}")
    parts.append("END OF PAPER")
    parts.append(separator)

    return "\n".join(parts)


def parse_claims_response(response_text: str) -> list[dict]:
    """Parse the LLM response into a list of claim dicts."""
    if not response_text:
        return []

    # Extract JSON from response
    json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_match = re.search(r'(\[\s*\{.*?\}\s*\])', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            return []

    try:
        claims = json.loads(json_str)
        if isinstance(claims, list):
            valid = []
            for c in claims:
                if isinstance(c, dict) and "claim_text" in c:
                    c.setdefault("claim_type", "qualitative")
                    c.setdefault("numbers", [])
                    c.setdefault("section", "unknown")
                    c.setdefault("context", "")
                    valid.append(c)
            return valid
    except json.JSONDecodeError:
        pass

    return []


def extract_numbers_from_text(text: str) -> list[str]:
    """Extract all number-like strings from text (percentages, decimals, integers)."""
    # Match patterns like: 82.1%, 0.870, 75/80, 93.8%, 3x, 200, 0.762
    patterns = [
        r'\d+\.\d+%',        # 82.1%
        r'\d+%',             # 10%
        r'\d+\.\d+',         # 0.870
        r'\d+/\d+',          # 75/80
        r'\d+',              # 200
    ]
    numbers = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            numbers.append(match.group(0))
    return numbers


def normalize_number(s: str) -> Optional[float]:
    """Normalize a number string to a float for comparison."""
    s = s.strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        # Handle fractions like 75/80
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
        return None


def search_csv_for_number(
    csv_files: list[dict],
    target_number: float,
    tolerance: float = 0.015,
) -> list[dict]:
    """Search all CSV files for values matching the target number.

    Returns list of matches with file, row, column info.
    """
    matches = []
    for csv_file in csv_files:
        for row_idx, row in enumerate(csv_file["rows"]):
            for col_idx, cell in enumerate(row):
                cell_val = normalize_number(cell)
                if cell_val is not None:
                    # Check for direct match (absolute tolerance)
                    if abs(cell_val - target_number) <= tolerance:
                        col_name = (
                            csv_file["headers"][col_idx]
                            if col_idx < len(csv_file["headers"])
                            else f"col_{col_idx}"
                        )
                        matches.append({
                            "file": csv_file["filename"],
                            "filepath": csv_file["filepath"],
                            "row": row_idx + 2,  # +2 for 1-indexed + header
                            "column": col_name,
                            "value": cell,
                            "parsed_value": cell_val,
                            "full_row": dict(zip(csv_file["headers"], row)),
                        })
                    # Also check percentage form (0.821 matches 82.1)
                    elif abs(cell_val * 100 - target_number) <= tolerance:
                        col_name = (
                            csv_file["headers"][col_idx]
                            if col_idx < len(csv_file["headers"])
                            else f"col_{col_idx}"
                        )
                        matches.append({
                            "file": csv_file["filename"],
                            "filepath": csv_file["filepath"],
                            "row": row_idx + 2,
                            "column": col_name,
                            "value": cell,
                            "parsed_value": cell_val,
                            "note": f"matched as percentage ({cell_val} * 100 = {cell_val * 100})",
                            "full_row": dict(zip(csv_file["headers"], row)),
                        })
                    # Check inverse: target is 0.821, CSV has 82.1
                    elif target_number < 1.0 and abs(cell_val - target_number * 100) <= tolerance:
                        col_name = (
                            csv_file["headers"][col_idx]
                            if col_idx < len(csv_file["headers"])
                            else f"col_{col_idx}"
                        )
                        matches.append({
                            "file": csv_file["filename"],
                            "filepath": csv_file["filepath"],
                            "row": row_idx + 2,
                            "column": col_name,
                            "value": cell,
                            "parsed_value": cell_val,
                            "note": f"matched as raw value ({target_number} * 100 = {target_number * 100})",
                            "full_row": dict(zip(csv_file["headers"], row)),
                        })
    return matches


def verify_claim_against_data(
    claim: dict,
    csv_files: list[dict],
) -> dict:
    """Verify a single claim against CSV data.

    Returns a finding dict with status and evidence.
    """
    numbers = claim.get("numbers", [])
    if not numbers:
        return {
            "status": "UNVERIFIED",
            "severity": "LOW",
            "evidence": "No specific numbers to verify against data.",
            "evidence_location": None,
            "matches": [],
        }

    all_matches = []
    verified_count = 0
    discrepancy_count = 0

    for num_str in numbers:
        # Clean up the number string
        clean = num_str.strip().rstrip("%").replace(",", "")
        target = normalize_number(clean)
        if target is None:
            continue

        # If the original had %, the paper number is a percentage
        is_percent = "%" in num_str
        search_target = target

        matches = search_csv_for_number(csv_files, search_target)

        if matches:
            verified_count += 1
            all_matches.extend(matches[:3])  # Keep top 3 matches per number
        else:
            discrepancy_count += 1

    if not all_matches and numbers:
        # No numbers found in any CSV
        return {
            "status": "UNVERIFIED",
            "severity": "MEDIUM",
            "evidence": f"Could not find {numbers} in any CSV file.",
            "evidence_location": None,
            "matches": [],
        }
    elif discrepancy_count > 0 and verified_count == 0:
        return {
            "status": "UNVERIFIED",
            "severity": "MEDIUM",
            "evidence": (
                f"None of the claimed numbers {numbers} found in CSV data. "
                f"Numbers may be computed (averages, derived metrics) rather "
                f"than stored directly."
            ),
            "evidence_location": None,
            "matches": [],
        }
    elif verified_count > 0:
        match_descriptions = []
        for m in all_matches[:5]:
            desc = f"{m['file']}:{m['row']} ({m['column']}={m['value']})"
            if m.get("note"):
                desc += f" [{m['note']}]"
            match_descriptions.append(desc)

        return {
            "status": "VERIFIED",
            "severity": "LOW",
            "evidence": f"Found {verified_count}/{len(numbers)} numbers in CSV data.",
            "evidence_location": "; ".join(match_descriptions),
            "matches": all_matches[:5],
        }
    else:
        return {
            "status": "UNVERIFIED",
            "severity": "LOW",
            "evidence": "Claim has no verifiable numbers.",
            "evidence_location": None,
            "matches": [],
        }


QUALITATIVE_VERIFICATION_SYSTEM = """\
You are a research auditor verifying qualitative claims from an academic paper \
against experimental data. You will be given a claim from the paper and \
relevant CSV data previews.

Determine whether the data supports the claim. Consider:
  - Does the data show what the claim asserts?
  - Are there any rows/values that contradict the claim?
  - Is the claim a fair characterization of the data?

Respond with a JSON object:
```json
{
  "status": "VERIFIED" | "DISCREPANCY" | "UNVERIFIED",
  "severity": "HIGH" | "MEDIUM" | "LOW",
  "evidence": "what the data actually shows (specific values)",
  "explanation": "why this is a match or mismatch"
}
```

Severity guide:
  - HIGH: Claim directly contradicts data (wrong number, wrong direction)
  - MEDIUM: Claim is misleading or overstated given the data
  - LOW: Claim is supported or data is ambiguous"""


def verify_qualitative_claim(
    claim: dict,
    csv_files: list[dict],
    model: str,
    temperature: float,
    api_key: str,
) -> dict:
    """Use LLM to verify a qualitative or comparative claim against CSV data.

    Returns finding dict with status and evidence.
    """
    # Build context from relevant CSV files
    csv_context_parts = []
    for cf in csv_files:
        csv_context_parts.append(f"\n--- {cf['filename']} ({cf['n_rows']} data rows) ---")
        csv_context_parts.append(cf["preview"])
    csv_context = "\n".join(csv_context_parts)

    if not csv_context.strip():
        return {
            "status": "UNVERIFIED",
            "severity": "LOW",
            "evidence": "No CSV data available for verification.",
            "evidence_location": None,
        }

    user_prompt = (
        f"CLAIM FROM PAPER:\n"
        f'"{claim["claim_text"]}"\n\n'
        f"Claim type: {claim.get('claim_type', 'unknown')}\n"
        f"Context: {claim.get('context', 'none')}\n\n"
        f"AVAILABLE DATA:\n{csv_context}\n\n"
        f"Verify this claim against the data. Does the data support it?"
    )

    result = call_with_retry(
        model=model,
        temperature=temperature,
        system_prompt=QUALITATIVE_VERIFICATION_SYSTEM,
        user_prompt=user_prompt,
        api_key=api_key,
        max_tokens=2048,
    )

    if not result["success"]:
        return {
            "status": "UNVERIFIED",
            "severity": "LOW",
            "evidence": f"LLM verification failed: {result['error']}",
            "evidence_location": None,
            "api_error": result["error"],
        }

    # Parse LLM response
    resp = result["response_text"]
    json_match = re.search(r'```json\s*\n(.*?)\n```', resp, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_match = re.search(r'(\{[^{}]*"status"[^{}]*\})', resp, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            return {
                "status": "UNVERIFIED",
                "severity": "LOW",
                "evidence": f"Could not parse LLM response.",
                "evidence_location": None,
                "raw_response": resp[:500],
            }

    try:
        parsed = json.loads(json_str)
        status = parsed.get("status", "UNVERIFIED").upper()
        if status not in ("VERIFIED", "DISCREPANCY", "UNVERIFIED"):
            status = "UNVERIFIED"

        severity = parsed.get("severity", "LOW").upper()
        if severity not in ("HIGH", "MEDIUM", "LOW"):
            severity = "LOW"

        return {
            "status": status,
            "severity": severity,
            "evidence": parsed.get("evidence", ""),
            "explanation": parsed.get("explanation", ""),
            "evidence_location": None,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        }
    except json.JSONDecodeError:
        return {
            "status": "UNVERIFIED",
            "severity": "LOW",
            "evidence": "Could not parse LLM verification response as JSON.",
            "evidence_location": None,
            "raw_response": resp[:500],
        }


def run_paper_to_data(
    tex_sections: list[dict],
    csv_files: list[dict],
    model: str,
    temperature: float,
    api_key: str,
    output_file: Path,
    run_id: str,
    dry_run: bool = False,
) -> list[dict]:
    """Execute Pass 1: Paper -> Data.

    Extract claims from paper, verify against CSV data.
    Returns list of finding dicts.
    """
    print(f"\n{'=' * 72}")
    print("PASS 1: PAPER -> DATA")
    print(f"{'=' * 72}")

    if dry_run:
        prompt = build_claim_extraction_prompt(tex_sections)
        print(f"\n[DRY RUN] Would send {len(prompt)} chars for claim extraction")
        print(f"[DRY RUN] Would verify against {len(csv_files)} CSV files")
        return []

    # Step 1: Extract claims from paper via LLM
    print(f"\n  Step 1: Extracting empirical claims from paper...")
    prompt = build_claim_extraction_prompt(tex_sections)

    result = call_with_retry(
        model=model,
        temperature=temperature,
        system_prompt=CLAIM_EXTRACTION_SYSTEM,
        user_prompt=prompt,
        api_key=api_key,
        max_tokens=16384,
    )

    if not result["success"]:
        print(f"  ERROR: Claim extraction failed: {result['error']}")
        return []

    claims = parse_claims_response(result["response_text"])
    print(f"  Extracted {len(claims)} empirical claims")
    print(f"  Tokens: {result['prompt_tokens']}+{result['completion_tokens']}, "
          f"Latency: {result['latency_ms']}ms")

    # Write extraction record
    extraction_record = {
        "record_type": "claim_extraction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "pass": "paper_to_data",
        "model": model,
        "n_claims": len(claims),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "latency_ms": result["latency_ms"],
    }
    append_jsonl(output_file, extraction_record)

    # Step 2: Verify each claim against data
    print(f"\n  Step 2: Verifying {len(claims)} claims against {len(csv_files)} CSV files...")

    findings = []
    n_numeric = sum(1 for c in claims if c.get("numbers"))
    n_qualitative = len(claims) - n_numeric

    # First pass: programmatic verification for numeric claims
    numeric_claims = [c for c in claims if c.get("numbers")]
    qualitative_claims = [c for c in claims if not c.get("numbers")]

    for i, claim in enumerate(numeric_claims):
        finding_id = f"P2D-{i + 1:03d}"
        verification = verify_claim_against_data(claim, csv_files)

        finding = {
            "record_type": "finding",
            "finding_id": finding_id,
            "pass": "paper_to_data",
            "severity": verification["severity"],
            "status": verification["status"],
            "paper_claim": claim["claim_text"],
            "paper_location": f"{claim.get('section', 'unknown')}",
            "claim_type": claim.get("claim_type", "numeric"),
            "claimed_numbers": claim.get("numbers", []),
            "evidence_location": verification.get("evidence_location"),
            "evidence": verification["evidence"],
            "explanation": verification.get("explanation", ""),
            "csv_matches": [
                {k: v for k, v in m.items() if k != "full_row"}
                for m in verification.get("matches", [])
            ],
        }
        findings.append(finding)
        append_jsonl(output_file, finding)

        status_icon = {
            "VERIFIED": "+", "DISCREPANCY": "!", "UNVERIFIED": "?"
        }.get(verification["status"], "?")
        print(f"    [{status_icon}] P2D-{i + 1:03d} ({verification['status']}): "
              f"{claim['claim_text'][:80]}...")

    # Second pass: LLM verification for qualitative/comparative claims
    if qualitative_claims:
        print(f"\n  Verifying {len(qualitative_claims)} qualitative claims via LLM...")

        # Process qualitative claims in parallel
        def verify_one_qualitative(args):
            idx, claim = args
            return idx, verify_qualitative_claim(
                claim, csv_files, model, temperature, api_key
            )

        tasks = list(enumerate(qualitative_claims, start=len(numeric_claims)))

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as executor:
            futures = {
                executor.submit(verify_one_qualitative, (idx, claim)): (idx, claim)
                for idx, claim in tasks
            }

            for future in as_completed(futures):
                idx, claim = futures[future]
                finding_id = f"P2D-{idx + 1:03d}"

                try:
                    _, verification = future.result()
                except Exception as e:
                    verification = {
                        "status": "UNVERIFIED",
                        "severity": "LOW",
                        "evidence": f"Verification failed: {e}",
                        "explanation": "",
                    }

                finding = {
                    "record_type": "finding",
                    "finding_id": finding_id,
                    "pass": "paper_to_data",
                    "severity": verification.get("severity", "LOW"),
                    "status": verification.get("status", "UNVERIFIED"),
                    "paper_claim": claim["claim_text"],
                    "paper_location": f"{claim.get('section', 'unknown')}",
                    "claim_type": claim.get("claim_type", "qualitative"),
                    "claimed_numbers": claim.get("numbers", []),
                    "evidence_location": verification.get("evidence_location"),
                    "evidence": verification.get("evidence", ""),
                    "explanation": verification.get("explanation", ""),
                }
                findings.append(finding)
                append_jsonl(output_file, finding)

                status_icon = {
                    "VERIFIED": "+", "DISCREPANCY": "!", "UNVERIFIED": "?"
                }.get(verification.get("status", "?"), "?")
                print(f"    [{status_icon}] {finding_id} ({verification.get('status', '?')}): "
                      f"{claim['claim_text'][:80]}...")

    return findings


# ---------------------------------------------------------------------------
# Pass 2: Code -> Paper
# ---------------------------------------------------------------------------

CODE_AUDIT_SYSTEM = """\
You are a research auditor comparing experiment code against its description \
in an academic paper. Your task is to identify where the paper accurately \
describes, inaccurately describes, or fails to mention what the code does.

For each finding, report:
  - "status": one of:
    - "CONSISTENT": paper accurately describes what the code does
    - "INCONSISTENT": paper says X but code does Y (specific mismatch)
    - "UNDISCLOSED": code does something significant the paper does not mention
  - "severity": HIGH (would affect reproducibility or conclusions), MEDIUM \
(misleading but not fatal), LOW (minor omission or imprecision)
  - "paper_claim": the relevant text from the paper (exact quote if possible)
  - "code_evidence": what the code actually does (specific lines, values, logic)
  - "explanation": why this matters

Focus on:
  - Model IDs: does the paper list the exact same models the code uses?
  - Evaluation logic: does the paper describe the same criteria for correct/incorrect?
  - Data transformations: averaging, filtering, thresholding
  - Query counts and categories
  - Budget levels and verification strategies
  - Any hardcoded parameters that differ from paper description

Output as a JSON array:
```json
[
  {
    "status": "CONSISTENT" | "INCONSISTENT" | "UNDISCLOSED",
    "severity": "HIGH" | "MEDIUM" | "LOW",
    "paper_claim": "text from paper",
    "code_evidence": "what the code does (with line references)",
    "explanation": "why this is a match/mismatch/omission"
  },
  ...
]
```"""


def build_code_audit_prompt(
    script: dict,
    paper_text: str,
) -> str:
    """Build prompt for auditing a script against the paper."""
    # Truncate very long scripts to focus on the important parts
    code = script["content"]
    if len(code) > 15000:
        # Keep the docstring, imports, config, and main function
        lines = code.split("\n")
        # Keep first 200 lines (docstring + imports + config) and last 200 lines
        if len(lines) > 400:
            code = "\n".join(lines[:200]) + "\n\n... [TRUNCATED] ...\n\n" + "\n".join(lines[-200:])

    parts = [
        f"EXPERIMENT SCRIPT: {script['filename']}\n",
        "=" * 72,
        code,
        "=" * 72,
        f"\nPAPER DESCRIPTION (all sections):\n",
        "=" * 72,
        paper_text,
        "=" * 72,
        "\nCompare the code against the paper. Identify every place where the "
        "paper describes something differently from what the code does, and "
        "every significant thing the code does that the paper does not mention. "
        "Also confirm things the paper gets right.",
    ]
    return "\n".join(parts)


def parse_code_audit_response(response_text: str) -> list[dict]:
    """Parse the LLM response into a list of code audit finding dicts."""
    if not response_text:
        return []

    json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_match = re.search(r'(\[\s*\{.*?\}\s*\])', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            return []

    try:
        findings = json.loads(json_str)
        if isinstance(findings, list):
            valid = []
            for f in findings:
                if isinstance(f, dict) and "status" in f:
                    status = f.get("status", "").upper()
                    if status not in ("CONSISTENT", "INCONSISTENT", "UNDISCLOSED"):
                        status = "UNDISCLOSED"
                    f["status"] = status

                    severity = f.get("severity", "LOW").upper()
                    if severity not in ("HIGH", "MEDIUM", "LOW"):
                        severity = "LOW"
                    f["severity"] = severity

                    f.setdefault("paper_claim", "")
                    f.setdefault("code_evidence", "")
                    f.setdefault("explanation", "")

                    valid.append(f)
            return valid
    except json.JSONDecodeError:
        pass

    return []


def run_code_to_paper(
    tex_sections: list[dict],
    scripts: list[dict],
    model: str,
    temperature: float,
    api_key: str,
    output_file: Path,
    run_id: str,
    dry_run: bool = False,
) -> list[dict]:
    """Execute Pass 2: Code -> Paper.

    For each experiment script, check if the paper accurately describes it.
    Returns list of finding dicts.
    """
    print(f"\n{'=' * 72}")
    print("PASS 2: CODE -> PAPER")
    print(f"{'=' * 72}")

    # Build full paper text for comparison
    paper_text = "\n\n".join(s["plain"] for s in tex_sections)

    if dry_run:
        print(f"\n[DRY RUN] Would audit {len(scripts)} scripts against paper")
        for s in scripts:
            print(f"  {s['filename']} ({s['line_count']} lines)")
        return []

    # Focus on scripts most relevant to the paper
    # Prioritize: experiment27* (bounded verification), experiment24* (self-report),
    # experiment23* (alignment tax)
    priority_patterns = [
        r"experiment27",
        r"experiment24",
        r"experiment23",
        r"experiment29",
        r"experiment1\b",
    ]

    priority_scripts = []
    other_scripts = []
    for s in scripts:
        is_priority = any(re.search(p, s["filename"]) for p in priority_patterns)
        if is_priority:
            priority_scripts.append(s)
        else:
            other_scripts.append(s)

    # Process priority scripts first, then others
    ordered_scripts = priority_scripts + other_scripts

    print(f"\n  Auditing {len(ordered_scripts)} scripts ({len(priority_scripts)} priority)...")

    findings = []
    finding_counter = 0

    def audit_one_script(script):
        prompt = build_code_audit_prompt(script, paper_text)
        result = call_with_retry(
            model=model,
            temperature=temperature,
            system_prompt=CODE_AUDIT_SYSTEM,
            user_prompt=prompt,
            api_key=api_key,
            max_tokens=8192,
        )
        return script, result

    # Process scripts in parallel (limited concurrency)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as executor:
        futures = {
            executor.submit(audit_one_script, script): script
            for script in ordered_scripts
        }

        for future in as_completed(futures):
            script = futures[future]

            try:
                _, result = future.result()
            except Exception as e:
                print(f"  ERROR auditing {script['filename']}: {e}")
                continue

            if not result["success"]:
                print(f"  ERROR: {script['filename']}: {result['error']}")
                continue

            audit_findings = parse_code_audit_response(result["response_text"])
            print(f"  {script['filename']}: {len(audit_findings)} findings "
                  f"(tokens: {result['prompt_tokens']}+{result['completion_tokens']}, "
                  f"latency: {result['latency_ms']}ms)")

            for af in audit_findings:
                finding_counter += 1
                finding_id = f"C2P-{finding_counter:03d}"

                finding = {
                    "record_type": "finding",
                    "finding_id": finding_id,
                    "pass": "code_to_paper",
                    "severity": af["severity"],
                    "status": af["status"],
                    "paper_claim": af["paper_claim"],
                    "paper_location": "see paper text",
                    "evidence_location": f"{script['filename']}",
                    "evidence": af["code_evidence"],
                    "explanation": af["explanation"],
                }
                findings.append(finding)
                append_jsonl(output_file, finding)

                status_icon = {
                    "CONSISTENT": "+", "INCONSISTENT": "!",
                    "UNDISCLOSED": "~"
                }.get(af["status"], "?")
                severity_tag = f"[{af['severity']}]" if af["severity"] != "LOW" else ""
                print(f"    [{status_icon}] {finding_id} ({af['status']}) "
                      f"{severity_tag}: {af['explanation'][:70]}...")

    return findings


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    p2d_findings: list[dict],
    c2p_findings: list[dict],
):
    """Print a human-readable summary of all findings."""
    print(f"\n{'=' * 72}")
    print("PROVENANCE AUDIT SUMMARY")
    print(f"{'=' * 72}")

    all_findings = p2d_findings + c2p_findings

    if not all_findings:
        print("\nNo findings generated.")
        return

    # --- Pass 1 summary ---
    if p2d_findings:
        print(f"\n--- Pass 1: Paper -> Data ({len(p2d_findings)} claims checked) ---")

        p2d_by_status = {}
        for f in p2d_findings:
            s = f.get("status", "UNKNOWN")
            p2d_by_status[s] = p2d_by_status.get(s, 0) + 1

        for status in ["VERIFIED", "UNVERIFIED", "DISCREPANCY"]:
            count = p2d_by_status.get(status, 0)
            if count > 0:
                print(f"  {status:14s}: {count}")

        # Highlight discrepancies
        discrepancies = [f for f in p2d_findings if f.get("status") == "DISCREPANCY"]
        if discrepancies:
            print(f"\n  DISCREPANCIES ({len(discrepancies)}):")
            for d in discrepancies:
                claim = d.get("paper_claim", "?")
                if len(claim) > 100:
                    claim = claim[:100] + "..."
                print(f"    [{d.get('severity', '?')}] {d.get('finding_id', '?')}: {claim}")
                if d.get("evidence"):
                    evidence = d["evidence"]
                    if len(evidence) > 120:
                        evidence = evidence[:120] + "..."
                    print(f"         Evidence: {evidence}")

    # --- Pass 2 summary ---
    if c2p_findings:
        print(f"\n--- Pass 2: Code -> Paper ({len(c2p_findings)} findings) ---")

        c2p_by_status = {}
        for f in c2p_findings:
            s = f.get("status", "UNKNOWN")
            c2p_by_status[s] = c2p_by_status.get(s, 0) + 1

        for status in ["CONSISTENT", "INCONSISTENT", "UNDISCLOSED"]:
            count = c2p_by_status.get(status, 0)
            if count > 0:
                print(f"  {status:14s}: {count}")

        # Highlight inconsistencies
        inconsistencies = [
            f for f in c2p_findings
            if f.get("status") == "INCONSISTENT"
        ]
        if inconsistencies:
            print(f"\n  INCONSISTENCIES ({len(inconsistencies)}):")
            for inc in inconsistencies:
                claim = inc.get("paper_claim", "?")
                if len(claim) > 100:
                    claim = claim[:100] + "..."
                print(f"    [{inc.get('severity', '?')}] {inc.get('finding_id', '?')}: {claim}")
                if inc.get("evidence"):
                    evidence = inc["evidence"]
                    if len(evidence) > 120:
                        evidence = evidence[:120] + "..."
                    print(f"         Code: {evidence}")

        # Highlight undisclosed
        undisclosed = [
            f for f in c2p_findings
            if f.get("status") == "UNDISCLOSED" and f.get("severity") in ("HIGH", "MEDIUM")
        ]
        if undisclosed:
            print(f"\n  SIGNIFICANT UNDISCLOSED ({len(undisclosed)}):")
            for u in undisclosed:
                evidence = u.get("evidence", "?")
                if len(evidence) > 120:
                    evidence = evidence[:120] + "..."
                print(f"    [{u.get('severity', '?')}] {u.get('finding_id', '?')}: {evidence}")

    # --- Overall ---
    print(f"\n--- Overall ---")

    total = len(all_findings)
    high = sum(1 for f in all_findings if f.get("severity") == "HIGH")
    medium = sum(1 for f in all_findings if f.get("severity") == "MEDIUM")
    low = sum(1 for f in all_findings if f.get("severity") == "LOW")

    print(f"  Total findings: {total}")
    print(f"  Severity: HIGH={high}, MEDIUM={medium}, LOW={low}")

    if high > 0:
        print(f"\n  HIGH-SEVERITY FINDINGS:")
        for f in all_findings:
            if f.get("severity") == "HIGH":
                print(f"    {f.get('finding_id', '?')} ({f.get('pass', '?')}, "
                      f"{f.get('status', '?')}): "
                      f"{f.get('explanation', f.get('evidence', '?'))[:100]}...")

    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_provenance_judge(args) -> Optional[Path]:
    """Execute the provenance judge pipeline.

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

    # --- Resolve directories ---
    paper_dir = Path(args.paper_dir)
    if not paper_dir.is_absolute():
        paper_dir = project_root / paper_dir

    data_dir = Path(args.data_dir) if args.data_dir else project_root
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    scripts_dir = Path(args.scripts_dir) if args.scripts_dir else project_root / "scripts"
    if not scripts_dir.is_absolute():
        scripts_dir = project_root / scripts_dir

    # --- Determine which passes to run ---
    pass_mode = args.pass_mode if hasattr(args, "pass_mode") else "both"

    # --- Load input files ---
    print("=" * 72)
    print("PROVENANCE JUDGE CONFIGURATION")
    print("=" * 72)

    # Load paper
    tex_sections = load_tex_files(paper_dir)
    if not tex_sections:
        print(f"ERROR: No .tex files found in {paper_dir}")
        sys.exit(1)
    print(f"\nPaper sections ({len(tex_sections)}):")
    for s in tex_sections:
        sha_short = s["sha256"][:12]
        print(f"  {s['filename']:30s} ({s['line_count']} lines, "
              f"{s['char_count']} chars, SHA256: {sha_short}...)")

    # Load CSV data (for Pass 1)
    csv_files = []
    if pass_mode in ("paper-to-data", "both"):
        csv_files = load_csv_files(data_dir)
        print(f"\nCSV data files ({len(csv_files)}):")
        for cf in csv_files:
            sha_short = cf["sha256"][:12]
            print(f"  {cf['filename']:50s} ({cf['n_rows']} rows, SHA256: {sha_short}...)")

    # Load scripts (for Pass 2)
    scripts = []
    if pass_mode in ("code-to-paper", "both"):
        scripts = load_script_files(scripts_dir)
        print(f"\nExperiment scripts ({len(scripts)}):")
        for s in scripts:
            sha_short = s["sha256"][:12]
            print(f"  {s['filename']:45s} ({s['line_count']} lines, SHA256: {sha_short}...)")

    # --- Configure model ---
    model = args.model if args.model else DEFAULT_MODEL
    temperature = args.temperature if hasattr(args, "temperature") else DEFAULT_TEMPERATURE

    print(f"\nModel: {model} (temp={temperature})")
    print(f"Pass mode: {pass_mode}")

    # --- Output setup ---
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else project_root / "reviews"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"provenance_{date_str}.jsonl"
    print(f"Output: {output_file}")

    if args.dry_run:
        print(f"\n[DRY RUN] No API calls will be made.")
        if pass_mode in ("paper-to-data", "both"):
            run_paper_to_data(
                tex_sections, csv_files, model, temperature,
                api_key or "", output_file, date_str, dry_run=True
            )
        if pass_mode in ("code-to-paper", "both"):
            run_code_to_paper(
                tex_sections, scripts, model, temperature,
                api_key or "", output_file, date_str, dry_run=True
            )
        return None

    # --- Write provenance record ---
    provenance = {
        "record_type": "provenance",
        "timestamp": now.isoformat(),
        "run_id": date_str,
        "tool": "provenance_judge",
        "pass_mode": pass_mode,
        "model": model,
        "temperature": temperature,
        "paper_sections": [
            {
                "filename": s["filename"],
                "filepath": s["filepath"],
                "line_count": s["line_count"],
                "char_count": s["char_count"],
                "sha256": s["sha256"],
            }
            for s in tex_sections
        ],
        "csv_files": [
            {
                "filename": cf["filename"],
                "filepath": cf["filepath"],
                "n_rows": cf["n_rows"],
                "sha256": cf["sha256"],
            }
            for cf in csv_files
        ],
        "scripts": [
            {
                "filename": s["filename"],
                "filepath": s["filepath"],
                "line_count": s["line_count"],
                "sha256": s["sha256"],
            }
            for s in scripts
        ],
    }
    append_jsonl(output_file, provenance)

    # --- Execute passes ---
    p2d_findings = []
    c2p_findings = []

    if pass_mode in ("paper-to-data", "both"):
        p2d_findings = run_paper_to_data(
            tex_sections, csv_files, model, temperature,
            api_key, output_file, date_str
        )

    if pass_mode in ("code-to-paper", "both"):
        c2p_findings = run_code_to_paper(
            tex_sections, scripts, model, temperature,
            api_key, output_file, date_str
        )

    # --- Write summary record ---
    all_findings = p2d_findings + c2p_findings
    summary_record = {
        "record_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": date_str,
        "pass_mode": pass_mode,
        "p2d_total": len(p2d_findings),
        "p2d_verified": sum(1 for f in p2d_findings if f.get("status") == "VERIFIED"),
        "p2d_unverified": sum(1 for f in p2d_findings if f.get("status") == "UNVERIFIED"),
        "p2d_discrepancy": sum(1 for f in p2d_findings if f.get("status") == "DISCREPANCY"),
        "c2p_total": len(c2p_findings),
        "c2p_consistent": sum(1 for f in c2p_findings if f.get("status") == "CONSISTENT"),
        "c2p_inconsistent": sum(1 for f in c2p_findings if f.get("status") == "INCONSISTENT"),
        "c2p_undisclosed": sum(1 for f in c2p_findings if f.get("status") == "UNDISCLOSED"),
        "total_findings": len(all_findings),
        "total_high": sum(1 for f in all_findings if f.get("severity") == "HIGH"),
        "total_medium": sum(1 for f in all_findings if f.get("severity") == "MEDIUM"),
        "total_low": sum(1 for f in all_findings if f.get("severity") == "LOW"),
    }
    append_jsonl(output_file, summary_record)

    # --- Print summary ---
    print_summary(p2d_findings, c2p_findings)
    print(f"\nResults written to: {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Provenance judge for academic papers. Traces empirical claims "
            "backward to code and data in two non-commutative passes: "
            "Paper->Data (do the numbers match?) and Code->Paper (does the "
            "paper describe what the code does?)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Both passes (default)\n"
            "  python scripts/provenance_judge.py --paper-dir papers/sosp/\n"
            "\n"
            "  # Paper-to-data only\n"
            "  python scripts/provenance_judge.py \\\n"
            "    --paper-dir papers/sosp/ --pass paper-to-data\n"
            "\n"
            "  # Code-to-paper only\n"
            "  python scripts/provenance_judge.py \\\n"
            "    --paper-dir papers/sosp/ --pass code-to-paper\n"
            "\n"
            "  # Custom directories and model\n"
            "  python scripts/provenance_judge.py \\\n"
            "    --paper-dir papers/sosp/ --data-dir . --scripts-dir scripts/ \\\n"
            "    --model anthropic/claude-sonnet-4\n"
            "\n"
            "  # Dry run to preview configuration\n"
            "  python scripts/provenance_judge.py \\\n"
            "    --paper-dir papers/sosp/ --dry-run\n"
        ),
    )
    parser.add_argument(
        "--paper-dir", type=str, required=True,
        help="Directory containing .tex files for the paper.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing CSV data files (default: project root).",
    )
    parser.add_argument(
        "--scripts-dir", type=str, default=None,
        help="Directory containing experiment scripts (default: scripts/).",
    )
    parser.add_argument(
        "--pass", dest="pass_mode", type=str, default="both",
        choices=["paper-to-data", "code-to-paper", "both"],
        help=(
            "Which pass to run: paper-to-data, code-to-paper, or both "
            "(default: both)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: reviews/).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "OpenRouter model ID to use. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Temperature for LLM calls (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration without making API calls.",
    )

    args = parser.parse_args()
    run_provenance_judge(args)


if __name__ == "__main__":
    main()
