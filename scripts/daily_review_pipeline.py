#!/usr/bin/env python3
"""Daily paper review pipeline with tensor-ready provenance capture.

Sends the current paper to randomly-selected OpenRouter models with
diverse reviewer personas and temperatures. Captures ALL output —
full reviews, model metadata, paper version hash, timestamps.

Heartbeat model: designed to run daily via cron. Each run produces:
  - review JSONL (raw testimony from bounded supervisors)
  - manifest entry (paper state + run metadata — the spine)
  - state update (cross-run tracking)

Usage:
    # Full daily run (3 reviewers + 2 scourers)
    python scripts/daily_review_pipeline.py

    # Quick test (1 reviewer only)
    python scripts/daily_review_pipeline.py --quick

    # Custom reviewer count
    python scripts/daily_review_pipeline.py --reviewers 2 --scourers 1

    # Specific output directory
    python scripts/daily_review_pipeline.py --output-dir reviews/

Environment:
    OPENROUTER_API_KEY: Required. Set in environment or in .env file
    in the project root.

Output:
    reviews/
      state.json                          # Heartbeat state
      manifest.jsonl                      # One entry per run — the spine
      review_YYYYMMDD_HHMMSS.jsonl        # Raw reviews per run
      triage/                             # Post-run decisions (written by triage helper)
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration: model pool
# ---------------------------------------------------------------------------
# Models with sufficient context (>=32K) for a full paper + review prompt.
# Update this list as models become available/unavailable on OpenRouter.
# Large models (70B+ or frontier) — used for reviewer personas where depth matters.
LARGE_MODEL_POOL = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-pro-preview",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2411",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat-v3-0324",
    "cohere/command-r-plus-08-2024",
]

# All models including smaller ones — used for scourer personas where
# surface-level pattern detection benefits from diversity over depth.
ALL_MODEL_POOL = LARGE_MODEL_POOL + [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "google/gemma-2-27b-it",
]

# Maximum retries for empty or failed responses
MAX_RETRIES = 1

# Temperature range. Uniform random from this interval.
TEMP_MIN = 0.0
TEMP_MAX = 2.0

# ---------------------------------------------------------------------------
# Reviewer personas
# ---------------------------------------------------------------------------
REVIEWER_PERSONAS = [
    {
        "name": "shepherd",
        "type": "reviewer",
        "system": (
            "You are Reviewer #1 for SOSP 2026, an expert systems researcher who has "
            "published at SOSP, OSDI, and EuroSys. You are a potential shepherd for this "
            "paper. You want to see it succeed but ONLY if it meets the bar. You care "
            "deeply about rigor, clarity, and whether the contribution is real. You will "
            "push back on overclaiming, weak evaluation, and missing baselines. But you "
            "will also identify what is genuinely novel and worth developing."
        ),
        "prompt": (
            "Review this paper for SOSP 2026. Provide:\n"
            "1. Summary (2-3 sentences: what the paper claims)\n"
            "2. Strengths (bullet points, specific)\n"
            "3. Weaknesses (bullet points, specific, with page/section references)\n"
            "4. Questions for the authors (things that would change your assessment)\n"
            "5. Missing related work\n"
            "6. Overall assessment: Accept / Weak Accept / Weak Reject / Reject\n"
            "7. Confidence: 1 (low) to 5 (high)\n\n"
            "Be specific. Quote the paper when identifying problems. Do not summarize "
            "at a high level — point to exact claims and evaluate them."
        ),
    },
    {
        "name": "hostile_rejecter",
        "type": "reviewer",
        "system": (
            "You are Reviewer #2 for SOSP 2026. You are skeptical that this paper belongs "
            "at a systems venue. Your default stance is rejection — you are looking for "
            "reasons to reject, not reasons to accept. You will attack: venue fit, theorem "
            "novelty, evaluation methodology, missing baselines, and overclaiming. You "
            "give credit where due but your bar is very high."
        ),
        "prompt": (
            "Review this paper for SOSP 2026. Your default recommendation is Reject. "
            "Provide:\n"
            "1. Summary (2-3 sentences)\n"
            "2. Reasons to reject (numbered, specific, with quotes from the paper)\n"
            "3. Reasons NOT to reject (what, if anything, is genuinely valuable)\n"
            "4. Missing baselines or comparisons\n"
            "5. Would this alone cause rejection? (for each weakness)\n"
            "6. Overall assessment: Accept / Weak Accept / Weak Reject / Reject\n"
            "7. Confidence: 1 (low) to 5 (high)\n\n"
            "Be ruthless but fair. If the paper has real merit, acknowledge it. But do "
            "not give the benefit of the doubt on unsupported claims."
        ),
    },
    {
        "name": "non_domain_expert",
        "type": "reviewer",
        "system": (
            "You are Reviewer #3 for SOSP 2026. You are a systems researcher with strong "
            "distributed systems and OS background, but NO machine learning expertise. "
            "You will reject the paper if you cannot follow the argument. You flag: "
            "undefined terms, logical jumps, unclear notation, claims without evidence, "
            "and sections that assume ML knowledge a systems reader would not have."
        ),
        "prompt": (
            "Review this paper for SOSP 2026 as a non-ML expert. Provide:\n"
            "1. Summary (in your own words — what did you understand the paper claims?)\n"
            "2. Where you got confused (specific locations, what was unclear)\n"
            "3. Undefined or poorly defined terms\n"
            "4. Logical gaps (where does the argument skip a step?)\n"
            "5. What would help you understand (missing explanations, examples, figures)\n"
            "6. Does the paper earn its systems-venue placement?\n"
            "7. Overall assessment: Accept / Weak Accept / Weak Reject / Reject\n"
            "8. Confidence: 1 (low) to 5 (high)\n\n"
            "If you cannot follow the argument, say so. Do not pretend to understand "
            "something you do not."
        ),
    },
]

SCOURER_PERSONAS = [
    {
        "name": "open_ended_scourer",
        "type": "scourer",
        "system": (
            "You are a research assistant with expertise in both systems and ML. Your job "
            "is to examine a paper draft and report what you find. You have NO hypothesis "
            "and NO agenda. Look at the data, the claims, the structure, and report "
            "anything interesting — inconsistencies, patterns, strengths, weaknesses, "
            "missing connections, opportunities."
        ),
        "prompt": (
            "Examine this paper and report what you find. Do not limit yourself to any "
            "particular lens. Look at:\n"
            "- Numbers: do they add up? Are they consistent across sections?\n"
            "- Claims: are they supported by the evidence presented?\n"
            "- Structure: does the argument flow? Are there orphaned concepts?\n"
            "- Rhetoric: where does the paper overclaim or underclaim?\n"
            "- Missing: what is conspicuously absent?\n"
            "- Surprising: what did you not expect to find?\n\n"
            "Report everything. Do not filter for importance — we will triage later."
        ),
    },
    {
        "name": "adversarial_scourer",
        "type": "scourer",
        "system": (
            "You are a devil's advocate. Your job is to find the weakest points in this "
            "paper — the claims most likely to be wrong, the evidence most likely to be "
            "misleading, the arguments most likely to fail under scrutiny. You are not "
            "hostile; you are thorough. You want the paper to be strong, and you achieve "
            "that by finding where it is weak."
        ),
        "prompt": (
            "Find the weakest points in this paper. For each weakness:\n"
            "1. State the claim\n"
            "2. Quote the relevant text\n"
            "3. Explain why it might be wrong or misleading\n"
            "4. Suggest what evidence would address the weakness\n\n"
            "Prioritize: which weaknesses would cause a knowledgeable reviewer to lose "
            "confidence in the paper? Focus on those."
        ),
    },
]


# ---------------------------------------------------------------------------
# Paper metadata
# ---------------------------------------------------------------------------

def get_paper_section_lines(paper_dir: str) -> dict:
    """Count lines per .tex section file."""
    counts = {}
    for f in sorted(Path(paper_dir).glob("*.tex")):
        name = f.stem
        counts[name] = sum(1 for _ in f.open())
    return counts


def get_paper_pages(paper_dir: str) -> int:
    """Get page count from compiled PDF."""
    pdf_path = Path(paper_dir) / "epistemic_honest.pdf"
    if not pdf_path.exists():
        return -1
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return -1


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_git_hash(repo_dir: str) -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_dirty(repo_dir: str) -> bool:
    """Check if working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        )
        return len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        return True


def get_git_diff_summary(repo_dir: str, since_hash: str) -> str:
    """Get a one-line summary of changes since a given commit."""
    if not since_hash or since_hash == "unknown":
        return ""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", since_hash, "HEAD", "--",
             "papers/sosp/"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        )
        lines = result.stdout.strip().splitlines()
        return lines[-1].strip() if lines else ""
    except subprocess.CalledProcessError:
        return ""


# ---------------------------------------------------------------------------
# State & manifest
# ---------------------------------------------------------------------------

def load_state(output_dir: Path) -> dict:
    """Load heartbeat state, creating defaults if absent."""
    state_file = output_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "last_run_id": None,
        "last_run_timestamp": None,
        "last_paper_sha256": None,
        "last_paper_git_hash": None,
        "run_count": 0,
        "total_reviews": 0,
        "total_tokens": 0,
    }


def save_state(output_dir: Path, state: dict):
    """Persist heartbeat state."""
    state_file = output_dir / "state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def append_manifest(output_dir: Path, entry: dict):
    """Append a manifest entry."""
    manifest_file = output_dir / "manifest.jsonl"
    with open(manifest_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Paper text collection
# ---------------------------------------------------------------------------

def collect_paper_text(paper_dir: str) -> str:
    """Read all .tex files in order and concatenate."""
    # Read the main file to get include order
    main_file = os.path.join(paper_dir, "epistemic_honest.tex")
    if not os.path.exists(main_file):
        # Fallback: read all .tex files
        tex_files = sorted(Path(paper_dir).glob("*.tex"))
        parts = []
        for f in tex_files:
            parts.append(f"% === {f.name} ===\n{f.read_text()}")
        return "\n\n".join(parts)

    # Parse \input commands from main file
    main_text = Path(main_file).read_text()
    sections = []

    # Add preamble
    sections.append(f"% === epistemic_honest.tex (preamble) ===\n{main_text}")

    # Extract \input{...} files in order
    for match in re.finditer(r"\\input\{(\w+)\}", main_text):
        fname = match.group(1) + ".tex"
        fpath = os.path.join(paper_dir, fname)
        if os.path.exists(fpath):
            sections.append(f"% === {fname} ===\n{Path(fpath).read_text()}")

    return "\n\n".join(sections)


def sha256_text(text: str) -> str:
    """SHA256 hash of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def call_openrouter(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> dict:
    """Call OpenRouter API and return full response dict."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fsgeek/ai-honesty",
        "X-Title": "AI Honesty Paper Review Pipeline",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 16384,  # Allow long reviews — never truncate
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


def load_env_file(project_root: Path):
    """Load .env file if it exists."""
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Execute the daily review pipeline."""
    project_root = Path(__file__).parent.parent
    load_env_file(project_root)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found.")
        print("Set it in .env or as an environment variable.")
        print("Get a key at https://openrouter.ai/keys")
        sys.exit(1)

    paper_dir = project_root / "papers" / "sosp"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "reviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "triage").mkdir(exist_ok=True)

    # Load heartbeat state
    state = load_state(output_dir)

    # Collect paper
    print("Collecting paper text...")
    paper_text = collect_paper_text(str(paper_dir))
    paper_hash = sha256_text(paper_text)
    git_hash = get_git_hash(str(project_root))
    git_dirty = get_git_dirty(str(project_root))
    section_lines = get_paper_section_lines(str(paper_dir))
    paper_pages = get_paper_pages(str(paper_dir))
    print(f"  Paper: {len(paper_text)} chars, SHA256: {paper_hash[:16]}...")
    print(f"  Git: {git_hash[:12]}{'*' if git_dirty else ''}")
    print(f"  Pages: {paper_pages}, Lines: {sum(section_lines.values())}")

    # Diff from previous run
    diff_summary = ""
    if state.get("last_paper_git_hash"):
        diff_summary = get_git_diff_summary(
            str(project_root), state["last_paper_git_hash"]
        )
    paper_changed = paper_hash != state.get("last_paper_sha256")

    # Select models and temperatures
    n_reviewers = args.reviewers
    n_scourers = args.scourers
    total = n_reviewers + n_scourers

    # Reviewers get large models (depth matters); scourers get any model (diversity matters)
    if len(LARGE_MODEL_POOL) >= n_reviewers:
        reviewer_models = random.sample(LARGE_MODEL_POOL, n_reviewers)
    else:
        reviewer_models = [random.choice(LARGE_MODEL_POOL) for _ in range(n_reviewers)]

    if len(ALL_MODEL_POOL) >= n_scourers:
        scourer_models = random.sample(ALL_MODEL_POOL, n_scourers)
    else:
        scourer_models = [random.choice(ALL_MODEL_POOL) for _ in range(n_scourers)]

    selected_models = reviewer_models + scourer_models

    # Assign personas
    reviewer_personas = random.sample(REVIEWER_PERSONAS, min(n_reviewers, len(REVIEWER_PERSONAS)))
    while len(reviewer_personas) < n_reviewers:
        reviewer_personas.append(random.choice(REVIEWER_PERSONAS))

    scourer_personas = random.sample(SCOURER_PERSONAS, min(n_scourers, len(SCOURER_PERSONAS)))
    while len(scourer_personas) < n_scourers:
        scourer_personas.append(random.choice(SCOURER_PERSONAS))

    all_personas = reviewer_personas + scourer_personas

    # Assign temperatures
    temperatures = [round(random.uniform(TEMP_MIN, TEMP_MAX), 2) for _ in range(total)]

    # Output file
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")
    run_number = state["run_count"] + 1
    output_file = output_dir / f"review_{date_str}.jsonl"

    print(f"\nPipeline configuration:")
    print(f"  Run #{run_number} (prev: {state.get('last_run_id', 'none')})")
    print(f"  Paper changed: {paper_changed}")
    print(f"  Reviewers: {n_reviewers}, Scourers: {n_scourers}")
    print(f"  Output: {output_file}")
    print()

    # Run each review
    results = []
    for i, (model, persona, temp) in enumerate(zip(selected_models, all_personas, temperatures)):
        print(f"[{i+1}/{total}] {persona['name']} via {model} (temp={temp:.2f})...")

        user_prompt = f"{persona['prompt']}\n\n---\n\nBEGIN PAPER\n\n{paper_text}\n\nEND PAPER"

        result = call_openrouter(
            model=model,
            temperature=temp,
            system_prompt=persona["system"],
            user_prompt=user_prompt,
            api_key=api_key,
        )

        # Retry on empty or failed responses with a different model
        retries = 0
        original_model = model
        while retries < MAX_RETRIES and (
            not result["success"] or len(result.get("response_text", "")) == 0
        ):
            retries += 1
            pool = LARGE_MODEL_POOL if persona["type"] == "reviewer" else ALL_MODEL_POOL
            fallback_models = [m for m in pool if m != model]
            if fallback_models:
                model = random.choice(fallback_models)
            reason = "empty response" if result["success"] else result.get("error", "unknown")
            print(f"  RETRY {retries}/{MAX_RETRIES}: {reason} — trying {model}...")
            time.sleep(2)
            result = call_openrouter(
                model=model,
                temperature=temp,
                system_prompt=persona["system"],
                user_prompt=user_prompt,
                api_key=api_key,
            )

        record = {
            "date": now.isoformat(),
            "run_id": date_str,
            "run_number": run_number,
            "review_index": i,
            "paper_git_hash": git_hash,
            "paper_git_dirty": git_dirty,
            "paper_sha256": paper_hash,
            "paper_char_count": len(paper_text),
            "model_id": model,
            "model_id_returned": result.get("model_id_returned", model),
            "temperature": temp,
            "persona_name": persona["name"],
            "persona_type": persona["type"],
            "system_prompt": persona["system"],
            "user_prompt_template": persona["prompt"],  # template only, not the full paper
            "response_text": result["response_text"],
            "success": result["success"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "finish_reason": result.get("finish_reason", ""),
        }

        results.append(record)

        # Write incrementally (one line per review, append mode)
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        if result["success"]:
            resp_len = len(result["response_text"])
            print(f"  OK: {resp_len} chars, {result['completion_tokens']} tokens, "
                  f"{result['latency_ms']}ms")
        else:
            print(f"  ERROR: {result['error']}")

        # Small delay between API calls to be polite
        if i < total - 1:
            time.sleep(2)

    # Compute run totals
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in results)
    n_success = sum(1 for r in results if r["success"])
    n_failed = sum(1 for r in results if not r["success"])
    total_latency_ms = sum(r["latency_ms"] for r in results)

    # Summary
    print(f"\n{'='*70}")
    print(f"Pipeline complete: {output_file}")
    print(f"  Total reviews: {len(results)}")
    print(f"  Successful: {n_success}")
    print(f"  Failed: {n_failed}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total latency: {total_latency_ms / 1000:.1f}s")

    # Print model/persona/temp summary
    print(f"\nReview assignments:")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] {r['persona_name']:<25s} {r['model_id']:<45s} temp={r['temperature']:.2f}")

    # --- Write manifest entry ---
    models_used = list({r["model_id"] for r in results})
    manifest_entry = {
        "run_id": date_str,
        "run_number": run_number,
        "timestamp": now.isoformat(),
        "paper_git_hash": git_hash,
        "paper_git_dirty": git_dirty,
        "paper_sha256": paper_hash,
        "paper_char_count": len(paper_text),
        "paper_pages": paper_pages,
        "paper_section_lines": section_lines,
        "paper_total_lines": sum(section_lines.values()),
        "paper_changed_since_prev": paper_changed,
        "diff_summary": diff_summary,
        "prev_run_id": state.get("last_run_id"),
        "reviews_file": output_file.name,
        "review_count": len(results),
        "review_success": n_success,
        "review_failed": n_failed,
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency_ms,
        "models_used": sorted(models_used),
        "personas_used": [r["persona_name"] for r in results],
        "temperatures": [r["temperature"] for r in results],
    }
    append_manifest(output_dir, manifest_entry)
    print(f"\nManifest entry appended: reviews/manifest.jsonl")

    # --- Update state ---
    state["last_run_id"] = date_str
    state["last_run_timestamp"] = now.isoformat()
    state["last_paper_sha256"] = paper_hash
    state["last_paper_git_hash"] = git_hash
    state["run_count"] = run_number
    state["total_reviews"] = state.get("total_reviews", 0) + len(results)
    state["total_tokens"] = state.get("total_tokens", 0) + total_tokens
    save_state(output_dir, state)
    print(f"State updated: reviews/state.json (run #{run_number})")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Daily paper review pipeline using OpenRouter models",
    )
    parser.add_argument("--reviewers", type=int, default=3,
                        help="Number of reviewer agents (default: 3)")
    parser.add_argument("--scourers", type=int, default=2,
                        help="Number of scourer agents (default: 2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: reviews/)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 reviewer, 0 scourers")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.quick:
        args.reviewers = 1
        args.scourers = 0

    if args.seed is not None:
        random.seed(args.seed)

    run_pipeline(args)


if __name__ == "__main__":
    main()
