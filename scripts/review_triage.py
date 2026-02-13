#!/usr/bin/env python3
"""Review triage helper — record post-run decisions.

After reviewing pipeline output, this script creates a structured triage
record linking findings to actions. The triage records are tensor-ready:
each captures authored decisions about what to act on, what to defer, and
what to dismiss — the meta-cognitive layer that would otherwise exist only
in conversation context and be lost to compaction.

Usage:
    # Create triage for the most recent run
    python scripts/review_triage.py

    # Create triage for a specific run
    python scripts/review_triage.py --run-id 20260213_040142

    # Add a finding
    python scripts/review_triage.py --add-finding \
        --source hostile_rejecter \
        --claim "Theorem is intuitively obvious" \
        --category 2 \
        --action pending \
        --note "Recurring. Options A-E presented."

    # Add an edit provenance record
    python scripts/review_triage.py --add-edit \
        --description "Fixed text baseline numbers" \
        --files eval.tex abstract.tex intro.tex \
        --motivated-by "hostile_rejecter:text baseline includes self-report"

    # Mark a finding as resolved
    python scripts/review_triage.py --resolve \
        --finding-id headline_auc_misread \
        --resolution "C+D reframing"

    # Show current triage status
    python scripts/review_triage.py --status
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_reviews_dir() -> Path:
    project_root = Path(__file__).parent.parent
    return project_root / "reviews"


def load_state(reviews_dir: Path) -> dict:
    state_file = reviews_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def get_latest_run_id(reviews_dir: Path) -> str:
    state = load_state(reviews_dir)
    return state.get("last_run_id", "")


def load_triage(reviews_dir: Path, run_id: str) -> dict:
    triage_file = reviews_dir / "triage" / f"triage_{run_id}.json"
    if triage_file.exists():
        with open(triage_file) as f:
            return json.load(f)
    return {
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "findings": [],
        "edits_applied": [],
        "resolved_from_prev": [],
    }


def save_triage(reviews_dir: Path, run_id: str, triage: dict):
    triage_dir = reviews_dir / "triage"
    triage_dir.mkdir(exist_ok=True)
    triage_file = triage_dir / f"triage_{run_id}.json"
    triage["last_modified"] = datetime.now(timezone.utc).isoformat()
    with open(triage_file, "w") as f:
        json.dump(triage, f, indent=2)
        f.write("\n")
    print(f"Triage saved: {triage_file}")


def cmd_add_finding(args, reviews_dir: Path, run_id: str):
    triage = load_triage(reviews_dir, run_id)
    finding = {
        "id": f"f{len(triage['findings']) + 1:03d}",
        "source_persona": args.source,
        "claim": args.claim,
        "category": args.category,
        "action": args.action,  # pending, fix_planned, dismissed, deferred
        "persists_from": args.persists_from,
        "note": args.note,
        "added": datetime.now(timezone.utc).isoformat(),
    }
    triage["findings"].append(finding)
    save_triage(reviews_dir, run_id, triage)
    print(f"  Added finding {finding['id']}: [{args.category}] {args.claim[:60]}...")


def cmd_add_edit(args, reviews_dir: Path, run_id: str):
    triage = load_triage(reviews_dir, run_id)
    edit = {
        "description": args.description,
        "files_changed": args.files,
        "motivated_by": args.motivated_by,
        "added": datetime.now(timezone.utc).isoformat(),
    }
    triage["edits_applied"].append(edit)
    save_triage(reviews_dir, run_id, triage)
    print(f"  Added edit: {args.description[:60]}...")


def cmd_resolve(args, reviews_dir: Path, run_id: str):
    triage = load_triage(reviews_dir, run_id)
    resolution = {
        "finding_id": args.finding_id,
        "from_run": args.from_run or "unknown",
        "resolution": args.resolution,
        "resolved": datetime.now(timezone.utc).isoformat(),
    }
    triage["resolved_from_prev"].append(resolution)
    save_triage(reviews_dir, run_id, triage)
    print(f"  Resolved: {args.finding_id} — {args.resolution[:60]}...")


def cmd_status(reviews_dir: Path):
    """Show triage status across all runs."""
    triage_dir = reviews_dir / "triage"
    if not triage_dir.exists():
        print("No triage records found.")
        return

    triage_files = sorted(triage_dir.glob("triage_*.json"))
    if not triage_files:
        print("No triage records found.")
        return

    # Collect all findings across runs
    all_pending = []
    all_resolved = []
    total_findings = 0
    total_edits = 0

    for tf in triage_files:
        with open(tf) as f:
            triage = json.load(f)
        run_id = triage.get("run_id", tf.stem.replace("triage_", ""))

        for finding in triage.get("findings", []):
            total_findings += 1
            if finding.get("action") in ("pending", "deferred"):
                all_pending.append((run_id, finding))

        for resolution in triage.get("resolved_from_prev", []):
            all_resolved.append((run_id, resolution))

        total_edits += len(triage.get("edits_applied", []))

    print(f"Triage status across {len(triage_files)} runs:")
    print(f"  Total findings recorded: {total_findings}")
    print(f"  Total edits recorded: {total_edits}")
    print(f"  Resolved: {len(all_resolved)}")
    print(f"  Pending/deferred: {len(all_pending)}")

    if all_pending:
        print(f"\nOutstanding findings:")
        for run_id, finding in all_pending:
            cat = finding.get("category", "?")
            src = finding.get("source_persona", "?")
            claim = finding.get("claim", "?")[:70]
            persists = finding.get("persists_from", "")
            persist_str = f" (persists from {persists})" if persists else ""
            print(f"  [{cat}] {src}: {claim}{persist_str}")


def main():
    parser = argparse.ArgumentParser(description="Review triage helper")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: most recent)")

    sub = parser.add_mutually_exclusive_group()
    sub.add_argument("--add-finding", action="store_true")
    sub.add_argument("--add-edit", action="store_true")
    sub.add_argument("--resolve", action="store_true")
    sub.add_argument("--status", action="store_true")

    # Finding args
    parser.add_argument("--source", type=str, help="Source persona")
    parser.add_argument("--claim", type=str, help="The claim/finding")
    parser.add_argument("--category", type=int, choices=[1, 2, 3],
                        help="1=mechanical, 2=decision, 3=noise")
    parser.add_argument("--action", type=str, default="pending",
                        choices=["pending", "fix_planned", "dismissed", "deferred"],
                        help="Action taken")
    parser.add_argument("--persists-from", type=str, default=None,
                        help="Run ID where this finding first appeared")
    parser.add_argument("--note", type=str, default="", help="Notes")

    # Edit args
    parser.add_argument("--description", type=str, help="Edit description")
    parser.add_argument("--files", nargs="+", help="Files changed")
    parser.add_argument("--motivated-by", type=str,
                        help="Finding that motivated this edit")

    # Resolve args
    parser.add_argument("--finding-id", type=str, help="Finding ID to resolve")
    parser.add_argument("--from-run", type=str, help="Run where finding originated")
    parser.add_argument("--resolution", type=str, help="How it was resolved")

    args = parser.parse_args()

    reviews_dir = get_reviews_dir()
    run_id = args.run_id or get_latest_run_id(reviews_dir)

    if not run_id and not args.status:
        print("ERROR: No run ID found. Run the pipeline first or specify --run-id.")
        sys.exit(1)

    if args.status:
        cmd_status(reviews_dir)
    elif args.add_finding:
        if not args.source or not args.claim or not args.category:
            print("ERROR: --add-finding requires --source, --claim, --category")
            sys.exit(1)
        cmd_add_finding(args, reviews_dir, run_id)
    elif args.add_edit:
        if not args.description:
            print("ERROR: --add-edit requires --description")
            sys.exit(1)
        cmd_add_edit(args, reviews_dir, run_id)
    elif args.resolve:
        if not args.finding_id or not args.resolution:
            print("ERROR: --resolve requires --finding-id and --resolution")
            sys.exit(1)
        cmd_resolve(args, reviews_dir, run_id)
    else:
        # Default: show status
        cmd_status(reviews_dir)


if __name__ == "__main__":
    main()
