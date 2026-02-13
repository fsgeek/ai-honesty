#!/usr/bin/env python3
"""Analytics on daily review pipeline output using DuckDB.

Loads all review JSONL files from the reviews/ directory into DuckDB
and provides various analytical queries.

Usage:
    # Summary of all reviews
    python scripts/review_analytics.py summary

    # Show finding frequency across runs
    python scripts/review_analytics.py findings

    # Compare assessments across models
    python scripts/review_analytics.py models

    # Show review for a specific run
    python scripts/review_analytics.py show --run-id 20260212_150000_0

    # Export to CSV for external analysis
    python scripts/review_analytics.py export --output reviews/analysis.csv

    # Full text of all reviews for a given date
    python scripts/review_analytics.py day --date 2026-02-12
"""

import argparse
import json
import os
import sys
from pathlib import Path

import duckdb


def load_reviews(db: duckdb.DuckDBPyConnection, reviews_dir: str) -> int:
    """Load all JSONL review files into DuckDB."""
    reviews_path = Path(reviews_dir)
    if not reviews_path.exists():
        print(f"No reviews directory found at {reviews_path}")
        return 0

    jsonl_files = sorted(reviews_path.glob("review_*.jsonl"))
    if not jsonl_files:
        print(f"No review files found in {reviews_path}")
        return 0

    # Create table
    db.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            date TIMESTAMP,
            run_id VARCHAR,
            paper_git_hash VARCHAR,
            paper_git_dirty BOOLEAN,
            paper_sha256 VARCHAR,
            paper_char_count INTEGER,
            model_id VARCHAR,
            model_id_returned VARCHAR,
            temperature DOUBLE,
            persona_name VARCHAR,
            persona_type VARCHAR,
            system_prompt VARCHAR,
            user_prompt_template VARCHAR,
            response_text VARCHAR,
            success BOOLEAN,
            error VARCHAR,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            finish_reason VARCHAR
        )
    """)

    total = 0
    for jf in jsonl_files:
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                db.execute("""
                    INSERT INTO reviews VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, [
                    record.get("date"),
                    record.get("run_id"),
                    record.get("paper_git_hash"),
                    record.get("paper_git_dirty"),
                    record.get("paper_sha256"),
                    record.get("paper_char_count"),
                    record.get("model_id"),
                    record.get("model_id_returned"),
                    record.get("temperature"),
                    record.get("persona_name"),
                    record.get("persona_type"),
                    record.get("system_prompt"),
                    record.get("user_prompt_template"),
                    record.get("response_text"),
                    record.get("success"),
                    record.get("error"),
                    record.get("latency_ms"),
                    record.get("prompt_tokens"),
                    record.get("completion_tokens"),
                    record.get("finish_reason"),
                ])
                total += 1

    return total


def cmd_summary(db: duckdb.DuckDBPyConnection):
    """Print summary statistics."""
    print("=" * 70)
    print("REVIEW PIPELINE SUMMARY")
    print("=" * 70)

    result = db.execute("""
        SELECT
            COUNT(*) as total_reviews,
            COUNT(DISTINCT paper_sha256) as paper_versions,
            COUNT(DISTINCT model_id) as distinct_models,
            COUNT(DISTINCT persona_name) as distinct_personas,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
            SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
            MIN(date) as first_review,
            MAX(date) as last_review,
            SUM(prompt_tokens + completion_tokens) as total_tokens,
            AVG(completion_tokens) as avg_response_tokens,
            AVG(latency_ms) as avg_latency_ms
        FROM reviews
    """).fetchone()

    labels = [
        "Total reviews", "Paper versions", "Distinct models",
        "Distinct personas", "Successful", "Failed",
        "First review", "Last review", "Total tokens",
        "Avg response tokens", "Avg latency (ms)",
    ]
    for label, val in zip(labels, result):
        if isinstance(val, float):
            print(f"  {label:<25s}: {val:.0f}")
        else:
            print(f"  {label:<25s}: {val}")

    # Per-persona breakdown
    print(f"\n{'='*70}")
    print("PER-PERSONA BREAKDOWN")
    print(f"{'='*70}")
    rows = db.execute("""
        SELECT
            persona_name,
            persona_type,
            COUNT(*) as n,
            AVG(completion_tokens) as avg_tokens,
            AVG(temperature) as avg_temp,
            COUNT(DISTINCT model_id) as models_used
        FROM reviews
        WHERE success
        GROUP BY persona_name, persona_type
        ORDER BY persona_type, persona_name
    """).fetchall()

    print(f"  {'Persona':<25s} {'Type':<10s} {'N':>4s} {'AvgTok':>8s} {'AvgTemp':>8s} {'Models':>7s}")
    print(f"  {'-'*25} {'-'*10} {'-'*4} {'-'*8} {'-'*8} {'-'*7}")
    for row in rows:
        print(f"  {row[0]:<25s} {row[1]:<10s} {row[2]:>4d} {row[3]:>8.0f} {row[4]:>8.2f} {row[5]:>7d}")

    # Per-model breakdown
    print(f"\n{'='*70}")
    print("PER-MODEL BREAKDOWN")
    print(f"{'='*70}")
    rows = db.execute("""
        SELECT
            model_id,
            COUNT(*) as n,
            AVG(completion_tokens) as avg_tokens,
            AVG(latency_ms) as avg_latency,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as ok,
            SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as fail
        FROM reviews
        GROUP BY model_id
        ORDER BY n DESC
    """).fetchall()

    print(f"  {'Model':<45s} {'N':>4s} {'AvgTok':>8s} {'AvgMs':>8s} {'OK':>4s} {'Fail':>5s}")
    print(f"  {'-'*45} {'-'*4} {'-'*8} {'-'*8} {'-'*4} {'-'*5}")
    for row in rows:
        print(f"  {row[0]:<45s} {row[1]:>4d} {row[2]:>8.0f} {row[3]:>8.0f} {row[4]:>4d} {row[5]:>5d}")

    # Paper version history
    print(f"\n{'='*70}")
    print("PAPER VERSION HISTORY")
    print(f"{'='*70}")
    rows = db.execute("""
        SELECT
            paper_sha256,
            paper_git_hash,
            MIN(date) as first_seen,
            COUNT(*) as n_reviews
        FROM reviews
        GROUP BY paper_sha256, paper_git_hash
        ORDER BY first_seen
    """).fetchall()

    for row in rows:
        print(f"  {row[0][:16]}... git:{row[1][:12]} first:{row[2]} reviews:{row[3]}")


def cmd_day(db: duckdb.DuckDBPyConnection, date_str: str):
    """Show all reviews for a specific date."""
    rows = db.execute("""
        SELECT run_id, persona_name, model_id, temperature, response_text,
               completion_tokens, latency_ms
        FROM reviews
        WHERE CAST(date AS DATE) = CAST(? AS DATE) AND success
        ORDER BY run_id
    """, [date_str]).fetchall()

    if not rows:
        print(f"No reviews found for date {date_str}")
        return

    for row in rows:
        print(f"\n{'='*70}")
        print(f"Run: {row[0]}")
        print(f"Persona: {row[1]} | Model: {row[2]} | Temp: {row[3]:.2f}")
        print(f"Tokens: {row[4]} | Latency: {row[5]}ms")
        print(f"{'='*70}")
        print(row[4])  # Full response text — never truncated


def cmd_show(db: duckdb.DuckDBPyConnection, run_id: str):
    """Show a specific review by run_id."""
    row = db.execute("""
        SELECT * FROM reviews WHERE run_id = ?
    """, [run_id]).fetchone()

    if not row:
        print(f"No review found with run_id {run_id}")
        return

    cols = [desc[0] for desc in db.description]
    for col, val in zip(cols, row):
        if col == "response_text":
            print(f"\n{'='*70}")
            print("RESPONSE TEXT:")
            print(f"{'='*70}")
            print(val)
        elif col in ("system_prompt", "user_prompt_template"):
            print(f"\n{col}: ({len(val)} chars)")
        else:
            print(f"  {col}: {val}")


def cmd_export(db: duckdb.DuckDBPyConnection, output: str):
    """Export reviews to CSV (without full text, for spreadsheet analysis)."""
    db.execute(f"""
        COPY (
            SELECT
                date, run_id, paper_git_hash, paper_sha256,
                model_id, temperature, persona_name, persona_type,
                success, latency_ms, prompt_tokens, completion_tokens,
                LENGTH(response_text) as response_chars
            FROM reviews
            ORDER BY date
        ) TO '{output}' (HEADER, DELIMITER ',')
    """)
    print(f"Exported to {output}")


def cmd_temperature(db: duckdb.DuckDBPyConnection):
    """Analyze relationship between temperature and review characteristics."""
    print("=" * 70)
    print("TEMPERATURE ANALYSIS")
    print("=" * 70)

    rows = db.execute("""
        SELECT
            CASE
                WHEN temperature < 0.5 THEN '0.0-0.5'
                WHEN temperature < 1.0 THEN '0.5-1.0'
                WHEN temperature < 1.5 THEN '1.0-1.5'
                ELSE '1.5-2.0'
            END as temp_bucket,
            COUNT(*) as n,
            AVG(completion_tokens) as avg_tokens,
            AVG(LENGTH(response_text)) as avg_chars,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as ok,
            SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as fail
        FROM reviews
        GROUP BY temp_bucket
        ORDER BY temp_bucket
    """).fetchall()

    print(f"  {'Temp Range':<12s} {'N':>4s} {'AvgTok':>8s} {'AvgChars':>10s} {'OK':>4s} {'Fail':>5s}")
    print(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*10} {'-'*4} {'-'*5}")
    for row in rows:
        print(f"  {row[0]:<12s} {row[1]:>4d} {row[2]:>8.0f} {row[3]:>10.0f} {row[4]:>4d} {row[5]:>5d}")


def main():
    parser = argparse.ArgumentParser(description="Review pipeline analytics")
    parser.add_argument("command", choices=["summary", "day", "show", "export",
                                            "temperature", "models"],
                        help="Analysis command")
    parser.add_argument("--reviews-dir", default=None,
                        help="Reviews directory (default: reviews/)")
    parser.add_argument("--date", help="Date for 'day' command (YYYY-MM-DD)")
    parser.add_argument("--run-id", help="Run ID for 'show' command")
    parser.add_argument("--output", help="Output file for 'export' command")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    reviews_dir = args.reviews_dir or str(project_root / "reviews")

    # Create in-memory DuckDB and load reviews
    db = duckdb.connect(":memory:")
    n = load_reviews(db, reviews_dir)
    if n == 0:
        print("No reviews loaded. Run daily_review_pipeline.py first.")
        sys.exit(1)
    print(f"Loaded {n} reviews.\n")

    if args.command == "summary":
        cmd_summary(db)
    elif args.command == "day":
        if not args.date:
            print("--date required for 'day' command")
            sys.exit(1)
        cmd_day(db, args.date)
    elif args.command == "show":
        if not args.run_id:
            print("--run-id required for 'show' command")
            sys.exit(1)
        cmd_show(db, args.run_id)
    elif args.command == "export":
        output = args.output or str(project_root / "reviews" / "analysis.csv")
        cmd_export(db, output)
    elif args.command == "temperature":
        cmd_temperature(db)
    elif args.command == "models":
        # Reuse summary's per-model section
        cmd_summary(db)

    db.close()


if __name__ == "__main__":
    main()
