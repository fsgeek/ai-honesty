"""Corrected self-report confidence by exp27 query subgroup.

The exp27 CSV labels queries only knowable/unknowable. The finer subgroups
(fictional people, fictional events, future/private, fabricated citations)
exist only as comments in experiment27_bounded_verification.py's
UNKNOWABLE_QUERIES list; this script recovers them from those comments.

Replaces the 0.751 "fabricated-entity" figure quoted in commit 3488f83 and in
the issue #1 reply, which could not be reproduced (Oct 6 2026 audit).
"""

import ast
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "experiment27_bounded_verification.py"
CSV = ROOT / "exp27_bounded_verification_20260206_205725.csv"

HEADERS = {
    "# Fictional people": "people",
    "# Fictional historical": "events",
    "# Future": "future_private",
    "# Fabricated citations": "citations",
}


def subgroup_map():
    lines = SCRIPT.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("UNKNOWABLE_QUERIES"))
    group, mapping = None, {}
    for line in lines[start:]:
        if line.startswith("]"):
            break
        for prefix, name in HEADERS.items():
            if line.strip().startswith(prefix):
                group = name
        m = re.match(r"\s*(\".*?\"|'.*?'),\s*None\)", line.strip().lstrip("("))
        if m and group:
            mapping[ast.literal_eval(m.group(1))] = group
    return mapping


def main():
    df = pd.read_csv(CSV)
    mapping = subgroup_map()
    df["subgroup"] = df["query"].map(mapping)
    df.loc[df.is_knowable, "subgroup"] = "knowable"
    assert df.subgroup.notna().all(), "unmapped unknowable queries"

    conf = "self_report_confidence"
    print(df.groupby("subgroup")[conf].agg(["mean", "count"]).round(3))
    fabricated = df.subgroup.isin(["people", "events", "citations"])
    print(f"\nfabricated entities (people+events+citations): "
          f"{df.loc[fabricated, conf].mean():.3f}  n={fabricated.sum()}")
    print(f"knowable: {df.loc[df.is_knowable, conf].mean():.3f}")


if __name__ == "__main__":
    main()
