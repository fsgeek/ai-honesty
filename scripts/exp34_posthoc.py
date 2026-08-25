#!/usr/bin/env python3
"""Post-hoc analysis for Experiment 34 (pipeline conviction).

Reads an exp34_summary_<ts>.csv and reports, per stage x decode:
  - n fabrication-probe responses, mean/median fabrication-span entropy
  - disclaimer rate
  - response-length distribution and count of runaway outliers
  - Spearman rho(stage order, mean fab entropy) -- H1 predicts rho < 0
plus a LENGTH-CONTROLLED variant restricted to responses whose token count
falls in a band shared by all stages (guards against the entropy-length
correlation, r~0.7 project-wide; see memory exp34-length-confound).

Usage: python scripts/exp34_posthoc.py exp34_summary_<ts>.csv [--max-len N]
"""
import argparse
import numpy as np
import pandas as pd

STAGES = ["base", "sft", "dpo", "rlvr"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def table(df, label):
    print(f"\n== {label} ==")
    rows = []
    for decode in ["greedy", "sampled"]:
        for st in STAGES:
            sub = df[(df.stage == st) & (df.decode == decode)]
            fab = sub.fab_entropy_mean.dropna()
            rows.append(dict(
                decode=decode, stage=st, n=len(sub),
                fab_ent_mean=fab.mean(), fab_ent_median=fab.median(),
                disclaimer_rate=sub.loc_disclaimer_present.astype(bool).mean() if len(sub) else np.nan,
                len_median=sub.n_tokens.median(), len_max=sub.n_tokens.max(),
            ))
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))
    for decode in ["greedy", "sampled"]:
        sub = t[t.decode == decode].dropna(subset=["fab_ent_mean"])
        if len(sub) >= 3:
            idx = [STAGES.index(s) for s in sub.stage]
            print(f"  Spearman rho(stage, fab_ent_mean) [{decode}] = "
                  f"{spearman(idx, sub.fab_ent_mean):+.2f}   "
                  f"rho(stage, disclaimer_rate) = {spearman(idx, sub.disclaimer_rate):+.2f}")



# ----------------------------------------------------------------------------
# Trace-based span controls (need exp34_traces_<ts>.jsonl).
# Addresses the compositional confound: post-trained stages emit markdown
# scaffolding (**, ###, bullets) and hedging connectives that a base completion
# model does not; those are near-deterministic tokens that lower the mean for
# reasons unrelated to conviction about the fabricated content.
#   (a) first-K content tokens: position-matched spans across stages
#   (b) formatting filter: drop tokens with no alphanumeric character
# ----------------------------------------------------------------------------
import gzip
import json
import re

_ALNUM = re.compile(r"[A-Za-z0-9]")


def trace_controls(trace_path, k=30, cap=None):
    per = {}  # (stage, decode) -> list of (mean_all, mean_firstK, mean_nofmt)
    n_cap = 0
    for line in (gzip.open(trace_path, "rt") if str(trace_path).endswith(".gz") else open(trace_path)):
        try:
            r = json.loads(line)
        except Exception:
            break
        if r["category"] not in ("glavinsky", "westphalia"):
            continue
        toks = r["per_token"]
        if cap and len(toks) >= cap:
            n_cap += 1
            continue
        fab = [t for t, c in zip(toks, r["token_classes"]) if c == "fabrication"]
        if not fab:
            continue
        e_all = np.mean([t["entropy"] for t in fab])
        e_k = np.mean([t["entropy"] for t in fab[:k]])
        nf = [t["entropy"] for t in fab if _ALNUM.search(t["token_text"])]
        e_nf = np.mean(nf) if nf else np.nan
        per.setdefault((r["stage"], r["decode"]), []).append((e_all, e_k, e_nf, len(fab)))
    print(f"\n== TRACE SPAN CONTROLS (fabrication probes; first-K={k}; "
          f"{n_cap} cap-hits excluded) ==")
    print(f"{'decode':7} {'stage':5} {'n':>3} {'all':>6} {'firstK':>7} {'nofmt':>6} {'fab_len_med':>11}")
    for decode in ["greedy", "sampled"]:
        idx, ka, kk, kn = [], [], [], []
        for st in STAGES:
            v = per.get((st, decode))
            if not v:
                continue
            a = np.array(v, dtype=float)
            print(f"{decode:7} {st:5} {len(v):3} {a[:,0].mean():6.3f} {a[:,1].mean():7.3f} "
                  f"{np.nanmean(a[:,2]):6.3f} {np.median(a[:,3]):11.0f}")
            idx.append(STAGES.index(st)); ka.append(a[:,0].mean()); kk.append(a[:,1].mean()); kn.append(np.nanmean(a[:,2]))
        if len(idx) >= 3:
            print(f"  rho(stage, ·) [{decode}]: all={spearman(idx,ka):+.2f} "
                  f"firstK={spearman(idx,kk):+.2f} nofmt={spearman(idx,kn):+.2f}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("trace", nargs="?", help="optional exp34_traces_<ts>.jsonl for span controls")
    ap.add_argument("--max-len", type=int, default=None,
                    help="length band ceiling for the controlled variant "
                         "(default: min over stages of the 90th pct length)")
    ap.add_argument("--outlier", type=int, default=1000)
    a = ap.parse_args()

    df = pd.read_csv(a.summary)
    fabp = df[df.is_fabrication_probe.astype(str).str.lower() == "true"].copy()
    print(f"{a.summary}: {len(df)} rows, {len(fabp)} fabrication-probe rows, "
          f"categories={sorted(fabp.category.unique())}")

    if "hit_cap" in df.columns:
        out = df[df.hit_cap.astype(str).str.lower() == "true"]
        what = "hit the generation cap"
    else:
        out = df[df.n_tokens >= a.outlier]
        what = f"n_tokens >= {a.outlier} (no hit_cap column; legacy run)"
    print(f"\nRunaway responses ({what}): {len(out)} -- EXCLUDED below")
    if len(out):
        print(out[["stage", "decode", "probe_id", "category", "n_tokens"]].to_string(index=False))
    df = df.drop(out.index)

    table(fabp, "ALL fabrication-probe responses")

    cap = a.max_len or int(fabp.groupby("stage").n_tokens.quantile(0.9).min())
    ctrl = fabp[fabp.n_tokens <= cap]
    table(ctrl, f"LENGTH-CONTROLLED (n_tokens <= {cap}; "
                f"{len(ctrl)}/{len(fabp)} rows retained)")

    # Knowable baseline for reference: same stat on control probes.
    ctl = df[df.category == "control"]
    if len(ctl):
        table(ctl, "CONTROL (knowable) probes, for reference")

    if a.trace:
        cap = int(df["max_tokens"].max()) if "max_tokens" in df.columns else None
        trace_controls(a.trace, cap=cap)


if __name__ == "__main__":
    main()
