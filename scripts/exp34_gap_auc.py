#!/usr/bin/env python3
"""Exp34: per-stage knowable/unknowable separation (gap + AUC) and the
within-stage disclaimer cross-tab. Reads exp34_traces_<ts>.jsonl.
Knowable = control + wombat; unknowable = glavinsky + westphalia + private
(the exp24/27 convention). Statistic = mean entropy over 'fabrication'-class
tokens (content tokens; disclaimer phrases and boundaries excluded)."""
import gzip, json, sys
import numpy as np

STAGES = ["base", "sft", "dpo", "rlvr"]
KNOW = {"control", "wombat"}; UNK = {"glavinsky", "westphalia", "private"}
FAB = {"glavinsky", "westphalia"}

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return np.nan
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq

def load(path, cap):
    out = []
    for line in (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)):
        try: r = json.loads(line)
        except Exception: break
        if len(r["token_classes"]) >= cap: continue
        e = [t["entropy"] for t, c in zip(r["per_token"], r["token_classes"]) if c == "fabrication"]
        if not e: continue
        out.append(dict(stage=r["stage"], decode=r["decode"], cat=r["category"],
                        ent=float(np.mean(e)), disc="disclaimer" in r["token_classes"]))
    return out

def main(path, cap=2048):
    rows = load(path, cap)
    for decode in ["greedy", "sampled"]:
        print(f"\n== {decode} ==  (mean content-token entropy; AUC = unknowable ranked above knowable)")
        print(f"{'stage':5} {'control':>8} {'wombat':>7} {'fab':>6} {'private':>8} {'gap(fab-ctl)':>13} {'AUC k/u':>8} {'AUC ctl/fab':>11}")
        for st in STAGES:
            R = [r for r in rows if r["stage"] == st and r["decode"] == decode]
            if not R: continue
            m = lambda cats: np.mean([r["ent"] for r in R if r["cat"] in cats]) if any(r["cat"] in cats for r in R) else np.nan
            a1 = auc([r["ent"] for r in R if r["cat"] in UNK], [r["ent"] for r in R if r["cat"] in KNOW])
            a2 = auc([r["ent"] for r in R if r["cat"] in FAB], [r["ent"] for r in R if r["cat"] == "control"])
            print(f"{st:5} {m({'control'}):8.3f} {m({'wombat'}):7.3f} {m(FAB):6.3f} {m({'private'}):8.3f} "
                  f"{m(FAB)-m({'control'}):13.3f} {a1:8.3f} {a2:11.3f}")
        print("  disclaimer cross-tab on fabrication probes (glavinsky+westphalia): entropy of CONTENT tokens")
        for st in STAGES:
            R = [r for r in rows if r["stage"] == st and r["decode"] == decode and r["cat"] in FAB]
            if not R: continue
            d = [r["ent"] for r in R if r["disc"]]; n = [r["ent"] for r in R if not r["disc"]]
            print(f"   {st:5} with-disclaimer n={len(d):2} ent={np.mean(d) if d else np.nan:.3f} | "
                  f"without n={len(n):2} ent={np.mean(n) if n else np.nan:.3f}")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2048)
