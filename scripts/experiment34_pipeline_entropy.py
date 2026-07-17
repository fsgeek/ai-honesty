#!/usr/bin/env python3
"""Experiment 34 — Pipeline entropy: does post-training manufacture conviction?

Tests whether token-level entropy *while fabricating* decreases monotonically
across the OLMo-3 post-training pipeline:

    base  ->  SFT  ->  DPO  ->  Instruct (RLVR)

Hypothesis (H1): entropy over fabricated CONTENT tokens falls monotonically
    base -> Instruct. The pipeline manufactures conviction, not just fluency.
Null (H0): entropy is flat across stages; only surface form changed.

This is the rerun specified after the Jan-2026 audit found that captured
per-token traces existed only for the Instruct endpoint (exp27c), which cannot
distinguish H1 from H0 — a single pipeline point has no slope.

Design (per Fable's spec):
  * Same 200 prompts as exp27c (imported, not copied — identical-prompt trend
    test). 100 knowable controls + 100 unknowable (fabrication-prone).
  * All four checkpoints.
  * Greedy + one sampled run (temp=1.0, fixed seed) each.
  * Per token: entropy, chosen-token logprob, top-1 probability, top-5 mass/ids.
  * Optional full-logit dump (--dump-logits): fp16 [num_tokens, vocab] sidecar.
    ~48 GB total at fp16 over all stages/runs — trivial against 40 TB.
  * Per-token disclaimer labels: HEDGE/REFUSAL marker char-spans mapped to token
    indices, so entropy over CONTENT tokens can be separated from hedge tokens
    and disclaimer placement correlated with local entropy.

Usage:
    python scripts/experiment34_pipeline_entropy.py                  # greedy+sampled, all stages
    python scripts/experiment34_pipeline_entropy.py --dump-logits    # also dump full logits
    python scripts/experiment34_pipeline_entropy.py --stages base sft # subset
    python scripts/experiment34_pipeline_entropy.py --analyze-only run_dir  # re-run analysis

Outputs (under exp34_<timestamp>/):
    traces_<stage>_<mode>.jsonl   per-token records (entropy/logprob/top1/disclaimer)
    logits/<stage>_<mode>/q<NNN>.npz   full logits, if --dump-logits
    exp34_summary.csv             per-stage content/disclaimer entropy summary
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime

import numpy as np

# Heavy deps (torch, transformers) and the exp27c probe set are imported lazily
# inside the functions that need them, so the pure-logic paths — the disclaimer
# labeler and `--analyze-only` — run without a GPU stack installed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE_SEED = 1994  # fixed so the sampled run is reproducible


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _probe_set():
    """The EXACT exp27c probe set + markers (byte-identical prompts across stages)."""
    from experiment27c_full_traces import (
        KNOWABLE_QUERIES, UNKNOWABLE_QUERIES, HEDGE_MARKERS, REFUSAL_MARKERS,
        SYSTEM_PROMPT, is_citation_query,
    )
    return {
        "knowable": KNOWABLE_QUERIES, "unknowable": UNKNOWABLE_QUERIES,
        "markers": HEDGE_MARKERS + REFUSAL_MARKERS, "system": SYSTEM_PROMPT,
        "is_citation": is_citation_query,
    }

# ============================================================================
# OLMo-3 post-training pipeline checkpoints
# Lineage confirmed from the Olmo-3-7B-Instruct model card (HF, Jun 2026):
#   base -> Instruct-SFT -> Instruct-DPO -> Instruct (RLVR final).
# ============================================================================
PIPELINE = [
    ("base",     "allenai/Olmo-3-1025-7B"),        # pretrained base, no chat template
    ("sft",      "allenai/Olmo-3-7B-Instruct-SFT"),
    ("dpo",      "allenai/Olmo-3-7B-Instruct-DPO"),
    ("instruct", "allenai/Olmo-3-7B-Instruct"),     # RLVR final
]
STAGE_ORDER = {name: i for i, (name, _) in enumerate(PIPELINE)}


# ============================================================================
# Prompt formatting (chat template if the checkpoint has one, else completion)
# ============================================================================

def format_prompt(tokenizer, query, system_prompt):
    """Chat template for SFT/DPO/Instruct; bare completion framing for base.

    The base checkpoint has no chat template — forcing one would inject tokens
    the base model never saw in training. We detect this and fall back to a
    minimal completion prompt so the base stage is a fair pipeline anchor.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ), "chat"
    # Base completion framing: a single Q/A turn, no special tokens.
    return f"Question: {query}\nAnswer:", "completion"


# ============================================================================
# Disclaimer labeling: marker char-spans -> token indices
# ============================================================================

def label_disclaimer_tokens(token_texts, markers):
    """Map HEDGE/REFUSAL marker occurrences to the token indices they cover.

    Reconstructs the response from token_texts, finds each marker's character
    span (case-insensitive), then flags every token whose char range overlaps a
    marker. Robust to subword tokenization, where a marker phrase rarely aligns
    to token boundaries.

    `markers` is the combined HEDGE + REFUSAL list (passed in to keep this a
    pure, dependency-free function).

    Returns (is_disclaimer[list[bool]], disclaimer_positions[list[int]],
             matches[list[dict]]).
    """
    text = ""
    spans = []  # (char_start, char_end) for each token
    for t in token_texts:
        spans.append((len(text), len(text) + len(t)))
        text += t
    low = text.lower()

    is_disc = [False] * len(token_texts)
    matches = []
    for marker in markers:
        m = marker.lower()
        start = 0
        while True:
            idx = low.find(m, start)
            if idx < 0:
                break
            end = idx + len(m)
            covered = [ti for ti, (s, e) in enumerate(spans) if s < end and e > idx]
            for ti in covered:
                is_disc[ti] = True
            matches.append({
                "marker": marker,
                "char_start": idx,
                "char_end": end,
                "token_start": covered[0] if covered else None,
                "token_end": covered[-1] if covered else None,
            })
            start = idx + 1
    disc_positions = [i for i, b in enumerate(is_disc) if b]
    return is_disc, disc_positions, matches


# ============================================================================
# Generation with full per-token capture
# ============================================================================

def generate_capture(model, tokenizer, prompt, mode, markers, max_tokens=150,
                     dump_logits=False):
    """Generate and capture per-token entropy/logprob/top-1/top-5 (+ optional logits).

    mode: "greedy" (do_sample=False) or "sampled" (temp=1.0, fixed seed).
    Returns (response_text, record_dict, logits_array_or_None).
    """
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(prompt, return_tensors="pt").to(_device())

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if mode == "sampled":
        torch.manual_seed(SAMPLE_SEED)
        gen_kwargs.update(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    scores = outputs.scores
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]

    token_entropies, logprobs, top1_probs = [], [], []
    top5_masses, token_ids, token_texts = [], [], []
    logit_rows = [] if dump_logits else None

    for score, token_id in zip(scores, generated_ids):
        logits = score.squeeze(0).float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        token_entropies.append(-torch.sum(probs * log_probs).item())
        top5 = torch.topk(probs, k=min(5, probs.shape[-1]))
        top5_masses.append(top5.values.sum().item())
        top1_probs.append(top5.values[0].item())
        logprobs.append(log_probs[token_id].item())
        token_ids.append(int(token_id.item()))
        token_texts.append(tokenizer.decode([int(token_id.item())]))
        if dump_logits:
            logit_rows.append(logits.half().cpu().numpy())

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    is_disc, disc_positions, disc_matches = label_disclaimer_tokens(token_texts, markers)

    record = {
        "query": prompt,  # overwritten with the raw query by caller
        "response": response,
        "mode": mode,
        "num_tokens": len(token_ids),
        "token_entropies": token_entropies,
        "token_logprobs": logprobs,
        "token_top1_probs": top1_probs,
        "token_top5_masses": top5_masses,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "is_disclaimer": is_disc,
        "disclaimer_positions": disc_positions,
        "disclaimer_matches": disc_matches,
    }
    logits_array = np.stack(logit_rows) if dump_logits and logit_rows else None
    return response, record, logits_array


# ============================================================================
# Per-checkpoint collection
# ============================================================================

def collect_stage(stage, model_id, modes, out_dir, dump_logits):
    """Run all 200 prompts through one checkpoint in each requested mode."""
    print(f"\n{'='*70}\nStage: {stage}  ({model_id})\n{'='*70}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ps = _probe_set()
    markers, system_prompt, is_citation_query = (
        ps["markers"], ps["system"], ps["is_citation"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )

    probes = (
        [(q, exp, "knowable", True) for q, exp in ps["knowable"]]
        + [(q, exp, "unknowable", False) for q, exp in ps["unknowable"]]
    )

    for mode in modes:
        trace_path = os.path.join(out_dir, f"traces_{stage}_{mode}.jsonl")
        logit_dir = os.path.join(out_dir, "logits", f"{stage}_{mode}")
        if dump_logits:
            os.makedirs(logit_dir, exist_ok=True)

        with open(trace_path, "w") as tf:
            for qi, (query, expected, category, is_knowable) in enumerate(probes):
                prompt, prompt_kind = format_prompt(tokenizer, query, system_prompt)
                _, rec, logits = generate_capture(
                    model, tokenizer, prompt, mode, markers,
                    dump_logits=dump_logits,
                )
                rec.update({
                    "stage": stage,
                    "model_id": model_id,
                    "stage_order": STAGE_ORDER[stage],
                    "query": query,            # raw query, not the templated prompt
                    "prompt_kind": prompt_kind,
                    "expected_answer": expected,
                    "category": category,
                    "is_knowable": is_knowable,
                    "is_citation": is_citation_query(query),
                    "query_index": qi,
                })
                if dump_logits and logits is not None:
                    lp = os.path.join(logit_dir, f"q{qi:03d}.npz")
                    np.savez_compressed(lp, logits=logits)
                    rec["logits_path"] = os.path.relpath(lp, out_dir)
                tf.write(json.dumps(rec) + "\n")
                tf.flush()
                if qi % 25 == 0:
                    print(f"  [{mode}] {qi+1}/{len(probes)}  {query[:48]}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Analysis: per-stage entropy distribution + monotonic trend test
# ============================================================================

def analyze(out_dir):
    """Summarize content/disclaimer entropy per stage and test the trend."""
    import glob
    from scipy.stats import spearmanr

    rows = []  # per (stage, mode): content-token entropies over fabrications
    per_stage = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "traces_*.jsonl"))):
        for line in open(path):
            d = json.loads(line)
            if d["mode"] != "greedy":
                continue  # trend reported on the canonical greedy trajectory
            ents = d["token_entropies"]
            disc = d["is_disclaimer"]
            content = [e for e, dd in zip(ents, disc) if not dd]
            hedge = [e for e, dd in zip(ents, disc) if dd]
            key = d["stage"]
            s = per_stage.setdefault(key, {
                "order": d["stage_order"],
                "fab_content": [], "fab_hedge": [], "knowable_content": [],
            })
            if d["is_knowable"]:
                s["knowable_content"] += content
            else:
                s["fab_content"] += content
                s["fab_hedge"] += hedge

    # Fabrication content-token entropy is bimodal (long near-zero "conviction"
    # runs + sparse high "what-to-invent" spikes), so the mean is spike-dominated
    # and a weak H1 detector. We report three statistics; conviction_frac (share
    # of content tokens committed at near-zero entropy) most directly
    # operationalizes "manufactured conviction".
    CONVICTION_H = 0.1  # entropy threshold for a "committed" token

    def conviction_frac(xs):
        return float(np.mean([x < CONVICTION_H for x in xs])) if xs else float("nan")

    print(f"\n{'='*86}")
    print("Per-stage fabrication-content entropy (greedy). Fabrication = unknowable probes.")
    print(f"{'stage':10s} {'mean_H':>8s} {'median_H':>9s} {'convict_frac':>13s} "
          f"{'hedge_H':>9s} {'knowable_med':>13s} {'n_content':>10s}")
    ordered = sorted(per_stage.items(), key=lambda kv: kv[1]["order"])
    # H1 is tested on each statistic; conviction_frac is the primary endpoint.
    trends = {"mean": ([], []), "median": ([], []), "conviction_frac": ([], [])}
    for stage, s in ordered:
        fc = s["fab_content"]
        stats = {
            "mean": float(np.mean(fc)) if fc else float("nan"),
            "median": float(np.median(fc)) if fc else float("nan"),
            "conviction_frac": conviction_frac(fc),
        }
        mean_fh = float(np.mean(s["fab_hedge"])) if s["fab_hedge"] else float("nan")
        med_kn = float(np.median(s["knowable_content"])) if s["knowable_content"] else float("nan")
        print(f"{stage:10s} {stats['mean']:8.3f} {stats['median']:9.3f} "
              f"{stats['conviction_frac']:13.3f} {mean_fh:9.3f} {med_kn:13.3f} {len(fc):10d}")
        rows.append({
            "stage": stage, "stage_order": s["order"],
            "fab_content_mean_entropy": stats["mean"],
            "fab_content_median_entropy": stats["median"],
            "fab_content_conviction_frac": stats["conviction_frac"],
            "fab_hedge_mean_entropy": mean_fh,
            "knowable_content_median_entropy": med_kn,
            "n_fab_content_tokens": len(fc),
        })
        if fc:
            for k in trends:
                trends[k][0].append(s["order"])
                trends[k][1].append(stats[k])

    # Monotonic-trend test per statistic. For entropy (mean/median) H1 predicts
    # rho << 0 (falls base->Instruct); for conviction_frac H1 predicts rho >> 0
    # (commitment rises base->Instruct).
    n_stages = len(trends["mean"][0])
    if n_stages >= 3:
        print(f"\nMonotonic-trend tests (Spearman vs stage order, {n_stages} stages):")
        for k, (x, y) in trends.items():
            rho, p = spearmanr(x, y)
            direction = "rises (H1: +)" if k == "conviction_frac" else "falls (H1: -)"
            print(f"  {k:16s} rho={rho:+.3f}  p={p:.3f}   [{direction}]")
        print("H0 (only surface form changed): all rho ~ 0.")
    else:
        print(f"\n(Need >=3 stages with data for the trend test; have {n_stages}.)")

    import csv
    csv_path = os.path.join(out_dir, "exp34_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=[s for s, _ in PIPELINE],
                    choices=[s for s, _ in PIPELINE],
                    help="Subset of pipeline stages to run.")
    ap.add_argument("--modes", nargs="+", default=["greedy", "sampled"],
                    choices=["greedy", "sampled"])
    ap.add_argument("--dump-logits", action="store_true",
                    help="Also dump full fp16 logits per token (~48 GB total).")
    ap.add_argument("--analyze-only", metavar="RUN_DIR", default=None,
                    help="Skip generation; (re)run analysis on an existing run dir.")
    args = ap.parse_args()

    if args.analyze_only:
        analyze(args.analyze_only)
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"exp34_{stamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Stages: {args.stages}  Modes: {args.modes}  dump_logits={args.dump_logits}")

    id_by_stage = dict(PIPELINE)
    for stage in args.stages:
        try:
            collect_stage(stage, id_by_stage[stage], args.modes, out_dir,
                          args.dump_logits)
        except Exception as e:
            print(f"!! Stage {stage} failed: {e}")

    analyze(out_dir)


if __name__ == "__main__":
    main()
