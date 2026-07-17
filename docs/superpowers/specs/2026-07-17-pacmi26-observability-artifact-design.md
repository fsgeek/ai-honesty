# pacmi26-observability — Clean Artifact Design

**Date:** 2026-07-17
**Author:** Claude (ayllu member) + Tony Mason
**Target repo:** `../pacmi26-observability` (`git@github.com:fsgeek/pacmi26-observability.git`), Zenodo-tracked
**Source repo:** `ai-honesty` — full research record, DOI `10.5281/zenodo.21314137`, commit `9799e22` (v1.0.0)

## Purpose

Build a minimal, legible, self-contained public artifact for the PACMI 26 submission
that (a) a reviewer or reader can navigate in one sitting, and (b) still answers the
"is this cherry-picked?" question by shipping the actual state-object interface and
regenerating every figure from committed data. It supersedes neither the paper nor
the full record; it curates from the full record and points back to it by DOI.

This resolves a disagreement between two legitimate goals:
- **Legibility** (Vaastav): a clean front door; no mystery CSVs, WIP notes, or process cruft.
- **Verifiability** (Tony): nothing hidden; the interface is present and re-runnable so
  no one can claim the results were curated to flatter.

The resolution is *organize, don't amputate*: this repo is the legible front door; the
full record (by DOI) is the complete evidentiary trail. The chain between them is
explicit and bidirectional.

## Settled decisions

| Decision | Choice | Rationale |
|---|---|---|
| Reproduction depth | **Two-tier** | `make figures` regenerates all 3 figures from committed data on a laptop (no weights); `make traces` re-runs the interface against the models to regenerate that data. |
| `src/` scope | **Minimal** | Only the state-object extraction the 3 figures exercise. One auditable unit. |
| Canonical noun | **"state object"** (thing) + **"observability"** (framing/repo name) | "state object" claims exactly what is exposed and answers reviewers' objection to "tensor"; "observability" stays as the goal/framing word. "tensor" retired in this repo. |
| Budget-curve provenance | **Regenerate from seeded sim** | `experiment27_realistic_verification.py` is the real Monte Carlo source (seed, 1000 trials, budgets 10/20/30). No config fallback needed. |
| Spec location | `ai-honesty/docs/…` (here) | A design/process doc is process cruft; it must not land in the clean artifact. |

## The three figures (the artifact's entire visible output)

1. **`fig:entropy_trace`** (paper `design.tex`) — wombat-scat per-token entropy trace,
   rendered as `\etok{bin}{token}` LaTeX (macro defined in `epistemic_honest.tex:52`).
   Source: a committed per-token entropy trace.
2. **`exp27_confidence_distributions.pdf`** (paper `background.tex`) — self-report
   confidence, knowable vs. unknowable, 4 model families. Source: `regenerate_confidence_distributions.py`
   reads `exp27_bounded_verification_20260206_205725.csv`.
3. **`exp27_aggregate_budget_curve.pdf`** (paper `eval.tex`) — verification accuracy vs.
   budget for No-judge / text / state-object / composed. Source: seeded Monte Carlo in
   `experiment27_realistic_verification.py`.

## Repository layout

```
pacmi26-observability/
  README.md            # what it is · quickstart · provenance pointer · terminology note
  PROVENANCE.md        # curated-from DOI 10.5281/zenodo.21314137 @ 9799e22; what's in / out
  CITATION.cff         # cites the full-record DOI
  LICENSE              # MIT (already present)
  pyproject.toml       # uv; deps: torch, transformers, matplotlib, numpy, pandas
  Makefile             # `make figures` (laptop) · `make traces` (needs weights)
  src/
    observability/
      __init__.py      # exports StateObject / StateObservation
      interface.py     # lifted + renamed from scripts/tensor_interface.py, minimal
  scripts/
    fig1_entropy_trace.py            # data/entropy_trace_wombat.json -> \etok LaTeX (NEW, ~30 lines)
    fig2_confidence_distributions.py # from regenerate_confidence_distributions.py
    fig3_budget_curve.py             # seeded sim over committed exp27 data -> summary -> plot
    generate_trace.py                # OPTIONAL, needs weights: run interface on wombat query -> JSON
  data/
    exp27_bounded_verification_20260206_205725.csv   # feeds fig2 + fig3 sim
    entropy_trace_wombat.json                         # feeds fig1
  figures/             # output dir (regenerated); final PDFs committed for convenience
```

## Component design

### `src/observability/interface.py` (the one unit)
Lifted from `scripts/tensor_interface.py` and reduced to what the figures need: run a
model forward/generation and return a `StateObservation` carrying `text`,
`entropy_trace` (per-token), `attention_summary`, `mean_entropy`, `mean_logprob`,
`top5_mass`. Rename `TensorInterface` → `StateObjectInterface`, `TensorResult` →
`StateObservation`. Remove any methods the 3 figures don't exercise (attention
geometry / topological / multi-metric paths stay in the full record only).

- **What it does:** query → state observation (telemetry of the actual forward pass).
- **How you use it:** `StateObjectInterface(model_id).observe(prompt)`.
- **Depends on:** `torch`, `transformers`. Needs weights only when actually run.

### Figure scripts (thin drivers, 1:1 with figures)
Each is a pure function of `data/` + (for fig3) a fixed seed. No network, no weights
under `make figures`. `fig3` regenerates the budget numbers via the seeded sim and, at
build time, is **diffed against the paper's committed values (81.7/86.7/90.2…)**; any
mismatch is surfaced as a provenance finding, not silently accepted.

### `generate_trace.py` (optional, tier 2)
Runs `StateObjectInterface` on the wombat query to (re)produce `entropy_trace_wombat.json`.
Needs weights + HF access. Documented in README as the "regenerate the data itself" path.

## Data flow

`make figures`: `data/*` → figure scripts → `figures/*.pdf` (laptop, deterministic).
`make traces`: models → `StateObjectInterface` → `data/*` (needs weights) → then `make figures`.

## Provenance & release

- `PROVENANCE.md` + `CITATION.cff` + `README.md` state: "curated from the full research
  record, DOI `10.5281/zenodo.21314137`, commit `9799e22`," listing what was included
  (the 3 figures' code + data + the interface) and what was intentionally excluded
  (exploratory experiments, reviews, notes, WIP drafts).
- **Reciprocal pointer:** add one line to the *full* repo's README pointing to this
  artifact's DOI, so the chain is walkable in both directions.
- All commits GPG-signed (config already mirrored by Tony).
- Tony cuts the tagged release; Zenodo mints the new **canonical** DOI for the artifact.
- Terminology follow-on (NOT actioned here): recommend the paper move
  "tensor interface" → "state-object interface" for paper/artifact coherence — Tony's call.
- **Hamut'ay DOI: intentionally NOT cited in the artifact** (2026-07-17). It neither feeds
  nor derives from this package and postdates the results it freezes; citing it here would
  dilute the provenance chain and re-introduce an anachronism. Its correct home is the
  paper's future-work prose ("this direction continues"), if anywhere — a separate decision.

## Verification

- `make figures` on a clean checkout (no weights) regenerates all 3 figures.
- fig2/fig3 outputs visually match the submitted PDFs; fig3 numbers match the paper
  (or the mismatch is reported).
- `src/observability` imports and runs on the committed trace without a GPU for the
  laptop path.
- The uv-init scaffold (`main.py`, default `README.md`, `.python-version`) is removed.

## Out of scope (YAGNI)

- No broader observability module (attention geometry, topology, multi-metric).
- No experiments beyond exp27 + the single entropy trace.
- No paper edits.
- No CI, no packaging to PyPI, no docs site.

## Open risks

1. **fig3 number drift** — the plotted constants may not equal the sim's current output.
   Mitigation: diff and reconcile at build; report if they differ.
2. **Interface entanglement** — `tensor_interface.py` may pull in heavier deps than the
   figures need. Mitigation: trim imports to the minimal path; keep the laptop path GPU-free.
3. ~~**`entropy_trace_wombat.json` may not exist**~~ **RESOLVED (2026-07-17):** the wombat
   trace is already committed in `epistemic_trace_demo_20260208_184656.json` (and a newer
   `_20260316_061220.json`) as `knowable_weird` → "What shape is wombat scat?" with full
   per-token data. Figure 1 needs **no GPU run**; extract the matching trace into
   `data/entropy_trace_wombat.json`. Pick whichever timestamp matches the paper's committed
   `\etok` text (`epistemic_trace_latex.tex`); diff at build. Consequence: **all three
   figures regenerate on the GPU-free `make figures` path** — verified this box has no CUDA
   yet the laptop path needs none. The 4090 is not in the critical path.
