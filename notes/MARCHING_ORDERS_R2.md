# ai-honesty: Marching Orders Round 2 (Rikuy Reviews 2026-03-16)

**Reviews:**
- SOSP Supplement: `~/projects/rikuy/reviews/sosp-supplement/review_20260316_131201.jsonl` — 217 findings (3F, 74M, 140m)
- arXiv Version: `~/projects/rikuy/reviews/ai-honesty-arxiv/review_20260316_130959.jsonl` — 281 findings (2F, 100M, 179m)

**Context:** The instance fixed supplement Task 1 issues (Lean excerpts, TLA+ framing) and revised the arXiv version in the same pass. These reviews evaluate the result. The two reviews converge on a common theme.

---

## The Core Problem: Formal Apparatus Oversells What It Proves

Both reviews, independently, arrive at the same diagnosis: the paper conflates formal model consistency with empirical claims about real LLMs. This shows up in 4 of the 5 fatals:

- Supplement FATAL 1: TLA+ tensor escape spec is tautological (model constructed to have desired property, then TLC "verifies" it)
- Supplement FATAL 2: Lean proofs prove properties of abstract definitions, not properties of LLMs
- arXiv FATAL 1: Lean proof of Theorem 2 may be vacuous — substance potentially hidden in type-level axioms
- arXiv FATAL 2: Theorem 2 proof skips the critical step (never invokes Condition 2)

**The irony is not lost on anyone:** a paper about epistemic honesty is being flagged for overstating what its formal proofs establish. This is fixable, and fixing it well makes the paper a stronger example of its own thesis.

### Action: Epistemic Status Labels

Every formal result needs a clear epistemic status label. Three categories:

1. **Conditional formal result:** "Given our model's definitions, X follows by logical necessity." (This is what the Lean proofs and TLA+ specs actually establish.)
2. **Empirical claim:** "We observe X across N models under conditions Y." (This is what Exp 27/27b establishes.)
3. **Informal argument:** "We argue that X because Y, but this is not formally proven." (This is the bridge between 1 and 2.)

Specific fixes:

**TLA+ sections (both docs):** Add: "TLC verifies the Verifiability invariant *within our formal model*, where the axiom that fabrications produce incoherent topology is encoded by construction. The empirical question — whether real tensor signals satisfy this axiom — is addressed in Section [X]."

**Lean sections (both docs):** Add: "These proofs verify that our conclusions follow logically from our formal definitions. The substantive claim is that those definitions faithfully model real LLM behavior, which we argue informally in Section [X] and evaluate empirically in Section [Y]."

**Theorem 2 proof (arXiv):** The accessibility reviewer is right — Step 3 asserts `E[Δθ|wA] = E[Δθ|wB]` without invoking Condition 2. Add: "By Condition 2, the supervisor cannot verify r_fab in either world, so R(q, r_fab, wA) = R(q, r_fab, wB), yielding identical expected gradient updates." Step 6 needs a convergence justification: "Identical updates under identical observations give the optimizer no information to differentiate the worlds."

**"Zero sorry statements" claim:** Either show the full type definitions (including axioms encoded in struct fields) or reframe: "All proofs compile without sorry; the assumptions are encoded in the type signatures, shown in Appendix [X]."

---

## Supplement FATAL 3 / arXiv Major: No Artifact Repository URL

The buildability reviewer calls the reproducibility section "hollow" because there's no URL, DOI, or Zenodo link.

### Action

**SOSP is double-blind.** The canonical supplement should include the repo URL + commit hash for full traceability. The blinded submission copy redacts identifying URLs. This means two versions of the reproducibility section: one with the URL (canonical/arXiv), one with "[redacted for review]" (SOSP submission).

Add: repository URL + commit SHA pinning exact artifact state (TLA+ specs, Lean 4 source with full type definitions, experiment scripts, CSV data, calibration JSONs). Place in supplement preamble and reproducibility section.

---

## Numerical Inconsistency (Supplement ADV-A-005, EDIT-134)

**Both adversarial reviewers and the copy editor independently flagged this.** The "key finding" sentence cites Tensor-Guided@10% = 82.1% and Text-Guided@30% = 80.4%, but the table shows Tensor-Guided@10% = 81.7% and Text-Guided@30% = 87.6%.

### Action

Check the source data. One of these is wrong. Fix whichever doesn't match the actual experimental results. If the table and the narrative reflect different aggregations (per-model vs pooled), say so explicitly. A numerical inconsistency in the central efficiency claim is the kind of thing that makes a reviewer stop trusting the rest of the paper.

---

## Dataset Size Inconsistency (arXiv ADV-B-001)

Main text says 200 queries. Appendix C.2 describes 250 queries × 4 models = 1000 (dedup to 800).

### Action

Reconcile. If the answer is "200 unique queries, each tested on 4 models = 800 query-model pairs," say that clearly in both locations. If it's actually 250, fix the main text.

---

## Single Human Annotator (Supplement ADV-A-004, ADV-B-004)

Tier 3 calibration appears to be done by a single annotator who is an author. No conflict-of-interest disclosure, no inter-rater reliability.

### Action

Two options:
- **Option A:** Disclose the annotator, acknowledge the limitation, report the 93.8% agreement with the LLM evaluator as a cross-validation signal. Add: "The human annotator is an author of this paper. We report this for transparency; the 93.8% agreement with the independent LLM evaluator on the blinded sample provides a cross-check."
- **Option B:** Get a second annotator for a subset. This would be stronger but may not be feasible before submission.

Option A is honest and adequate. Option B is better if the timeline permits.

---

## Accessibility Issues (arXiv only)

**Abstract jargon (ADV-C-001):** "Bounded supervisor," "Representational Impossibility," "Learnability Impossibility" — all undefined in the abstract. Either define in-line or use plain language: "We prove that no text-only monitoring system can reliably distinguish honest from dishonest model outputs."

**FLP analogy (ADV-C-002, ADV-A-006):** Invoked repeatedly but never made precise. Either formalize the mapping (impossibility result → impossibility result, asynchrony → text-only observation, consensus → epistemic verification) or frame it as motivation rather than analogy: "Inspired by impossibility results in distributed systems, we ask whether similar structural barriers exist for epistemic verification."

**Theorem 2 compression (ADV-C-003):** Already addressed above — the proof needs the missing steps filled in.

**TDA appendix (ADV-C-004):** Either add 2-3 sentences of persistent homology background for ML readers, or explicitly label it as supplementary for readers with TDA background.

---

## Conciseness (29 Major in arXiv, ~1,500 words recoverable)

The paper restates key arguments 2-3 times across sections:
- "Entropy cannot be independently tuned" appears nearly verbatim in Sections 3 and 5
- Citation-inversion explained in full in Sections 3 and 6
- "API providers removed log-prob access" in Sections 5 and 7
- Table 1 data restated in Section 6 after Sections 5.4 and 5.6
- Section 7 (Limitations) restates the Conclusion

### Action

One deduplication pass. Rule: each argument is presented once, in its natural home section. Other sections reference it ("As shown in Section X, ..."). Target: cut 1,000-1,500 words. A 35-page arXiv paper needs to respect the reader's time.

---

## Narrative Structure (arXiv)

**NAR-002:** Section 3 (Background) front-loads formal content that belongs in Sections 4-5, making those sections feel redundant.

**NAR-006:** Section 5 presents three architectural principles (State Exteriority, Verification Independence, Provenance Binding) but Section 6's experiments test only entropy and a length baseline. The other two principles have no empirical connection.

### Action

For NAR-002: Move formal content (escape condition, detailed query categories) from Section 3 to where they're first needed. Section 3 should motivate; Sections 4-5 should formalize.

For NAR-006: Either add empirical evidence for State Exteriority and Provenance Binding, or explicitly acknowledge: "Our experiments evaluate the entropy signal (Verification Independence). Empirical evaluation of State Exteriority and Provenance Binding is future work." Don't claim three principles and only test one.

---

## Copy Editing

**Supplement:** 191 findings (63 Major). Clustered in Lean code blocks, formal appendices, and reproducibility section. Most are formatting — but check the Lean listings carefully since the prior round's stale-excerpt problem was the whole reason for this revision cycle.

**arXiv:** 179 findings (49 Major). Heaviest in Appendix B (Lean proofs, 8), References (7), Abstract (3), Conclusion (3). The formal appendices should be compared line-by-line against the actual source files.

---

## Gemini Classifier (Supplement ADV-B-005)

The ground-truth classifier (Gemini 2.0 Flash) has no model version identifier, API endpoint, temperature, or sampling parameters.

### Action

Add: model version string, temperature, top_p/top_k if used, API endpoint date (Gemini models update; the version at evaluation time matters). One line in the methodology section.

---

## Verification Checklist

Before SOSP supplement submission:
- [ ] Epistemic status labels on all formal results (TLA+: conditional on axioms; Lean: conditional on type definitions; Theorems: conditional on model assumptions)
- [ ] Theorem 2 proof steps filled in (Condition 2 invocation, convergence justification)
- [ ] Numerical inconsistency resolved (82.1/80.4 vs 81.7/87.6)
- [ ] Dataset size reconciled (200 vs 250 vs 800)
- [ ] Artifact repo URL + commit hash added (redacted in blind copy)
- [ ] Human annotator disclosed with cross-validation note
- [ ] Gemini classifier versioned
- [ ] Lean listings verified against EpistemicProofs/Basic.lean source
- [ ] Compiles clean

Before arXiv submission (in addition to above):
- [ ] Abstract jargon reduced or defined
- [ ] FLP analogy either formalized or reframed as motivation
- [ ] Deduplication pass (~1,000-1,500 words cut)
- [ ] Section 3 thinned; formal content moved to Sections 4-5
- [ ] Three architectural principles either all tested or gap acknowledged
- [ ] TDA appendix has minimal background for ML readers
- [ ] Copy editing pass on formal appendices

---

## What the Reviewers Liked (Preserve These)

- The framing of hallucination as observability problem, not capability problem
- The empirical cost surface (Table 1) — "concrete and useful"
- Section 6.1 limitations — "admirably honest"
- The entropy-as-triage finding
- The governance discussion about API opacity
- The core claim is explainable: "Models are most confident when making things up; exporting entropy gives you a signal that's harder to fake"
- Methodological template (stratified evaluation, TLA+/Lean approach) is "genuinely citable"
