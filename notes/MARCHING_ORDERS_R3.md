# ai-honesty: Marching Orders Round 3 (Rikuy Reviews 2026-03-17)

**Reviews:**
- SOSP Supplement: `~/projects/rikuy/reviews/sosp-supplement/review_20260317_133129.jsonl` — 220 findings (2F, 81M, 137m)
- arXiv Version: `~/projects/rikuy/reviews/ai-honesty-arxiv/review_20260317_133149.jsonl` — 328 findings (1F, 132M, 195m)

**Prior round:** R2 had 5 fatals across both documents. This round has 3. The epistemic status labeling worked — the honesty reviewer now calls Remark 1's taxonomy "genuinely commendable" and gave 0 fatals on the arXiv version. Theorem 2 proof completeness is no longer fatal. Lean framing is no longer fatal.

**Citation verification (separate pass):** All 39 bib entries verified as real papers. Zero fabricated citations. 12 unused bib entries (harmless — bibtex won't include them). Minor: `elhage2021mathematical` year field says 2022, should be 2021; `qwen3` could add arXiv ID 2505.09388.

**Commit hash:** Pinned in both documents: `a9c1334`.

---

## What's Been Resolved (Do Not Revisit)

- Epistemic status labels on formal results — working, praised by reviewers
- Theorem 2 proof steps — filled in, no longer fatal
- Lean proof framing — no longer fatal
- Artifact repo URL — present in both documents
- Commit hash — pinned

---

## FATAL 1 (Supplement): TLA+ Circularity — Persistent, Likely Structural

**ADV-A-001:** The TLA+ tensor escape spec encodes its conclusion as axioms, then TLC "verifies" it. The honesty reviewer flagged this in R2 and flags it again in R3 despite added framing.

### Assessment

This may be structurally unfixable from the reviewer's perspective because the spec *is* circular by design — that's what formal modeling does (model the system, verify the model is consistent). The framing grew from 73 to 143 words between rounds. The reviewer's concern is legitimate in principle but may represent a ceiling for this persona.

### Action

One more attempt: add a sentence like "The TLA+ specification is intentionally constructive: it encodes our design axioms and verifies their internal consistency. It does not prove that real systems satisfy these axioms — that is an empirical question addressed in Section 3. The value of the formal model is that it makes the assumptions explicit and machine-checkable, so that disagreement can focus on the axioms rather than on the logical consequences." If this doesn't satisfy the reviewer, accept it as a known limitation of the review persona and move on.

---

## FATAL 2 (Supplement): Commit Hash Deferred — ALREADY FIXED

**ADV-B-001:** The review was run on a version that still said "will be pinned at submission time." The commit hash `a9c1334` has now been pinned in both documents. This fatal is resolved.

---

## FATAL 3 (arXiv): "Cost Surface" Undefined — NEW, EASY FIX

**ADV-C-002:** The abstract and intro use "cost surface" as if self-explanatory. It sounds like a formal mathematical object (loss landscape) but is actually a 2×4 accuracy-vs-budget table. The accessibility reviewer couldn't figure out what it meant until Table 1.

### Action

Define on first use (abstract or intro): "By cost surface, we mean the empirical mapping from verification budget (fraction of queries receiving expensive checks) to detection accuracy for each judge strategy — a practical lookup for system builders deciding how to allocate verification resources." One sentence, first occurrence, both documents if the term appears in the supplement.

---

## NEW Issues From R3 (Likely Introduced During Revision)

### Qwen Circularity (arXiv ADV-B-003) — HIGH PRIORITY

Qwen3-4B-Instruct appears both as a test subject (one of the 4 local models) AND as the Tier 2 LLM classifier for ground truth evaluation. This is a methodological circularity that a human reviewer will catch.

**Action:** Either (a) use a different model for Tier 2 classification and re-run, or (b) explicitly acknowledge: "Qwen3-4B-Instruct appears in both roles. We mitigate potential circularity by noting that the Tier 2 classifier operates on a different task (factual verification) than the test task (response generation), and that the 93.8% human-agreement rate on blinded samples provides an independent check." Option (b) is faster; option (a) is cleaner.

### Attention Summaries Contradiction (arXiv ADV-B-005) — HIGH PRIORITY

Section 5.3 says the tensor-guided judge uses "per-token entropy and attention summaries." Section 4.1 says attention summaries were "measured but not used" in experiments. These contradict each other.

**Action:** Determine which is true and make both sections consistent. If attention summaries were measured but not included in the judge, update Section 5.3. If they were included, update Section 4.1.

### Only 3 of 5 API Models Named (arXiv ADV-B-004)

The paper claims API validation across 5 models but names only 3, leaving "two others" unspecified.

**Action:** Either name all 5 or explain why 2 are withheld (e.g., NDA, provider terms). Unspecified models undermine the cross-architecture generalization claim.

### Lean Theorem 2 Proves Weaker Statement Than Claimed (arXiv ADV-B-006)

The Lean proof proves learning updates are equal when observations are equal — it doesn't prove the policy cannot converge to honest behavior (which is the claimed theorem).

**Action:** Check whether this is a framing issue or a real gap. If the Lean proof proves the weaker "updates are identical" statement and the paper's prose claims the stronger "cannot converge to honesty" statement, either strengthen the proof or qualify the claim: "The formal proof establishes that parameter updates are identical in both worlds; the impossibility of convergence to honest behavior follows informally from the fact that identical updates cannot produce divergent policies."

---

## Persistent Issues (From R2, Still Flagged)

### Dataset Size: 200 vs 800 Queries (arXiv ADV-B-002)

Still unresolved. Main text says 200, appendix implies 800.

**Action:** Reconcile. This was flagged in R2. Fix it.

### Abstract/Conclusion Overclaiming (arXiv ADV-A-001, A-002, A-006)

The body is well-qualified (Remark 1 praised). But the abstract says "We prove" without caveats, and uses universal language ("regardless of model scale or training procedure") that the formal model doesn't support.

**Action:** Mirror the body's epistemic care in the abstract and conclusion. Replace "We prove" with "We prove, under explicit formal assumptions," or "Within our formal model, we prove." Replace "regardless of scale" with "for all models satisfying our formal assumptions." The body already does this well — the abstract and conclusion just need to catch up.

### Entropy-Can't-Be-Faked (arXiv ADV-A-004, ADV-C-007)

The central justification for the tensor interface — that per-token entropy resists adversarial manipulation — is stated as near-certain in the main text but acknowledged as an open question in the limitations. The accessibility reviewer calls it "the paper's most contestable claim."

**Action:** Foreground the uncertainty. In Section 5 where the claim is made, add: "This argument assumes current training methods; whether adversarial fine-tuning could decouple entropy from correctness is an open empirical question (see Section 7)." Don't bury the caveat in limitations — put it next to the claim.

### Conciseness (50 Major in arXiv, up from 29 in R2)

The paper grew from 35 to 38 pages. The conciseness judge found ~1,500+ words of duplication:
- "Cannot independently tune" argument appears in 4 sections
- AUC 0.757 restated identically in Sections 5 and 6
- "API access eroded" echoed between Sections 5 and 7
- Limitations restates the Conclusion
- Background pre-empts formal results (NAR-002)

**Action:** Deduplication pass. Each argument lives in one section. Other sections reference it. Target: bring the paper back to 35 pages or under. The conciseness findings doubled because the paper got longer — reverse that trend.

### Redundancy (3 Major)

- "Cannot independently tune" — 4 sections, nearly verbatim
- "AUC 0.757 consistent across architectures" — Sections 5 and 6
- "API log-prob access eroded" — Sections 5 and 7

These will be caught by the conciseness pass.

---

## Copy Editing Note

**Supplement:** 193 findings (71 Major). **arXiv:** 188 findings (60 Major). Most of these are PDF extraction artifacts (mid-word hyphens, stray page numbers). The copy editor is reviewing garbage input from the PDF extractor, not actual paper issues. Skim the findings for anything substantive (the Lean listings, reference formatting) but don't chase phantom hyphenation errors.

---

## Verification Checklist (Updated for R3)

Before SOSP supplement submission:
- [x] Epistemic status labels on formal results — done, praised
- [x] Theorem 2 proof steps — done, no longer fatal
- [x] Artifact repo URL + commit hash — done (a9c1334)
- [ ] TLA+ circularity framing — one more attempt (see above)
- [ ] Numerical inconsistency resolved (82.1/80.4 vs 81.7/87.6) — verify this was fixed in R2
- [ ] Gemini classifier versioned
- [ ] Compiles clean

Before arXiv submission:
- [ ] "Cost surface" defined on first use
- [ ] Qwen circularity addressed (acknowledge or use different Tier 2 model)
- [ ] Attention summaries contradiction resolved
- [ ] All 5 API models named or omission justified
- [ ] Lean Theorem 2 claim/proof scope aligned
- [ ] 200 vs 800 queries reconciled
- [ ] Abstract/conclusion language mirrors body's epistemic care
- [ ] Entropy-can't-be-faked caveat placed next to claim, not just in limitations
- [ ] Deduplication pass (target: ≤35 pages)
- [ ] Minor bib fixes: elhage year 2022→2021, qwen3 add arXiv ID

---

## What the Reviewers Liked (Preserve These)

- Remark 1's formal/empirical/informal taxonomy — "genuinely commendable"
- Framing of hallucination as observability problem, not capability problem
- Cost surface (Table 1) — "concrete and useful" (once defined!)
- Limitations section — "admirably honest," "engages with real weaknesses"
- Entropy-as-triage finding
- Governance discussion about API opacity
- Core claim is explainable to labmates in one sentence
- Methodological template is "genuinely citable"
- The paper improved significantly between R2 and R3
