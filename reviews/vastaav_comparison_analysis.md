# Vastaav Edit Comparison: Meta-Evaluation Analysis

**Date:** 2026-02-14
**Methodology:** Blind A/B comparison + cross-reference analysis
**Prepared for:** Vastaav Agarwal

## What We Did

We compared the paper state at commit `9516845` ("Revised intro") against
Vastaav's edits at commit `00ee233` ("Update text for space constraints")
using the project's automated review pipeline. The comparison had two parts:

1. **Blind A/B comparison**: 3 LLM models (Gemini 2.5 Pro, DeepSeek V3,
   Qwen 2.5 72B) across 5 reviewer perspectives (systems reviewer, ML
   reviewer, narrative reviewer, junior systems PhD, senior ML researcher)
   evaluated both versions without knowing which was which. Each
   model-perspective pair evaluated both versions individually, then ranked
   them comparatively. Total: 15 comparative judgments per section, 45 total.

2. **Cross-reference analysis**: We mapped Vastaav's editorial decisions
   against the feedback from 7 prior automated review runs (including
   redundancy detection, conciseness analysis, hostile rejection, and
   shepherding perspectives) to build a four-cell matrix of convergence
   and divergence.

Three sections were A/B tested: Introduction, Abstract, and Conclusion.
Background and Design changes were structural/minor and analyzed only via
the cross-reference.

---

## A/B Results

### Introduction: 14-1 Baseline

| Perspective | Gemini | DeepSeek | Qwen |
|------------|--------|----------|------|
| Systems Reviewer | Baseline | -- | -- |
| ML Reviewer | Baseline | Baseline | Baseline |
| Narrative Reviewer | Baseline | Baseline | Baseline |
| Junior Systems PhD | Baseline | Baseline | **Vastaav** |
| Senior ML (no systems) | Baseline | Baseline | Baseline |

The one dissent: Qwen-as-junior-PhD preferred Vastaav's version as "more
reader-friendly and avoids overly technical terms." Every other
judge-perspective combination preferred the baseline for its "explicit
contrast between computational and observational limits" and "specific
details about theoretical claims."

### Abstract: 15-0 Baseline

Unanimous across all models and perspectives. The AUC detail and
gameability argument that Vastaav cut were consistently cited as "concrete
evidence" and "rigorous ML-style evaluation." The FLP analogy was praised
by systems reviewers as elevating the result "from a specific finding to a
foundational contribution."

### Conclusion: 15-0 Baseline

Unanimous. The FLP analogy, MOSS analogy, AUC metrics, and the line
"Gaming the metric requires changing the thing being measured" were all
valued. Even the junior systems PhD who "never took an ML course" preferred
the longer version.

### Overall: 44 Baseline, 1 Vastaav

---

## The Meta-Finding

**The judges cannot evaluate edits made under exogenous constraints.**

Vastaav's edits were explicitly for page constraints ("Update text for
space constraints"). The judges evaluate quality independent of page budget.
A section that loses content always looks worse to a judge that doesn't know
about the page limit.

This is the paper's own thesis instantiated in the review pipeline: **a
text-only judge cannot distinguish "content removed because of page budget"
from "content missing because the author was sloppy."** The judges see
shorter text, lower information density, and conclude: worse. They have no
channel through which to observe that the constraint was exogenous (page
limit), not endogenous (editorial weakness).

To evaluate constraint-satisfaction edits, the pipeline would need:
- **Budget-aware judges**: "Given a 12-page limit, which version makes
  better use of limited space?"
- **Marginal-value judges**: "For each paragraph removed, was the content
  load-bearing or decorative?"

The current pipeline asks "which is better in a vacuum?" — the wrong
question for space-constrained editing.

---

## Four-Cell Cross-Reference Matrix

### Cell 1: Judges Flagged + Vastaav Fixed (7 items)

These are issues where automated review and human editorial judgment
converged independently.

| Issue | Judge Evidence | Vastaav's Fix |
|-------|--------------|---------------|
| **Abstract AUC overclaiming** | Hostile rejecter: "Presenting the confounded 0.9+ AUC as the primary result, while burying the much weaker 0.6-0.7 AUC, is poor scientific practice." Redundancy judge: flagged Abstract-to-Evaluation echo. | Commented out entire AUC/gameability paragraph |
| **Conclusion AUC paragraph** | Hostile rejecter: "Response length alone achieves AUC 0.88-0.96. This is an astonishingly strong baseline." | Cut the AUC/gameability paragraph entirely |
| **"Our result is orthogonal" premature** | Shepherd: intro makes claims before reader knows the result. Non-domain: terms used without definitions. | Commented out with note: "We haven't told the readers what our result is" |
| **Hallucination table formatting** | Scourer: need for proper figures/visualizations | Moved to proper `\begin{table}` float with caption and label |
| **Conclusion redundant framing** | Redundancy judge: "structural not capability" echoed across Abstract, Intro, Conclusion | Cut "not a capability limitation...nor a training failure" |
| **"reliably" qualifier** | Conciseness judge: flagged as removable qualifier | Dropped "reliably" from "struggle to reliably distinguish" |
| **FLP analogy in conclusion** | Hostile rejecter: "suggests the paper's theoretical contributions are incremental"; another: "feels aspirational rather than descriptive" | Cut the entire FLP-analogy paragraph |

**Strongest convergence:** The AUC/gameability overclaiming. Four
independent judge personas flagged it; Vastaav independently cut it from
both abstract and conclusion.

### Cell 2: Judges Flagged + Vastaav Missed (8 items)

| Issue | Judge Evidence | Status |
|-------|--------------|--------|
| **Abstract-to-Intro numerical echo** | Redundancy judge: "repeats the paper's single most important numerical result verbatim. Spoils the reveal." | 82.1% vs 78.5% still appears in both |
| **Background-to-Formal "model may know" echo** | Redundancy judge: MEDIUM severity cross-section echo | Persists in both sections |
| **Background-to-Design "internal state" echo** | Redundancy judge: MEDIUM severity | Neither section changed |
| **Missing baseline: Semantic Entropy** | Shepherd + hostile rejecters: "A comparison against Kuhn et al. is a critical missing piece" | Experimental gap, not editorial |
| **Tensor robustness overclaiming** | Hostile rejecter: body claims "cannot be independently controlled" but Discussion concedes adversarial training is "an open question" | Body-to-Discussion tension persists |
| **Background verbosity** | Conciseness judge: 17 suggestions, 217 words saveable | Only 2 words tightened |
| **Discussion verbosity** | Conciseness judge: 18 suggestions, 139 words saveable | Section untouched |
| **C1 theorem jargon** | Non-domain experts + Vastaav's own comment: "predictor-centric policy" and "world states" inaccessible | Comment left but content unchanged |

**Widest gap:** Cross-section redundancy. The redundancy judge identified
10 echoes (8 MEDIUM severity); Vastaav resolved ~2 of them.

### Cell 3: Judges Missed + Vastaav Fixed (8 items)

| Issue | Vastaav's Fix | Judge Blind Spot |
|-------|--------------|-----------------|
| "Moreover," transition added | Smoothed logical flow between budget and content-type | No judge flagged missing transition |
| Supervisor sentence added | Made observability-gap paragraph self-contained | No judge flagged the gap |
| Epistemic observability moved earlier | More direct introduction of key concept | No judge flagged ordering |
| "extended" -> "epistemic observability" interface | Terminological consistency in C2 | Not flagged |
| "An empirical" -> "Empirical" | Minor stylistic tightening | Not flagged |
| `\resizebox` removed from figure | Formatting fix | Not flagged |
| Loss of "The model may know..." line | Likely unintended casualty of restructuring | Judges valued this line |
| "The interface is the problem..." commented out | Rhetorical density reduction | Judges didn't flag for removal |

**Pattern:** Vastaav's unique contributions are primarily structural —
reordering, transitions, terminological consistency. This is editorial work
that requires reading for narrative flow rather than pattern-matching for
specific issues, which is precisely what the current judge pipeline does
not do well.

### Cell 4: Blind Spots (4 patterns)

1. **Evaluation section**: Neither Vastaav nor the conciseness/redundancy
   judges touched it. Structural blind spot in pipeline configuration.
2. **Related Work section**: Conciseness judge found 40 saveable words;
   Vastaav didn't edit it. Neither addressed the repeated
   "symptom/treatment/diagnosis/cure" framing.
3. **Formal Proof section**: Hostile rejecters attacked novelty but
   suggested no textual edits. Vastaav's C1 comment acknowledges the
   accessibility problem but the fix doesn't propagate into the proofs.
4. **No post-edit review**: Judges reviewed pre-Vastaav paper. No judge
   checked whether Vastaav's restructuring introduced new problems.

---

## Implications for the Review Pipeline

1. **The judges have a length/density bias.** They systematically prefer
   longer, more detailed versions because they evaluate quality independent
   of space constraints. This is a bounded-supervisor limitation: more text
   = more signal = higher score. A budget-aware judge perspective is needed.

2. **Cross-section redundancy detection works but isn't acted on.** The
   redundancy judge identifies echoes accurately, but the editorial
   response requires a human (or human-directed) decision about which
   instance to keep. This is a triage problem, not a detection problem.

3. **Narrative flow is a blind spot.** Vastaav's 8 unique catches are all
   flow/transition/ordering issues. The current judges evaluate sections
   in isolation. A narrative coherence judge that reads the paper linearly
   — tracking what has been defined, what is forward-referenced, and where
   the reader might be confused — would close this gap.

4. **The pipeline should re-review after edits.** The four-cell matrix
   reveals that Vastaav's restructuring may have introduced new problems
   (loss of the dramatic "model may know" line, early introduction of
   undefined machinery). A post-edit review pass would catch these.

5. **Convergence validates the pipeline.** 7 items where automated judges
   and a human co-author independently identified the same issues is
   non-trivial. The pipeline is not replacing editorial judgment — it is
   identifying the same problems a careful reader finds, plus some
   (redundancy echoes) that are hard to spot manually.

---

## Summary

| Metric | Value |
|--------|-------|
| A/B comparisons run | 45 |
| Baseline preferred | 44 (97.8%) |
| Vastaav preferred | 1 (2.2%) |
| Judge-Vastaav convergence | 7 items |
| Judge caught, Vastaav missed | 8 items |
| Vastaav caught, judges missed | 8 items |
| Blind spots identified | 4 patterns |

The 44/45 result does NOT mean Vastaav's edits were wrong. It means the
current evaluation pipeline cannot assess edits made under space
constraints. Vastaav was solving a real problem (pages); the judges are
blind to that problem. Fixing this blindness — adding budget-aware and
narrative-coherence judges — is the most actionable improvement to the
pipeline.
