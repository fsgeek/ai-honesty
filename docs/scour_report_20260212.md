# Scour Report: Paper Rewrite (Feb 12, 2026)

Five independent scourers with different lenses. No hypothesis specified.

## Convergent Findings (flagged by 3+ scourers)

### 1. Manifold/scaffolding framework promised, never delivered (4/5)
Abstract, intro, and conclusion all promise "a manifold parameterized by scaffolding ratio
and rigidity." No figure, no table, no definition of scaffolding ratio or rigidity anywhere
in the paper body. The abstract says "the contribution is the map" but the map is not drawn.

### 2. "Cannot independently control" unqualified 6x, contradicted by discussion (3/5)
Body text asserts the model "cannot independently control" its tensor signals six times.
Discussion calls this "an open question" (lines 25-27) and admits adversarial training
could break it (lines 48-49). Either qualify every use or remove the contradiction.

### 3. Visible `\va{}` author note in design.tex:159 (4/5)
"We need to define these signals properly." — renders as purple text. The three signals
listed after it are never defined. Signals unfinished work to any reader.

### 4. Fragmentation/Cognitive Slope defined, never evaluated (3/5)
design.tex:170-191 defines these topological metrics. They never appear in any table,
figure, or result. Vestigial from earlier TDA-focused draft. Cut or evaluate.

### 5. MOSS analogy overused (5 occurrences) and strained (3/5)
Abstract, intro, eval, discussion, conclusion. MOSS compares two artifacts; the tensor
interface examines one. Different task structure. And MOSS works because its signal is
strong; the residual AUC is 0.60-0.70.

### 6. No actual cost model despite "cost surface" framing (3/5)
No units of cost. No latency, FLOPs, dollars, or compute measurements. "Cost" is used
only as "fraction of outputs verified." The "cost surface" is metaphor, not model.

## Factual Errors (fix immediately)

### 7. Self-report AUC range "0.28--0.46" is wrong
Intro line 114. Highest per-model value is 0.362 (Mistral). Range should be 0.28--0.36.
The "0.46" is stale from an earlier experiment.

### 8. "Composed judge achieves highest accuracy at every budget level" is false
Figure caption and eval text. Table shows Composed=80.5% at 10% vs Tensor-guided=82.1%.
Composed is NOT highest at 10%. Fix claim or explain.

## Structural Issues

### 9. Missing baselines — especially Semantic Entropy (Kuhn et al., 2023)
Most direct competitor. Uses generation entropy to detect hallucination. Not cited or
benchmarked. Hostile ML reviewer calls this "the single most damaging omission."

### 10. Text-guided baseline artificially bad
Includes self-report (AUC 0.28-0.46), which is anti-correlated. A length-only text-guided
judge would likely match tensor-guided, potentially demolishing the headline result.

### 11. Four probe categories defined (background), never used in evaluation
Background:49-67 defines Adversarial Truth, Shattered Lie, Deceived Lie, Confused Truth.
Evaluation uses only knowable/unknowable binary. Orphaned taxonomy.

### 12. Theorem novelty concerns
Theorem 1 is "a function that ignores its input cannot depend on its input" — reviewers
will see this as definitional. Theorem 2 proves existence of hard cases but is deployed
as if the general problem is unsolvable. The FLP comparison flatters the result.

### 13. No system built
The "tensor interface" is a function signature. No latency, memory, throughput, or
production pipeline. SOSP papers build and evaluate systems.

## Numbers Without Tables

Per-model accuracy improvements, pooled AUC values, Cohen's d = 1.57, TruthfulQA AUC ~0.53,
initial evaluator 68.8% accuracy — all stated in text with no supporting table or figure.
No AUC in the paper carries a confidence interval.

## Page Budget: 14→12 pages

### Top 3 cuts (rhetoric reviewer, ~56 lines / ~1 column):
1. Discussion:56-72 — MOSS bridge restatement (17 lines, fifth occurrence)
2. Discussion:120-133 — "The methodology performed itself" (14 lines, self-congratulatory)
3. Eval:190-225 — Raw AUC numbers immediately called "misleading" (25 lines, redundant with decomposition)

### Additional cut candidates:
- TLA+ subsection (formal_proof:342-363) — ~0.5 page, adds nothing beyond proofs
- Composition Graph / Superlinear Verification Cost (formal_proof:281-305)
- QoS Tiers (design:114-148) — entirely qualitative, no evaluation support
- Tensor-Gated Composition (design:247-289) — n=3, anecdotal
- Deduplicate: "This paper provides the cost surface" (5 occurrences → 2)
- Deduplicate: self-report inversion (4 occurrences → 2)

## Repetition Count

| Phrase/idea | Count | Keep |
|---|---|---|
| MOSS analogy | 5 | intro + eval |
| "This paper provides the cost surface" | 5 | abstract + conclusion |
| Tensor@10% > Text@30% (82.1% vs 80.4%) | 4 | eval + abstract |
| Self-report inversion | 4 | background + eval |
| "Impossibility is structural" | 5 | intro + formal |
| "The testimony lies; the telemetry does not" | 3 | intro only |

## Verdict Summary

- Systems reviewer: "Does not meet the bar" — no system built, theorem near-trivial
- Hostile ML reviewer: "Reject (borderline)" — missing baselines, thin empirical coverage
- Clarity reviewer: "Abstract promises manifold that's never delivered"
- Numbers auditor: Two factual errors, many unsupported numbers
- Rhetoric reviewer: 6+ unqualified claims contradicted by own discussion

## What the Scourers Agree Would Help

1. Cite and benchmark semantic entropy (Kuhn et al., 2023)
2. Either deliver the manifold framework or remove it from abstract/intro/conclusion
3. Qualify "cannot independently control" everywhere it appears
4. Fix the two factual errors (self-report range, composed judge claim)
5. Cut ~56 lines of repetition (gets halfway to 12 pages)
6. Add confidence intervals to key numbers
7. Remove Fragmentation/Cognitive Slope or evaluate them
8. Resolve the `\va{}` note and define the three signals
