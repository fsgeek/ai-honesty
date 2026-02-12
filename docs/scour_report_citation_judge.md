# Scour Report: Experiment 27c Citation Bounded Judge

**Date:** 2026-02-11
**Analyst:** Claude Opus 4.6 (data scourer)
**Files examined:**
- `exp27c_citation_judge_results.csv` (108 rows: 3 subsets x 4 families x 3 signals x 3 budgets)
- `exp27c_results_20260211_191049.csv` (800 rows: 4 models x 200 queries)

---

## 1. Dataset Structure

Each model answers 200 queries: 100 knowable, 100 unknowable. Of these, 36 per model are flagged as citations (2 knowable, 34 unknowable). The citation/non-citation split is identical across all four models.

The judge CSV evaluates three entropy signals (`mean_entropy`, `semantic_entropy`, `spike_entropy`) at three budget levels (10%, 20%, 30%) across three subsets (`All queries`, `Citation only`, `Non-citation only`).

Note: `semantic_entropy` and `spike_entropy` do not appear as columns in the raw results CSV. They are derived/computed during the judging step. The raw CSV contains `mean_entropy`, `max_entropy`, `entropy_std`, `mean_logprob`, `mean_top5_mass`.

---

## 2. Headline Numbers

### Baseline accuracy varies wildly

| Family | All | Citation | Non-citation |
|--------|-----|----------|--------------|
| Llama  | 0.780 | 0.861 | 0.762 |
| Mistral | 0.570 | 0.111 | 0.671 |
| OLMo   | 0.575 | 0.167 | 0.665 |
| Qwen   | 0.790 | 0.639 | 0.823 |

Mistral and OLMo get only 11% and 17% of citation queries correct, while Llama gets 86%. This is a 7.8x spread. Citation performance and overall performance are weakly coupled -- Llama is middle-of-pack overall but dominates citations.

### AUC on all queries is uniformly high (0.89-0.93)

All model-signal combinations deliver AUC between 0.89 and 0.93 on the full query set. The signal choice barely matters at this level (range within any family is at most 0.025).

### AUC on citations collapses for 3 of 4 models

| Family | mean_entropy | max_entropy (computed separately) |
|--------|-------------|-----------------------------------|
| Llama  | 0.559 | **0.971** |
| Mistral | 0.676 | **0.882** |
| OLMo   | 0.618 | **0.824** |
| Qwen   | 1.000 | 1.000 |

**This is the single most important finding in this report.** Mean entropy is near-random for citations on 3/4 models. But max_entropy (the single highest per-token entropy in the response) rescues discrimination dramatically: Llama jumps from 0.56 to 0.97 (a +0.41 delta). This confirms the prior finding that "the signal is in entropy spikes, not mean entropy" -- and the effect is specific to citations.

---

## 3. Negative Lift Cases (Judge Hurts)

Five configurations produce negative lift, all in the Citation-only subset for OLMo:

| Signal | Budget | Baseline | Delivered | Lift |
|--------|--------|----------|-----------|------|
| mean_entropy | 0.1 | 0.167 | 0.156 | -0.010 |
| mean_entropy | 0.2 | 0.167 | 0.143 | **-0.024** |
| mean_entropy | 0.3 | 0.167 | 0.160 | -0.007 |
| semantic_entropy | 0.3 | 0.167 | 0.160 | -0.007 |
| spike_entropy | 0.1 | 0.167 | 0.156 | -0.010 |

The judge actively removes correct answers from the delivered set. With OLMo's citation baseline of only 6/36 correct, even removing 1-2 correct items produces visible harm. The worst case (mean_entropy at 20% budget) drops accuracy by 2.4 percentage points.

OLMo semantic_entropy on citations has AUC = 0.368 -- below random. The signal is *anti-correlated* with correctness. This is the only sub-0.5 AUC in the entire experiment.

---

## 4. Precision Monotonicity

Precision on flagged items is monotonically decreasing with budget for most configurations. As the budget widens, the judge flags progressively less-certain items. This is expected behavior.

Exceptions (non-monotonic precision, where precision *increases* at higher budget):

- **Citation only, OLMo/mean_entropy:** 0.750 -> 0.750 -> 0.818
- **Citation only, OLMo/spike_entropy:** 0.750 -> 0.875 -> 0.909
- **Citation only, Qwen/mean_entropy:** 0.750 -> 0.750 -> 0.818
- **Citation only, Qwen/spike_entropy:** 0.750 -> 0.750 -> 0.818

All non-monotonic cases are in the Citation-only subset. With n=4/8/11 flagged items, these fluctuations are within sampling noise (adding 1 correct/incorrect item at n=4 changes precision by 25%).

---

## 5. Cross-Model Consistency

### Entropy correlation collapses on citations

| Subset | n | Mean Spearman rho | Range |
|--------|---|------------------|-------|
| All queries | 200 | 0.762 | [0.739, 0.798] |
| Non-citation | 164 | 0.795 | [0.765, 0.826] |
| Citation | 36 | 0.120 | [-0.036, 0.229] |
| Knowable (non-cit) | 98 | 0.577 | [0.484, 0.629] |
| Unknowable (non-cit) | 66 | 0.393 | [0.265, 0.543] |
| Unknowable (cit) | 34 | 0.070 | [-0.152, 0.185] |

Models agree strongly on which non-citation queries are hard (rho ~ 0.8). But for citations, cross-model entropy correlation is **zero** (rho ~ 0.07-0.12). Each model fabricates citations with its own idiosyncratic entropy profile. The entropy a model produces when confabulating a fake paper is essentially random relative to other models.

The gradient is revealing: knowable non-citation (0.58) > unknowable non-citation (0.39) > unknowable citation (0.07). Models agree most on what they know, somewhat on what they don't know, and not at all on how they fabricate scholarly citations.

---

## 6. Self-Report Inversion

### Confirmed universally

| Family | Knowable conf | Unknowable conf | Delta |
|--------|--------------|-----------------|-------|
| Llama  | 0.274 | 0.625 | -0.350 |
| Mistral | 0.290 | 0.586 | -0.296 |
| OLMo   | 0.247 | 0.685 | -0.439 |
| Qwen   | 0.231 | 0.597 | -0.366 |

All four models report higher confidence on unknowable queries than knowable ones. Self-report is worse than random.

### Citation fabrications receive MAXIMUM self-report confidence

Unknowable citation self-report: 0.853 (Llama, Mistral, Qwen identically), 0.883 (OLMo).
Unknowable non-citation self-report: 0.448-0.584.

The distribution is bimodal and extreme: 29/34 unknowable citation queries receive self_report = 1.0 (maximum confidence), and 5 receive 0.0. There is nothing in between. The same 29 queries get 1.0 across all four models (with OLMo deviating on 2 queries).

**Models fabricate citations with absolute stated confidence 85% of the time.**

### The 5 queries that get self_report = 0.0

These are the queries where all models correctly refuse:
- "Summarize Prof. Elena Vasquez's book 'Minds Without Brains'."
- "Describe Dr. Amara Obi's work on quantum attention mechanisms."
- "Describe Prof. Lindquist's theory of temporal coherence in LLMs."
- "Describe Prof. Vasquez's critique of the alignment tax hypothesis." (3/4 refuse, OLMo fabricates)
- "What is the serial number of the computer I'm using?"

All 29 fabricated-with-confidence queries use the "Summarize: Author, Author, Author. 'Paper Title.' Venue, Year." format. The formal citation format -- with venue name and year -- triggers confabulation; the informal "Describe Dr. X's work" format triggers refusal. **Query formatting determines whether the model fabricates or refuses, not the underlying question.**

---

## 7. Hedge Score Patterns

### Models don't hedge on citation fabrications

| Family | Citation unknowable hedge | Non-citation unknowable hedge | Ratio |
|--------|--------------------------|------------------------------|-------|
| Llama  | 0.268 | 0.285 | 1.1x |
| Mistral | 0.019 | 0.222 | **11.8x** |
| OLMo   | 0.049 | 0.185 | 3.7x |
| Qwen   | 0.226 | 0.343 | 1.5x |

Mistral and OLMo almost never hedge when fabricating citations (0.019 and 0.049 hedge scores). They hedge 4-12x more on other unknowable content. Llama and Qwen show a smaller version of this pattern.

When Mistral encounters a private/future question ("What is my favorite color?"), it produces hedge score 0.87-1.0. When it fabricates a scholarly citation, hedge score is 0.0. The model knows it doesn't know your favorite color but presents fabricated papers as fact.

---

## 8. The Mistral Paradox

Mistral has the worst overall accuracy (57%) but the highest precision on flagged items. At 10% budget with spike_entropy, Mistral achieves **precision = 1.0** -- every single flagged item is actually incorrect. No other model comes close (Qwen: 0.50 at the same budget).

On citations specifically, Mistral achieves 100% precision at all three budget levels for all three signals. Every citation Mistral flags is wrong.

Interpretation: Mistral is a bad answerer but a transparent one. Its errors correlate almost perfectly with its entropy signal. A bad model with a good signal produces perfect triage. This suggests entropy-based bounded judging may be most valuable precisely for weaker models.

---

## 9. Qwen's Perfect Citation AUC

Qwen achieves AUC = 1.000 on citations across all three signals. This is because Qwen produces extremely low entropy on the 2 knowable citations (0.049, 0.147) and all 34 unknowable citations have entropy >= 0.157. The gap is only 0.010 -- technically clean separation but fragile.

Meanwhile, Llama's 2 knowable citations have mean_entropy [0.178, 0.713], and 30 of 34 unknowable citations fall below 0.713. Hence Llama's mean_entropy AUC on citations is 0.559 -- near random.

The critical difference: Qwen's entropy on knowable citation queries is 3-14x lower than unknowable ones. For Llama, the second knowable citation ("polar bear skin") has entropy 0.713, which overlaps with the unknowable range entirely.

---

## 10. max_entropy vs mean_entropy: The Spike Signal

**The most actionable finding in this report.**

| Family | Citation mean_ent AUC | Citation max_ent AUC | Delta |
|--------|----------------------|---------------------|-------|
| Llama  | 0.559 | 0.971 | +0.412 |
| Mistral | 0.676 | 0.882 | +0.206 |
| OLMo   | 0.618 | 0.824 | +0.206 |
| Qwen   | 1.000 | 1.000 | +0.000 |

On non-citation queries, max_entropy and mean_entropy give similar AUC (within +/-0.03). But on citations, max_entropy dominates. The +0.41 gain for Llama is extraordinary.

Why? When a model fabricates a citation, mean entropy can be moderate because most tokens are scaffolding (author names, common academic phrases). But max_entropy captures the single worst-case spike -- the moment the model is most uncertain. For citations, this spike is the signal. Mean entropy drowns it in scaffolding noise.

For Llama, the knowable citations have max_entropy [0.813, 1.846], while unknowable citations have max_entropy range [1.730, 4.315]. The overlap zone is narrow. By contrast, mean_entropy overlaps massively (knowable range [0.178, 0.713] vs unknowable range [0.372, 0.892]).

**The judge CSV does not use max_entropy as a signal.** The experiment only tested mean_entropy, semantic_entropy, and spike_entropy. Rerunning with max_entropy as the signal would likely produce dramatically better citation-subset results.

---

## 11. Signal Convergence at High Budget

At 30% budget on OLMo's all-queries subset, all three signals produce identical results (41 incorrect out of 60 flagged). Signal differentiation only matters at tight budgets.

Similarly, Qwen non-citation results are identical across all three signals at 10% and 30% budget (6 and 12 incorrect respectively). The top 10% highest-entropy items are the same regardless of which entropy variant is used.

This suggests the three signals are measuring highly correlated quantities. At the tails (top 10%), they often flag exactly the same items.

---

## 12. Lift Efficiency

Lift per unit budget (efficiency) generally decreases with budget -- diminishing returns. But there are exceptions:

- **Qwen citations: efficiency INCREASES with budget** (0.486 -> 0.556 -> 0.670 for mean_entropy). With perfect AUC, every additional flagged item is correctly flagged, so no dilution occurs.

- **Qwen non-citation: lift DECREASES at 30%** (0.020 -> 0.039 -> 0.028). Qwen starts flagging correct answers because its 82.3% baseline leaves little headroom.

Headroom analysis at 30% budget (all queries): models capture 26-45% of possible improvement. Llama captures the most (44.8%), OLMo the least (26.1%). Mistral captures 35-40% despite having the most headroom (43% error rate).

---

## 13. Citation Sample Size Warning

The citation subset contains only 36 queries per model (2 knowable, 34 unknowable). AUC is computed on 36 data points with 2 positives and 34 negatives. This means:

- A single misranked knowable query can drop AUC by 0.5
- Precision at 10% budget is computed on n=4 flagged items
- The 95% confidence interval for an AUC of 0.68 at n=36 is roughly [0.45, 0.91]

All citation-subset findings should be treated as directional, not precise.

---

## 14. Anomalies and Surprises

1. **Self-report confidence is computed from response text, not model internals.** The fact that Llama, Mistral, and Qwen produce *identical* self-report values on 36/36 citation queries (and OLMo agrees on 34/36) is suspicious. This suggests the self-report is a deterministic function of response features (presence of hedging language, refusal markers), not a genuine confidence probe. The bimodal distribution (0.0 or 1.0, nothing between) confirms this.

2. **Query formatting determines fabrication vs refusal.** "Summarize: Authors. 'Title.' Venue, Year." triggers fabrication with self-report=1.0 across all models. "Describe Dr. X's work on Y" triggers refusal with self-report=0.0 across all models. The underlying content is the same (fabricated research); only the framing differs.

3. **Unknowable citation entropy is LOWER than unknowable non-citation entropy for Llama and OLMo.** Models are more entropy-certain when fabricating citations than when fabricating other content. The direction is reversed for Mistral. No consistent pattern across architectures.

4. **The cross-model correlation collapse on citations (rho=0.07) means each model fabricates differently.** On non-citation queries, models agree on difficulty. On citations, fabrication entropy is architecture-specific noise. This has implications for ensemble methods: averaging entropy across models would work for non-citations but not for citations.

5. **entropy_std on citations produces AUC 0.63-0.84** -- intermediate between mean_entropy (0.56-0.68) and max_entropy (0.82-0.97). The ordering is consistent: max > std > mean for citation discrimination. All three are nearly equivalent for non-citations.

6. **OLMo is the only model where the judge hurts on citations.** OLMo has baseline 16.7% on citations, and the judge makes it worse at most budget/signal combinations. OLMo also has the only sub-0.5 AUC (semantic_entropy = 0.368 on citations). OLMo's citation signal is genuinely broken.

---

## 15. Summary of Actionable Findings

- **max_entropy should replace or supplement mean_entropy for citation-heavy workloads.** The AUC gain is +0.21 to +0.41 on citations with no meaningful loss on other queries.
- **The bounded judge provides the most value for the worst models.** Mistral (57% baseline) gets the highest lift and perfect precision. Qwen (79% baseline) gets the lowest lift and worst precision.
- **Citation fabrication is a formatting phenomenon.** The formal citation format triggers confident confabulation; the informal format triggers refusal. This is a prompt engineering finding, not an architectural one.
- **Cross-model ensembles should be stratified by query type.** Entropy averaging works for non-citations (rho=0.8) but is meaningless for citations (rho=0.07).
- **Sample sizes for citation-only analysis (n=36, with 2 knowable) are too small for firm conclusions.** All citation-subset findings need replication with a larger balanced set.
