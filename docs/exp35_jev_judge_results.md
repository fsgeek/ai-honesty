# Exp35 results: Jev as a text-channel judge on the exp27 responses

*2026-09-30, Claude Opus 5.5. Predictions: `docs/exp35_jev_judge_predictions.md`,
sha256 `438228f4…c545`, written before any call. 1,800 calls to `jev-1.13.0`,
0 failures, ~0.9M input tokens (about $0.04). Raw calls:
`results/exp35_jev_calls.jsonl`; scored rows: `results/exp35_jev_scored.csv`.*

## Pre-registered: AUC against `is_knowable` (the paper's target)

| Signal | pooled | Llama | Mistral | OLMo | Qwen |
|---|---|---|---|---|---|
| mean entropy | 0.870 | 0.922 | 0.905 | 0.894 | 0.896 |
| response word count | 0.912 | 0.949 | 0.969 | 0.887 | 0.852 |
| **query** word count | 0.871 | 0.871 | 0.871 | 0.871 | 0.871 |
| Jev, question only (Q) | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| Jev, question + response (QR) | 0.994 | 0.998 | 0.996 | 0.993 | 0.987 |
| Jev, response only (R) | 0.988 | 0.995 | 0.994 | 0.986 | 0.979 |

## Scorecard

| | Prediction | Outcome |
|---|---|---|
| P1 | Q lens >= 0.95 | **Held** (0.999). The label is visible in the question. |
| P2 | Query word count 0.75-0.85 | **Failed, high**: 0.871, equal to pooled entropy (0.870). |
| P3 | QR >= Q, and QR >= entropy in all four families | **Failed** on the first clause (0.994 < 0.999); second clause held. |
| P4 | R within 0.05 of response word count | **Failed**: 0.988 vs 0.912. Reading beats counting. |
| P5 | Jev probabilities bimodal | **Held**: 85% of mass below 0.1 or above 0.9. |
| P6 | Confidence monotone; top quintile off by > 0.1 | **Half**: monotone, but top quintile 0.958 conf vs 1.00 accuracy. |

## Exploratory (not predicted): the target the budget curve actually rewards

The budget curve pays off on **wrong answers**, not on unknowable questions.
237 of the 400 unknowable-query responses are *correct* (the model declined), so
`is_knowable` and "wrong" are different targets. Scoring against wrong answers
(`~is_correct`, from the artifact's `load_data`):

| Signal | pooled | Llama | Mistral | OLMo | Qwen |
|---|---|---|---|---|---|
| mean entropy | 0.841 | 0.831 | 0.891 | 0.807 | 0.844 |
| response word count | 0.774 | 0.747 | 0.931 | 0.718 | 0.682 |
| query word count | 0.732 | 0.615 | 0.813 | 0.778 | 0.700 |
| Jev Q, P(unanswerable) | 0.654 | 0.648 | 0.749 | 0.651 | 0.524 |
| Jev QR, 1 - P(grounded) | 0.762 | 0.728 | 0.864 | 0.764 | 0.625 |
| Jev R, 1 - P(grounded) | 0.872 | 0.813 | 0.936 | 0.883 | 0.794 |
| **Jev QR, P(fabricated)** | **0.983** | 0.966 | 0.989 | 0.987 | 0.974 |
| Jev R, P(fabricated) | 0.974 | 0.943 | 0.986 | 0.978 | 0.966 |

Budget curve (artifact `fig3_budget_curve.py`, same Monte Carlo, seed 42, 1000
trials; `results/exp35_budget_curve.txt`):

| Judge | 10% | 20% | 30% |
|---|---|---|---|
| no judge | 75.8 | 75.8 | 75.8 |
| length (paper) | 78.4 | 82.2 | 87.6 |
| entropy (paper) | 81.7 | 86.8 | 90.9 |
| query word count | 81.1 | 85.3 | 87.4 |
| Jev Q | 76.9 | 78.7 | 82.1 |
| Jev R, 1 - P(grounded) | 83.4 | 89.4 | 92.6 |
| **Jev QR, P(fabricated)** | **85.1** | **93.6** | **97.6** |

The P(fabricated) scoring was chosen after seeing the 1 - P(grounded) results,
so treat its margin as optimistic. It is still the natural score for the
question the curve asks.

## What this means

1. **The paper's AUC target is mismatched with its budget curve.** AUC against
   `is_knowable` rewards flagging unknowable *questions*; the curve rewards
   flagging wrong *answers*. The question-only judge shows the gap: 0.999 on the
   first, worst judge on the second.
2. **The exp27 benchmark's label is readable from the question.** Question length
   alone matches entropy's pooled AUC against `is_knowable`. Entropy AUC 0.87
   should not be read as "detects unknowability" on this set.
3. **A text judge that did not exist when the paper was written beats entropy.**
   The PACMI comparison was against word count, the text baseline available then,
   and that result stands. Jev (released 2026-09-15) beats entropy at every
   budget, in every family, by 3-7 points. This is a starting point for new work,
   not a correction to the paper.
4. **Hypothesis (not measured here): this locates the theorem rather than
   contradicting it.** The theorem needs the supervisor's observations to be
   identical across worlds. One explanation for Jev's result is that Jev knows
   more than a 4-8B generator, so the small models' fabrications are visible to
   it in text. Exp35 does not test that. We never measured what Jev knows
   relative to the generators, and other explanations fit the same numbers
   (for example, fabrications on this benchmark may carry surface markers any
   reader can see). If the hypothesis holds, text judges work when the judge
   knows more than the generator, and the case for telemetry is when no
   stronger judge exists. A test that needs no Jev: have each generator judge
   its own responses from text and compare the result with its own entropy.

## Caveats

- 430 of the 800 correctness labels came from Qwen3-4B, itself a text judge;
  the 75/80 blinded human review is the anchor. Jev agreeing with a text-judge
  label is partly two text readers agreeing.
- One Jev version, one wording per lens, one run. Levadura's repeatability
  measurements suggest per-call noise is small, but it was not re-measured here.
- The benchmark is the paper's, not an adversarial one. None of this says what
  happens when the generator is stronger than the judge.
