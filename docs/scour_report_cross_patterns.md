# Cross-Domain Pattern Scour Report

**Date:** 2026-02-11
**Data sources:** `exp27c_traces_20260211_191049.jsonl` (800 text traces), `exp27c_results_20260211_191049.csv` (800 evaluations), `exp27c_citation_judge_results.csv` (bounded judge results), `code_entropy_traces_20260211_200415.jsonl` (15 code traces)
**Models:** OLMo-3-7B-Instruct, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct, Mistral-7B-Instruct-v0.3 (text); Qwen3-4B-Instruct (code)

---

## Finding 1: The Entropy Inequality Gradient Is a Continuum, Not a Binary

The Gini coefficient of per-token entropy distributions forms a monotonic gradient across domains:

| Domain | Gini | Top-10% Share | Near-Zero (< 0.001) | Mean Entropy |
|--------|------|---------------|----------------------|--------------|
| Code (Qwen) | 0.927 | 92.6% | 76.4% | 0.058 |
| Knowable text (Qwen) | 0.837 | 80.3% | 47.4%* | 0.094 |
| Knowable text (all) | 0.714 | 62.4% | 30.4% | 0.237 |
| Unknowable text (Qwen) | 0.712 | 47.3%* | 35.9%* | 0.315 |
| Unknowable text (all) | 0.606 | 37.1% | 20.6% | 0.609 |

*Asterisked values computed from family-specific breakdowns.*

This is the format-constraint manifold made visible in a new dimension. The ordering (code > knowable > unknowable) holds across every metric: Gini coefficient, entropy concentration, near-zero token fraction, mean entropy. Code extends the manifold endpoint beyond "factual text" into "maximally format-constrained content."

**Why this matters:** The earlier observation (`entropy_code_observations.md`) noted that BPE scaffolding is only 11-19% of code tokens. But the entropy data shows that 76.4% of code tokens have near-zero entropy -- far more than the syntactic scaffolding percentage. The excess comes from "semantic scaffolding" (conventional names like `left`, `right`, `mid`), confirming that concept quantitatively. For familiar algorithms, conventions are as constraining as grammar.

---

## Finding 2: Natural Language Inside Code Carries 60% of the Entropy from 31% of the Tokens

Separating code traces into docstrings, comments, and pure code:

| Region | Token % | Entropy Share | Mean Entropy | Spike Rate (> 0.1) |
|--------|---------|---------------|--------------|---------------------|
| Docstrings | 22.1% | 29.3% | 0.077 | 16.0% |
| Comments | 9.2% | 30.8% | 0.194 | 33.9% |
| Pure code | 68.7% | 39.9% | 0.034 | 6.5% |

Comments are 9.2% of tokens but carry 30.8% of entropy. Their mean entropy (0.194) is 5.7x higher than pure code (0.034). Comments have a spike rate (33.9%) that is 5.2x the pure code spike rate (6.5%).

**The surprising connection:** Comments in code are natural language embedded in a format-constrained medium. Their entropy profile (mean 0.194, spike rate 33.9%) is structurally between knowable text (mean 0.094 for Qwen) and unknowable text (mean 0.315 for Qwen). The model is *more uncertain about what to say about code than about the code itself*. This is not a noise artifact -- it tracks the fact that there are many valid ways to describe an algorithm, but fewer valid implementations of it.

**Implication for the bounded judge:** If Tensor@10% is applied to code, most flagged tokens will be in comments and docstrings, not in executable code. The bounded judge for code should focus on the natural-language portions, not the syntax.

---

## Finding 3: Code Is Not "Super-Knowable" -- It Is Structurally Different

Adding code traces as "knowable" to the Qwen text dataset improves AUC from 0.896 to 0.909. But the improvement is small, and 49.0% of knowable text has *higher* entropy than the code median. Code is not simply at one end of the text entropy scale.

The structural difference is in the **trajectory**. Normalized entropy over the course of a response:

| Position (decile) | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
|--------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| Code | 0.91 | 1.45 | 1.10 | 0.85 | 1.18 | 1.10 | 0.90 | 1.30 | 0.48 | 0.75 |
| Knowable text | 0.87 | 0.75 | 0.95 | 1.00 | 1.05 | 1.06 | 1.14 | 0.96 | 1.02 | 1.25 |
| Unknowable text | 0.49 | 0.65 | 0.87 | 0.98 | 1.09 | 1.21 | 1.23 | 1.21 | 1.20 | 1.10 |

Code entropy is **flat** (oscillates without trend). Knowable text **rises gently then spikes at end**. Unknowable text has a **monotonic ramp-up** that plateaus late.

The flat trajectory of code means spikes in code are position-independent -- they can occur anywhere. In unknowable text, uncertainty accumulates. This is the difference between format-constrained generation (where each token is locally determined) and open-ended generation (where each token choice constrains future choices, building up uncertainty).

---

## Finding 4: The Length Confound Halves the Entropy Ratio But Does Not Kill It

Knowable responses average 32.7 tokens; unknowable responses average 111.6 tokens (3.4x longer). The entropy ratio (unknowable/knowable) drops substantially when controlling for length:

| Model | Uncontrolled Ratio | Length-Controlled Ratio | Ratio of Ratios |
|-------|-------------------|------------------------|-----------------|
| OLMo | 2.43x | 1.33x | 0.55 |
| Llama | 3.12x | 1.97x | 0.63 |
| Qwen | 3.36x | 1.47x | 0.44 |
| Mistral | 2.06x | 1.30x | 0.63 |

**The entropy discrimination survives length control** (all ratios > 1.0), but the raw numbers overstate it by 1.6-2.3x. The length confound is particularly strong for Qwen (ratio drops by 56%).

**Connection to code:** Code responses (94-512 tokens) overlap with both knowable and unknowable text lengths, yet their mean entropy (0.058) is far below either. This confirms that the code signal is not a length artifact.

**Implication:** Papers reporting entropy-based AUC should report length-controlled AUC alongside raw AUC. The current paper's AUC numbers (0.89-0.92) include some length contribution.

---

## Finding 5: Citation-Only AUC Collapses for Most Models

The citation judge results reveal a dramatic subset effect:

| Signal | All Queries AUC | Citation-Only AUC | Non-Citation AUC |
|--------|----------------|-------------------|------------------|
| mean_entropy | 0.904 | 0.713 | 0.910 |
| semantic_entropy | 0.916 | 0.702 | 0.920 |
| spike_entropy | 0.910 | 0.757 | 0.913 |

The AUC drop on citation-only queries is severe: from 0.90+ to 0.70. But the pattern is not uniform across models:

- **Qwen: AUC = 1.000 on citations** (all signals, all budgets)
- **OLMo: AUC = 0.368 on citations** (semantic_entropy) -- *below chance*
- **Llama: AUC = 0.559-0.691 on citations** -- weak
- **Mistral: AUC = 0.676-0.794 on citations** -- moderate

Qwen achieves perfect citation discrimination while OLMo fails completely. This is a 0.63 AUC gap between models on the same task. The earlier finding that entropy-based discrimination is "architectural, not model-specific" holds for overall queries but **fails for citations**. Citation discrimination is training-procedure-specific.

**Connection to code:** Citations and code are both format-constrained domains. But citation AUC varies wildly across models while overall text AUC is stable. The format-constraint manifold may need a model-specific dimension: some models handle format-constrained content differently than others.

---

## Finding 6: Qwen Is an Outlier on Every Dimension

Qwen3-4B-Instruct behaves differently from the other three models:

| Metric | OLMo | Llama | Qwen | Mistral |
|--------|------|-------|------|---------|
| Near-zero token % (knowable) | 29.8% | 37.0% | **70.2%** | 32.4% |
| Near-zero token % (unknowable) | 12.8% | 13.1% | **36.6%** | 21.1% |
| Code-like text traces (top10% > 80%) | 22.7% | 14.7% | **61.3%** | 16.7% |
| Mean entropy (knowable) | 0.325 | 0.227 | **0.094** | 0.304 |
| Citation AUC (mean_entropy) | 0.618 | 0.559 | **1.000** | 0.677 |
| AUC via near-zero fraction | 0.697 | 0.818 | **0.864** | 0.698 |

Qwen generates text where most tokens have near-zero entropy, producing a code-like entropy profile even for text. Its near-zero fraction gap between knowable (70.2%) and unknowable (36.6%) is 33.6 percentage points -- wider than any other model. This may explain its perfect citation AUC: when citations appear in Qwen outputs, their entropy stands out sharply against the near-zero background.

**The mechanism is different.** For OLMo/Llama/Mistral, discrimination comes from mean entropy differences. For Qwen, it comes from the fraction of tokens that are near-zero. The near-zero AUC (0.864) is the highest of any model, while its mean entropy AUC (0.896) is the lowest. Qwen discriminates by *how many tokens are certain*, not by *how uncertain the average token is*.

---

## Finding 7: Positional Entropy Inversions Are Widespread and Structurally Organized

At specific token positions within responses, knowable text has HIGHER entropy than unknowable text -- the opposite of the aggregate signal.

| Model | Inverted Positions (out of 50) | Fraction |
|-------|-------------------------------|----------|
| OLMo | 12 | 24% |
| Llama | 4 | 8% |
| Qwen | 18 | 36% |
| Mistral | 12 | 24% |

These inversions are not random noise. In Qwen, they cluster at positions 10-20 and 23-26. The pattern: knowable responses reach their answer by position 10-15 and generate high-entropy padding (elaboration, formatting), while unknowable responses are still in their low-entropy preamble ("I think that..." "Based on my knowledge...").

This is the positional analog of the self-report inversion: early in a response, unknowable text is more uncertain. Late in a response, knowable text has finished its confident answer and is generating uncertain filler, while unknowable text is still producing its detailed fabrication.

**Connection to code:** Code has NO such positional structure -- its normalized trajectory is flat. Positional inversions are a property of natural language generation with variable-length answers, not of format-constrained generation.

---

## Finding 8: Self-Report Confidence Is Uncorrelated with Entropy Within Category

If self-report captured any genuine epistemic signal, it should correlate (negatively) with entropy within the unknowable category: responses where the model "knows" it is uncertain should have higher entropy. The data:

| Model | r(self_report, mean_entropy) within unknowable |
|-------|-------------------------------------------------|
| OLMo | +0.037 |
| Llama | -0.154 |
| Qwen | -0.063 |
| Mistral | +0.128 |

No model exceeds |r| = 0.16. Self-report confidence is essentially uncorrelated with the tensor signal within category. The self-report inversion (higher confidence on fabrications) is a between-category phenomenon: the model reports high confidence on unknowable topics because its training rewards confident-sounding text. But within either category, self-report adds no information that entropy does not already capture.

---

## Finding 9: Hedging Reduces Entropy in Unknowable Responses

Within unknowable responses, higher hedge scores are associated with LOWER entropy:

| Model | High-Hedge Mean Entropy | Low-Hedge Mean Entropy | Ratio |
|-------|------------------------|------------------------|-------|
| OLMo | 0.768 | 0.797 | 0.96 |
| Llama | 0.643 | 0.764 | 0.84 |
| Qwen | 0.287 | 0.366 | 0.78 |
| Mistral | 0.525 | 0.664 | 0.79 |

Hedging phrases ("I think," "it's possible that") are formulaic -- they are semantic scaffolding for uncertainty, just as `left = 0` is semantic scaffolding for binary search. The model is MORE certain about HOW to express uncertainty than about the content it is uncertain about.

This connects to code: hedging in text plays the same structural role as comments in code. Both are natural-language regions embedded in content that has different entropy characteristics. Both reduce local entropy while the surrounding context has higher uncertainty.

---

## Finding 10: Early-Token AUC Varies Dramatically Across Models

How many tokens do you need for reliable discrimination?

| Model | First-3 AUC | First-5 AUC | First-10 AUC | First-20 AUC | All-Token AUC |
|-------|------------|------------|-------------|-------------|---------------|
| OLMo | 0.716 | 0.681 | 0.656 | 0.733 | 0.894 |
| Llama | 0.830 | 0.859 | **0.895** | 0.848 | 0.922 |
| Qwen | 0.707 | 0.820 | 0.820 | 0.754 | 0.896 |
| Mistral | 0.637 | 0.653 | 0.587 | 0.598 | 0.905 |

Llama achieves 0.895 AUC with just 10 tokens (97% of full-response AUC). Mistral needs all tokens -- its early-token AUC (0.587-0.653) is barely above chance.

**Connection to the format-constraint theory:** Llama's early discrimination means its first few tokens already carry the epistemic signal. Mistral's failure means its signal is distributed across the whole response. This connects to the trajectory analysis: Mistral's entropy discrimination requires the full ramp-up pattern. A Tensor@3% budget on Llama would work; on Mistral it would fail.

---

## Finding 11: The Shared-Token Entropy Gap

The same BPE tokens appear in both code and text with dramatically different entropy profiles. For tokens appearing 3+ times in code and 5+ times in Qwen text:

| Token | Code Mean Entropy | Text Mean Entropy | Text/Code Ratio |
|-------|-------------------|-------------------|-----------------|
| `"2"` | 0.000000 | 0.016 | 2,017,442x |
| `"**"` | 0.000016 | 0.280 | 17,271x |
| `" while"` | 0.000101 | 0.209 | 2,072x |
| `" result"` | 0.012 | 0.546 | 45x |
| `" self"` | 0.024 | 0.546 | 23x |
| `" the"` | 0.161 | 0.322 | 2.0x |
| `" of"` | 0.055 | 0.032 | 0.6x |

The token `"2"` has entropy 0.000000 in code (it is completely determined by context -- binary search step, loop bound) but entropy 0.016 in text (where "2" competes with other numbers). Even common English words like `" the"` (0.161 in code vs 0.322 in text) carry less entropy in code because code context constrains their usage more tightly.

The exception: `" of"` is LOWER entropy in text (0.032) than in code (0.055). In text, "of" follows predictable constructions ("the capital of," "the speed of"). In code, "of" appears in comments and docstrings where its context is less constrained.

---

## Synthesized Insight: Three Types of Scaffolding

The cross-domain analysis reveals three distinct types of scaffolding, each with a different entropy signature:

1. **Syntactic scaffolding** (code operators, punctuation, keywords): entropy ~ 0.000. Determined by grammar. 16% of code tokens.

2. **Semantic scaffolding** (conventional variable names, format templates, hedging phrases): entropy ~ 0.001-0.01. Determined by training-data convention. 60%+ of code tokens, 30-47% of knowable text tokens.

3. **Content tokens** (actual assertions, novel names, specific claims): entropy > 0.01. The only tokens where the model makes a genuine generative "decision."

The format-constraint manifold from `domain_exploration_fourth_point.md` predicted that scaffolding ratio would predict entropy behavior. The data confirms this but adds nuance: it is not scaffolding *percentage* alone but the *type distribution* of scaffolding that matters. Code has all three types. Knowable text has mostly type 2 and 3. Unknowable text has mostly type 3.

The bounded judge should target type-3 tokens exclusively. For code, this means examining the 6.5% of pure-code tokens that spike plus the 33.9% of comment tokens that spike. For text, the entire response must be scanned because type-3 tokens are distributed throughout. The format-constraint gradient predicts the judge budget: **higher scaffolding ratio permits lower inspection budget**.

---

## What Contradicts or Complicates the Theory

1. **The positional inversion problem.** The format-constraint theory assumes more constraint = more reliable signal. But positions 10-20 in Qwen knowable text show HIGHER entropy than unknowable text. The mean-entropy signal works despite these local inversions, but any position-specific analysis would be misled. The trajectory matters more than any single position.

2. **Qwen breaks the uniformity assumption.** The paper claims tensor signals are "architectural, not model-specific." The cross-model correlation (Spearman rho = 0.762) supports this for overall queries. But citation-only AUC ranges from 0.368 (OLMo) to 1.000 (Qwen) -- a model-specific effect. The uniformity claim needs qualification: it holds for the aggregate signal but fails for domain-specific subsets.

3. **The length confound is not negligible.** Length-controlled ratios (1.30-1.97x) are 44-55% of uncontrolled ratios (2.06-3.36x). Some of the reported AUC comes from the model generating longer responses when it is uncertain, not from per-token uncertainty per se. This does not invalidate the signal (length is itself informative), but it means raw AUC overstates the per-token contribution.

4. **Comments in code are a domain hybrid.** The format-constraint manifold assumes each domain sits at one point. But code has TWO domains interleaved: pure code (Gini ~0.95, mean entropy ~0.034) and comments (Gini ~0.7, mean entropy ~0.194). Any domain with embedded natural language creates a mixture that cannot be characterized by a single scaffolding ratio.

5. **Early-token AUC is model-specific.** If the signal were purely architectural, early-token AUC should be similar across models. Instead, Llama gets AUC 0.895 from 10 tokens while Mistral gets 0.587. The signal distribution within responses is training-procedure-dependent, complicating the claim that tensor inspection can be done at a fixed budget.

---

## Files Referenced

- `/home/tony/projects/ai-honesty/exp27c_traces_20260211_191049.jsonl` -- 800 per-token traces
- `/home/tony/projects/ai-honesty/exp27c_results_20260211_191049.csv` -- 800 evaluation results
- `/home/tony/projects/ai-honesty/exp27c_citation_judge_results.csv` -- citation bounded judge
- `/home/tony/projects/ai-honesty/code_entropy_traces_20260211_200415.jsonl` -- 15 code traces
- `/home/tony/projects/ai-honesty/docs/entropy_code_observations.md` -- earlier code entropy analysis
- `/home/tony/projects/ai-honesty/docs/domain_exploration_fourth_point.md` -- format-constraint manifold theory
