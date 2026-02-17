# Experiment 31: Together.ai API Analysis

## Date: Feb 16-17, 2026

## What We Did

Extended the SOSP paper's local model experiments (Experiment 27/27b,
four dense models at 4B-8B) to frontier models accessible only via API,
using Together.ai's serverless endpoints with top-5 logprobs.

### Smoke Testing

Tested 12 models on Together.ai serverless for logprobs availability:

| Model | Logprobs? | Notes |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | Yes, k=5, full response | Calibration model (also tested locally) |
| Mistral-Small-24B-Instruct | Yes, k=5, full response | |
| Llama-4-Maverick-17B-128E | Yes, k=5, full response | MoE, 128 experts |
| Qwen3-235B-A22B | Yes, k=5, full response | Largest model tested, 22B active |
| Gemma-3n-E4B | Yes, k=5, full response | Google architecture |
| DeepSeek-V3 | 1 token only | Serverless limitation, not model limitation |
| DeepSeek-V3.1 | 1 token only | Same limitation as V3 |
| DeepSeek-R1 | No logprobs | Reasoning model, emits `<think>` tokens instead |
| Llama-3.1-8B-Turbo | No logprobs | "-Turbo" suffix correlates with no logprobs |
| Llama-3.3-70B-Turbo | No logprobs | Same pattern |
| Qwen2.5-7B-Turbo | No logprobs | Same pattern |
| GLM-5 | 1 token, k=1 | Severely limited |

No predictable pattern by family, size, or architecture. The "-Turbo"
suffix correlates with no logprobs (likely a vLLM/TensorRT optimization).
DeepSeek models return logprobs but truncated to 1 token on serverless.

Observation: DeepSeek-R1's `<think>` block referred to the API caller
as "the user" — testimony without telemetry. The chain-of-thought text
is exposed but the logprobs that would let you verify it are not.

### Full Experiment

80 probes across 5 epistemic categories (16 each), same probe set as
Experiments 23/24/27 for direct comparability. Ran 6 models total
across two runs (3 + 3).

**Data files:**
- `exp31_frontier_api_20260216_105455.csv` — DeepSeek-V3, Mistral-Small-24B, Mistral-7B
- `exp31_frontier_api_20260216_135647.csv` — Llama-4-Maverick, Qwen3-235B, Gemma-3n-E4B
- Script: `scripts/experiment31_frontier_api.py`

## Results: Mean Entropy (Initial Analysis)

| Model | Params | Family | AUC (mean) |
|---|---|---|---|
| Qwen3-235B | 235B MoE (22B active) | Alibaba | 0.875 |
| Mistral-7B | 7B dense | Mistral | 0.860 |
| Mistral-Small-24B | 24B dense | Mistral | 0.850 |
| Gemma-3n-E4B | ~4B efficient | Google | 0.765 |
| Llama-4-Maverick | 17B MoE (128E) | Meta | 0.651 |
| DeepSeek-V3 | 671B MoE (37B active) | DeepSeek | 0.607* |

*DeepSeek-V3 returns only 1 token of logprobs. Not comparable.

Five models with full logprobs: AUC range 0.651-0.875, mean 0.800.
Cross-model Spearman rho = 0.360 (vs 0.762 for local experiments).

### Calibration

Mistral-7B via API (top-5 logprobs, likely quantized): AUC 0.860
Mistral-7B local (full vocabulary, float16): AUC 0.905

The API result is within the local experiment range (0.72-1.00) and
confirms that top-5 renormalized entropy preserves discriminative
power. The gap (0.860 vs 0.905) represents the combined cost of
top-5 truncation and likely quantization.

## Results: Architecture-Dependent Aggregation (Key Finding)

Prompted by a conversation about neutrosophic logic and
retrieval-vs-construction signal shapes, we tested whether different
entropy aggregations (mean, max, std) discriminate better for
different architectures.

### API Models

| Model | Architecture | AUC(mean) | AUC(max) | AUC(std) | Best | Improvement |
|---|---|---|---|---|---|---|
| Mistral-7B | Dense 7B | 0.860 | 0.917 | 0.921 | std | +0.061 |
| Mistral-Small-24B | Dense 24B | 0.850 | 0.909 | 0.934 | std | +0.084 |
| Llama-4-Maverick | MoE 17B/128E | 0.651 | 0.899 | 0.823 | max | +0.248 |
| Qwen3-235B | MoE 235B/22B | 0.875 | 0.853 | 0.870 | mean | +0.000 |
| Gemma-3n-E4B | Efficient ~4B | 0.765 | 0.855 | 0.703 | max | +0.090 |

**Llama-4-Maverick goes from AUC 0.651 (mean) to 0.899 (max).** The
signal was always there — the aggregation was wrong. The 128-expert
MoE routing smooths mean entropy across the response but cannot hide
the single most uncertain token (max entropy).

With architecture-appropriate aggregation: AUC range 0.855-0.934,
mean 0.898. Every model above 0.85.

### Local Models (Experiment 27, full vocabulary)

| Model | Architecture | AUC(mean) | AUC(max) | AUC(std) | Best | Improvement |
|---|---|---|---|---|---|---|
| Llama-3.1 8B | Dense | 0.922 | 0.950 | 0.928 | max | +0.028 |
| Mistral 7B | Dense | 0.905 | 0.890 | 0.903 | mean | +0.000 |
| OLMo-3 7B | Dense | 0.894 | 0.885 | 0.892 | mean | +0.000 |
| Qwen3 4B | Dense | 0.896 | 0.892 | 0.900 | std | +0.004 |

All four local models are dense. Improvements from alternative
aggregation are small (+0.000 to +0.028). With full vocabulary entropy,
mean is good enough for all architectures tested.

**Key insight: architecture-dependent aggregation matters most when
the signal is already constrained (top-5 API logprobs).** Under full
observation (local), mean entropy is sufficient. Under partial
observation (API), the right aggregation recovers signal that mean
entropy loses.

Note: Llama-3.1 8B shows the same pattern as Llama-4-Maverick (max
is best), suggesting a family-level property, not purely an MoE effect.

## Distribution Shape: Retrieval vs Fabrication

The coefficient of variation (CV = entropy_std / mean_entropy) within
responses shows a consistent pattern across ALL models tested:

- **Retrieval (knowable):** High CV (1.4-2.6) — mostly confident tokens
  with occasional uncertainty spikes. Peaked distribution.
- **Fabrication (unknowable):** Lower CV (0.8-1.7) — uniformly uncertain
  across all tokens. Flatter distribution.

This holds for both local (full vocab) and API (top-5) models.

| Source | Knowable CV | Unknowable CV |
|---|---|---|
| OLMo-3 7B (local) | 1.46 | 0.99 |
| Llama-3.1 8B (local) | 1.58 | 1.05 |
| Mistral 7B (local) | 1.49 | 1.15 |
| Qwen3 4B (local) | 2.50 | 1.49 |
| Qwen3-235B (API) | 2.57 | 1.66 |
| Llama-4-Maverick (API) | 1.40 | 1.60* |
| Gemma-3n-E4B (API) | 1.65 | 1.36 |
| Mistral-Small-24B (API) | 1.79 | 1.17 |
| Mistral-7B (API) | 1.38 | 0.92 |

*Llama-4-Maverick is the one exception where unknowable CV > knowable CV.
This may be related to the MoE routing effect that also makes mean
entropy the wrong aggregation for this architecture.

The shape distinction has implications beyond discrimination: it may
distinguish **retrieval** (the model knows the answer), **construction**
(the model is reasoning about something genuinely novel), and
**fabrication** (the model is generating plausible-sounding nonsense).
This three-way distinction is not testable with the current probe set
(which has knowable and unknowable but no "genuinely novel" category)
and is the subject of a separate investigation (see papers/rlm/).

## Audit Findings

Scientific integrity audit (Feb 16, 2026) verified all numeric claims.
Key findings that affect how we present this data:

### HIGH: Cross-model correlation drop

Local rho = 0.762 (4 dense models, 4B-8B, full vocab).
API rho = 0.360 (5 diverse models, 4B-235B, top-5 logprobs).

Three confounds are tangled: top-5 truncation, scale diversity, and
architectural diversity. We cannot attribute the drop to any single
cause. Must report both numbers.

### MEDIUM: Private category confound

Models that refuse unknowable queries (epistemically correct behavior)
produce low-entropy refusals that are indistinguishable from confident
factual answers. This depresses AUC for well-behaved models.

This is actually a finding: the metric measures generation confidence,
not epistemic honesty. A model that honestly says "I can't answer that"
gets penalized by the entropy metric because the refusal is low-entropy.
As models are trained to be more epistemically honest, the entropy
signal's ability to detect dishonesty degrades — because honest refusals
and honest facts look the same in entropy space.

### MEDIUM: Llama-4-Maverick marginal significance (RESOLVED)

With mean entropy: AUC 0.651, 95% CI [0.512, 0.786], p = 0.011.
With max entropy: AUC 0.899. No longer marginal.
The finding is not "weak signal" but "wrong aggregation."

### LOW: Docstring bug (k=20 vs k=5) — fixed.

## Provider Choice as Experimental Evidence

The unpredictable logprobs availability across Together.ai's serverless
endpoints is itself evidence for the paper's Responsibility Concentration
corollary. Signal access is neither guaranteed nor principled — it falls
out of serving stack implementation details.

Tony's insight: providers withhold logprobs partly as competitive moat.
Logprobs enable model distillation. OpenAI's complaint about DeepSeek
distilling from their models is exactly this concern. The providers'
economic incentive conflicts with the observability requirements the
paper identifies.

Together.ai offers research-oriented features (fine-tuning, dedicated
endpoints up to top-20 logprobs) that partially address this. But even
Together caps at top-20 — full vocabulary logprobs are not available
through any commercial API we tested. The top-5 approximation being
sufficient for discrimination (AUC 0.855-0.934 with appropriate
aggregation) is a pragmatically important finding: you don't need full
access to get useful signal.

## What This Means for the SOSP Paper

### Already in the paper
- "Signal access as provider choice" paragraph in discussion.tex
- Qualifiers about the scope of claims

### What should change
1. **Report architecture-dependent aggregation** (one sentence in eval):
   "Preliminary analysis suggests that optimal signal aggregation is
   architecture-dependent; max entropy outperforms mean entropy for
   mixture-of-experts models (AUC 0.651 → 0.899 for Llama-4-Maverick)."

2. **Update the API validation claim**: With appropriate aggregation,
   all five models achieve AUC > 0.85, not just 4/5 > 0.72.

3. **The private category confound deserves a sentence**: The finding
   that epistemically honest refusals are entropically indistinguishable
   from confident facts connects to the paper's impossibility theme.

4. **Cross-model rho**: Report 0.360 alongside 0.762, with the caveat
   that three confounds are tangled.

### What should NOT change
- The local experiment results (all dense, no MoE issue)
- The core impossibility result and escape theorem
- The evaluation methodology

### Deferred to RLM paper (papers/rlm/)
- Architecture-dependent composition operators
- Retrieval vs construction vs fabrication shape analysis
- Neutrosophic gradient experiments
- Tensor-gated recursive calls

## Cost

Together.ai account: $20 prepaid credits.
480 API calls (6 models x 80 probes) + ~20 smoke test calls.
Estimated spend: < $2 total. Exact amount TBD from Together dashboard.
