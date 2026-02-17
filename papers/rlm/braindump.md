# RLM Paper: Tensor-Augmented Recursive Language Models

## Brain Dump — Feb 17, 2026

This document captures the current state of thinking. It is not an outline.
It is a pile of observations, connections, and hypotheses that a future
session should organize into a paper structure.

## Origin

Zhang/Kraska/Khattab (MIT CSAIL, Dec 2025): "Recursive Language Models."
Key finding: LLM outputs can feed into subsequent LLM calls for unbounded
context through recursive decomposition. But even at depth 1, the model
builds a correct answer through sub-calls, then ignores it and fabricates
from scratch. The failure mode is confident fabrication with a clean
envelope — the Westphalia class from the SOSP paper.

The SOSP paper (epistemic observability) explains WHY text-only recursion
degrades: the impossibility result applies to inter-model communication,
not just human-model communication. A model at depth N has no way to know
whether the output from depth N-1 was confident knowledge or confident
fabrication. Text-only boundaries discard epistemic state.

This paper asks: what happens when you DON'T discard it?

## Core Thesis

The tensor interface — entropy, attention topology, activation patterns,
and other signals extractable during inference — provides the missing
primitive for stable recursive language models. Composition boundaries
that carry epistemic metadata can gate fabrication propagation, distinguish
retrieval from construction, and adapt to the architecture of the model
doing the work.

## Key Findings Already in Hand (from SOSP Experiment 31)

### 1. Architecture-Dependent Signal Aggregation

Different model architectures express epistemic uncertainty through
different summary statistics of the token-level entropy distribution:

| Architecture | Best Signal | AUC | Why |
|---|---|---|---|
| Dense (Mistral-7B) | entropy_std | 0.921 | Variance across tokens captures uncertainty pattern |
| Dense (Mistral-Small-24B) | entropy_std | 0.934 | Same pattern scales with model size |
| MoE 128E (Llama-4-Maverick) | max_entropy | 0.899 | Expert routing smooths mean; peak spike survives |
| Efficient (Gemma-3n-E4B) | max_entropy | 0.855 | Similar to MoE pattern |
| Large MoE (Qwen3-235B) | mean_entropy | 0.875 | Well-calibrated; all aggregations work |

Mean entropy alone gave AUC 0.651 for Llama-4-Maverick. Max entropy
recovers it to 0.899. The signal was always there — the aggregation
was wrong. This means a composition gate that uses a fixed aggregation
will fail for some architectures. The gate must be architecture-aware.

### 2. Distribution Shape Distinguishes Retrieval from Fabrication

From the coefficient of variation (CV) analysis:
- **Retrieval** (control category): High CV — mostly confident tokens
  with occasional uncertainty spikes. Peaked distribution.
- **Fabrication** (glavinsky/westphalia): Lower CV — uniformly uncertain
  across all tokens. Flat distribution.
- **Construction** (not yet measured): Hypothesized to fall between these,
  with clustered competition among a few strong candidate tokens.

This is the key insight for RLM: the SHAPE of the entropy distribution
within a response carries information about the generative process, not
just the confidence level. A gate that reads shape can distinguish:
- "I retrieved this" (trust it)
- "I constructed this" (verify it)
- "I fabricated this" (block it)

### 3. Provider Choice as Deployment Constraint

Tested 12 models on Together.ai serverless. Only 6 returned logprobs.
No pattern by family, size, or architecture — it's a per-deployment
implementation detail. This constrains which models can participate
in tensor-gated recursive chains. A practical RLM system must handle
mixed-signal environments where some models expose logprobs and others
don't.

## Proposed Experiments

### Experiment A: The Neutrosophic Gradient

A natural test for the retrieval→construction transition. Questions at
four depths of training data familiarity:

1. Boolean logic properties — saturated in training data
2. Fuzzy logic, Dempster-Shafer — well-represented
3. Neutrosophic logic basics (Smarandache, T/I/F) — sparse
4. Functional T/I/F with tensor composition — essentially novel

Collect per-token logprobs at each depth. Hypothesis: logprob distribution
shape shifts measurably as the model moves from retrieval to construction.
Cross-model comparison reveals which parts of the question space are
structurally novel versus training-data-absent for a specific model.

### Experiment B: Tensor-Gated Recursive Calls

Implement the composition boundary from the SOSP paper's cut section
(cut_tensor_composition.tex) but with architecture-appropriate gating:
- For dense models: gate on entropy_std
- For MoE models: gate on max_entropy
- For well-calibrated large models: gate on mean_entropy

Compare recursive chain reliability (correct answer retention across
depths) with and without tensor gating. The RLM paper's failure mode
(model builds correct answer, then ignores it) should be detectable
and preventable.

### Experiment C: Cross-Architecture Recursive Chains

RLM assumes a single model recurses on itself. But composition can
cross architectures: Qwen3-235B generates, Mistral-7B verifies,
Llama-4 synthesizes. Each step produces tensor metadata. The composition
operator must handle heterogeneous signals — entropy_std from Mistral,
max_entropy from Llama-4, mean_entropy from Qwen3.

This is where Smarandache's under-specification becomes engineering:
you can't pre-define the composition rules because the right rule
depends on which models are composing. The composition operator is
a function of the architecture graph, not a fixed truth table.

### Experiment D: Calibration Across the Gradient

Use models available both locally (full vocabulary) and via API (top-5
logprobs) as calibration bridges. Mistral-7B is one such bridge (SOSP
Experiment 31 demonstrated AUC 0.860 API vs 0.72-1.00 local). Extend
to additional calibration models to quantify the top-k approximation
error as a function of model architecture and query type.

## Connections

### To Smarandache / Neutrosophic Logic
If T/I/F are functions (not scalars), composition is function composition.
The rules emerge from the structure of the functions themselves. The
tensor's indeterminacy value (I) can be partially grounded in the
generation process — the token distribution demonstrates uncertainty
rather than the model self-reporting it. This is the difference between
"I think I'm uncertain" and "the token distribution shows uncertainty."

### To the SOSP Paper
This paper takes the SOSP impossibility result as given and builds on
the escape. The SOSP paper proves you need the tensor interface. This
paper shows what you can do with it once you have it. The SOSP paper's
Theorem 2 (verifiability under tensor observation) is the foundation.
The RLM paper's contribution is the architecture-dependent composition
operator that makes Theorem 2 operational across recursive boundaries.

### To the Yanantin / Apacheta Tensor Database
Tensor-gated recursive chains produce sequences of tensors. Each
composition step generates metadata: which model, what entropy profile,
what gating decision, what the downstream call did with the result.
This is provenance data. The Apacheta tensor database (from the
Yanantin project) is designed to store exactly this kind of immutable,
authored, non-commutatively-composable epistemic metadata.

### To Provider Choice / API Economics
If tensor-gated recursion demonstrably improves reliability, providers
have an engineering incentive (not just a compliance incentive) to expose
logprobs. "Your recursive system breaks without this signal" is a
stronger argument than "regulators might want you to." The RLM failure
mode is their problem too — it degrades the quality of their agentic
products.

## What We Don't Know Yet

- Does the retrieval→construction shape distinction hold empirically,
  or is it an artifact of our probe categories?
- What is the right composition operator for heterogeneous model chains?
- Does tensor gating degrade performance on correct recursive chains
  (false positive rate of the gate)?
- How does the approximation error from top-k logprobs interact with
  gating decisions at composition boundaries?
- Is there a phase transition in the neutrosophic gradient, or is the
  retrieval→construction shift continuous?
- Can expert routing weights in MoE models provide a better signal
  than token-level entropy?

## Venue

ML venue (NeurIPS, ICML, or ICLR). The contribution is an architectural
primitive and empirical evidence that it works, not a systems design.
The SOSP paper provides the theoretical foundation; this paper provides
the ML application.

## Status

Brain dump. No experiments beyond what SOSP Experiment 31 already
produced. The finding that architecture-dependent aggregation recovers
Llama-4's signal (0.651 → 0.899) was discovered during SOSP data
analysis on Feb 16-17, 2026, and is the catalyst for this paper.
