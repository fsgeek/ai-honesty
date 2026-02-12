# ML Terminology Audit for SOSP 2026 Submission

**Paper:** "Epistemic Honesty in Predictive Systems: An Impossibility Result"
**Target audience:** Systems researchers (SOSP 2026)
**Date:** 2026-02-11

This audit identifies ML-specific terminology in the paper that may be unfamiliar
to the SOSP audience and categorizes each term by urgency of definition.

---

## Section A: Terms that MUST Be Defined

These terms are central to the paper's argument and a systems researcher likely
will not know them without explanation. Each needs a definition in the text (a
sentence or two, ideally with a systems analogy).

### 1. Hallucination / Confabulation / Fabrication

- **First appears:** Abstract ("fabrications"), Introduction line 10 ("hallucinate")
- **Issue:** The paper uses "hallucinate" in scare quotes on first use and later
  switches between "hallucination," "fabrication," and "confabulation" without
  formally distinguishing them. A systems reader's instinct for "hallucination"
  is hardware fault (e.g., phantom memory reads). The paper clearly treats
  hallucination as *fluent fabrication*, but never defines it in one crisp
  sentence. The `\cite{xu2024hallucination}` impossibility result is referenced
  but the reader needs to know *what* is being proven impossible.
- **Suggested definition:** "When a language model generates plausible but
  factually incorrect output -- inventing citations, people, or events that do
  not exist -- the ML community calls this *hallucination*. We prefer the term
  *fabrication* to emphasize that this is the system's expected behavior under
  coherence optimization, not a pathological failure."
- **Note:** The paper already gestures at this distinction ("implies pathology:
  a glitch in an otherwise functional system") but never lands on an explicit
  definition. One sentence would close the gap.

### 2. Token / Tokenization

- **First appears:** Introduction line 63 ("predictive token generation systems"),
  Design section ("per-token entropy"), Evaluation ("token entropy")
- **Issue:** To a systems reader, "token" means an authentication credential or
  a mutex token in a token ring. The paper uses "token" to mean a subword unit
  of text -- the atomic unit that the model processes and generates. The phrase
  "predictive token generation systems" is the paper's central abstraction and
  the reader must understand what a token is for the entropy trace to make sense.
- **Suggested definition:** "A *token* is the atomic unit of text that a language
  model processes. Modern models decompose text into subword units (typically
  3-4 characters each) using a fixed vocabulary, analogous to how a network
  protocol fragments a message into fixed-size packets. The model generates
  output one token at a time, selecting each token from a probability
  distribution over the vocabulary."

### 3. Entropy (in the generation/ML sense)

- **First appears:** Introduction line 49 ("entropy traces"), Design section
  ("per-token entropy of the output distribution during generation")
- **Issue:** Systems researchers know Shannon entropy from information theory,
  but may not immediately connect it to language model generation. The paper's
  core empirical finding depends on per-token entropy as a discriminative signal.
  The reader needs to understand that at each generation step, the model produces
  a probability distribution over possible next tokens, and the entropy of *that*
  distribution measures the model's uncertainty at that step.
- **Suggested definition:** "At each generation step, the model produces a
  probability distribution over its vocabulary of possible next tokens. The
  *entropy* of this distribution (Shannon entropy, $H = -\sum p_i \log p_i$)
  measures how uncertain the model is about its next output. Low entropy means
  the model strongly favors one continuation; high entropy means many
  continuations are roughly equally likely. This is a per-step measurement that
  the model cannot separately control -- it is a byproduct of the computation."

### 4. AUC (Area Under the ROC Curve)

- **First appears:** Background section line 124 ("self-report AUC is below 0.5"),
  used extensively in Evaluation
- **Issue:** AUC is a standard ML evaluation metric but is not common in systems
  papers. The paper uses AUC as its primary discrimination metric without
  defining it. A systems reader encountering "AUC = 0.87" and "AUC below 0.5
  (worse than random)" needs to know the scale and interpretation.
- **Suggested definition:** "AUC (Area Under the Receiver Operating
  Characteristic Curve) measures a classifier's ability to discriminate between
  two classes across all decision thresholds. AUC = 1.0 indicates perfect
  discrimination, AUC = 0.5 indicates no better than random, and AUC < 0.5
  indicates the classifier is systematically *inverted* -- it assigns higher
  scores to the wrong class. We use AUC to measure how well each signal
  separates knowable from unknowable queries."

### 5. Attention / Attention Patterns / Attention Summary

- **First appears:** Introduction line 49 ("attention geometry"), Design section
  ("attention summary," "self-attention ratio," "reasoning layers")
- **Issue:** "Attention" in the transformer sense is a specific computational
  mechanism, not general attentiveness. It is the mechanism by which the model
  decides which parts of its input and intermediate state are relevant to
  producing the next token. The paper proposes exporting "attention summaries"
  as part of the tensor interface, so the reader must understand what attention
  is being summarized.
- **Suggested definition:** "In transformer architectures, *attention* is the
  mechanism by which the model computes weighted relevance scores between all
  pairs of positions in its input. Each layer produces an attention pattern -- a
  matrix showing how much each output position 'attends to' each input position.
  This is analogous to a dependency graph: the attention pattern reveals which
  parts of the context the model considered relevant when producing each part of
  its output."

### 6. Transformer

- **First appears:** Introduction line 167 ("current transformers"), Design
  section line 291 ("current transformers")
- **Issue:** The paper assumes the reader knows that all major language models
  are built on the transformer architecture. A systems reader may know the term
  but not understand what architectural properties matter here (autoregressive
  generation, layer-by-layer processing, attention mechanism).
- **Suggested definition:** "The *transformer* is the dominant neural network
  architecture underlying all major language models. It processes input as a
  sequence of tokens, passing through a stack of layers, each of which computes
  attention patterns (weighted relevance between positions) and applies learned
  transformations. Generation is autoregressive: the model produces one token at
  a time, each conditioned on all previously generated tokens."
- **Note:** A single-paragraph "Architecture Primer" early in the paper (e.g.,
  in Background) could define transformer, token, attention, and entropy together
  and serve the entire rest of the paper.

### 7. Logits / Log-probabilities

- **First appears:** Introduction line 50 ("token-level log-probabilities"),
  Design section line 163-166 ("mean log-probability"), Evaluation extensively
- **Issue:** "Logits" are the raw unnormalized output scores of the model before
  the softmax normalization that produces probabilities. "Log-probabilities" are
  the logarithm of the resulting probabilities. The paper uses
  "log-probabilities" as one of three epistemic signals but never defines the
  term. A systems reader encountering "mean log-probability AUC = 0.87" needs
  to understand what is being measured.
- **Suggested definition:** "At each generation step, the model produces raw
  scores (*logits*) for every token in its vocabulary. These are normalized into
  a probability distribution (via the softmax function). The *log-probability*
  of the selected token is the logarithm of its probability under this
  distribution -- a measure of how expected that particular choice was. Low
  log-probability (large negative value) indicates the model's choice was
  surprising even to itself."

### 8. RLHF (Reinforcement Learning from Human Feedback)

- **First appears:** Keywords ("RLHF"), Background line 72 ("Reinforcement
  Learning from Human Feedback, RLHF")
- **Issue:** The paper spells out the acronym on first use in the Background
  section, which is good. However, the concept itself -- using human preference
  judgments as a reward signal to fine-tune model behavior -- is not explained.
  The paper's argument that RLHF creates perverse incentives (coherence over
  factuality) requires the reader to understand the mechanism.
- **Suggested definition:** "RLHF is the dominant method for aligning language
  models with human preferences after initial training. Human raters compare
  pairs of model outputs and indicate which they prefer; these preferences are
  used to train a reward model, which then provides the optimization signal for
  fine-tuning the language model. The key insight for our argument is that the
  reward signal is bounded by what the human rater can observe -- which is only
  the text."

### 9. Top-k Probability Mass

- **First appears:** Design section line 164 ("Top-5 probability mass"),
  Evaluation line 169 ("top-$k$ probability mass")
- **Issue:** This is one of the three proposed epistemic signals. A systems
  reader needs to know what it measures.
- **Suggested definition:** "The *top-k probability mass* is the sum of
  probabilities assigned to the k most likely next tokens. When this mass is
  high (close to 1.0), the model is concentrating its prediction on a few
  candidates; when it is low, probability is spread across many alternatives.
  Like entropy, this measures generation certainty but is more robust to
  distribution shape."

### 10. Instruct-tuning / Instruction-tuning

- **First appears:** Evaluation line 29 ("instruction-tuning effects")
- **Issue:** Mentioned only once but in a methodologically critical sentence
  explaining why base models were chosen. The reader needs to know what
  instruct-tuning is to understand the experimental design choice.
- **Suggested definition:** "Instruction-tuned (or 'instruct') models are base
  models that have been further trained to follow natural language instructions
  and produce conversational responses. This additional training can alter the
  model's internal signal profile, so we use base models to avoid confounding."

---

## Section B: Terms that SHOULD Be Briefly Glossed

These terms are used in passing or in specific contexts where a parenthetical
or footnote would prevent confusion. They are not central enough to warrant
full definitions but could trip up a careful systems reader.

### 1. Base model vs. Instruct model

- **First appears:** Background line 20 ("Base models"), Evaluation line 28-29
- **Issue:** The distinction between a base model (trained only on next-token
  prediction) and an instruct model (further fine-tuned for instruction
  following) matters for the experimental design. A one-sentence gloss would
  help.
- **Suggestion:** Parenthetical: "(a *base model* is the language model after
  initial training on text prediction, before any additional fine-tuning for
  instruction-following behavior)"

### 2. Softmax

- **First appears:** Not explicitly used in paper text, but implicit in
  discussion of probability distributions and logits
- **Issue:** If "logits" are defined (Section A), softmax should be mentioned as
  the normalization function. A parenthetical suffices.
- **Suggestion:** "(softmax: the standard function that converts raw scores into
  a probability distribution summing to 1)"

### 3. ROC Curve

- **First appears:** Evaluation line 170 ("ROC curves")
- **Issue:** ROC (Receiver Operating Characteristic) curves are not common in
  systems papers. AUC is defined in Section A; ROC should get a brief gloss.
- **Suggestion:** Parenthetical: "(a curve plotting true positive rate against
  false positive rate across all classification thresholds)"

### 4. Cohen's d

- **First appears:** Evaluation line 182 ("Cohen's $d = 1.57$")
- **Issue:** A standard effect-size measure in social sciences and ML evaluation,
  but not in systems. The number 1.57 means nothing without scale context.
- **Suggestion:** Parenthetical: "(a standardized effect size; $d > 0.8$ is
  conventionally 'large')"

### 5. Spearman rho ($\rho$)

- **First appears:** Evaluation line 106 ("mean pairwise Spearman $\rho = 0.762$")
- **Issue:** Rank correlation coefficient. Systems readers may know Pearson
  correlation but not Spearman specifically.
- **Suggestion:** Parenthetical: "(rank-order correlation; $\rho = 1$ indicates
  identical rankings)"

### 6. Perplexity

- **First appears:** Not explicitly used in the paper, but closely related to
  entropy. If added during revision, it would need a gloss.
- **Status:** Not currently in the paper -- note for future drafts only.

### 7. Forward pass

- **First appears:** Design section line 231 ("cannot present a different
  forward pass")
- **Issue:** "Forward pass" means the computation that runs input through the
  model to produce output. A systems reader might parse this but it is jargon.
- **Suggestion:** Replace with "computation" or add parenthetical: "(the
  computation that transforms input into output through the model's layers)"

### 8. Gradient signal / Expected gradient updates

- **First appears:** Formal proof, Theorem 2 proof line 205 ("expected gradient
  updates"), Design section line 87 ("gradient signal")
- **Issue:** Gradient-based optimization is the standard ML training method.
  The proof uses "gradient updates" in a formal context where the systems reader
  needs to understand it as "the training signal that tells the model which
  direction to adjust its parameters."
- **Suggestion:** On first use: "(the optimization signal that adjusts model
  parameters during training)"

### 9. Reward model / Reward signal

- **First appears:** Background line 60-66 (reward inequality), Background
  line 82 ("reward signal")
- **Issue:** The concept of a reward model (a separate model trained on human
  preferences that scores language model outputs) is central to the RLHF
  discussion but unfamiliar to systems readers. The paper uses "reward" as if
  the reader understands the RL training loop.
- **Suggestion:** Brief gloss when first used: "(a numerical score assigned to
  each output by a separate model trained on human preferences)"

### 10. Hidden-state activations

- **First appears:** Discussion line 70 ("full hidden-state inspection")
- **Issue:** "Hidden state" in the transformer sense means the intermediate
  vector representations at each layer. A systems reader might think of hidden
  Markov model state or process-internal state. A brief gloss helps.
- **Suggestion:** "(the intermediate numerical representations computed at each
  layer during the model's processing)"

### 11. Semantic entropy

- **First appears:** Design section line 190 ("semantic entropy")
- **Issue:** Mentioned as one of several internal signals alongside attention
  divergence and logit-based uncertainty. Semantic entropy is a specific
  technique (clustering outputs by meaning and computing entropy over meaning
  clusters rather than tokens). A footnote would help.
- **Suggestion:** Footnote: "Semantic entropy clusters model outputs by meaning
  and computes uncertainty over meaning groups rather than individual tokens,
  capturing uncertainty at the level of claims rather than word choices."

### 12. Fragmentation / Cognitive Slope (topological metrics)

- **First appears:** Design section lines 171-188
- **Issue:** These are defined in the paper (good), but the definitions use
  phrases like "activation geometry" and "attention-derived geometry" that may
  not land with a systems reader. The geometric interpretation is
  well-motivated but the underlying data (what exactly is being measured) could
  be more explicit.
- **Suggestion:** Add a sentence clarifying that these metrics are computed from
  the attention weight matrices produced at each layer -- making explicit that
  this is derived from the same attention mechanism defined earlier.

### 13. Parameters / 4B-8B parameters

- **First appears:** Discussion line 71 ("4B--8B parameters")
- **Issue:** Model size in "parameters" (billions of learned numerical weights)
  is standard ML shorthand. Systems readers understand scaling but may not
  know the convention.
- **Suggestion:** Parenthetical on first use: "(the number of learned numerical
  weights in the model)"

### 14. Autoregressive generation

- **First appears:** Not explicitly used as a term, but the concept is central
  (token-by-token generation). If "transformer" is defined per Section A, this
  should be covered there.
- **Status:** Covered by the suggested transformer definition in Section A.

### 15. Open-weight vs. closed-weight models

- **First appears:** Discussion line 61 ("closed-weight models that expose
  logits through API")
- **Issue:** This distinction (whether model weights are publicly available)
  matters for the paper's claims about generalizability. Systems readers would
  understand "open-source vs. proprietary" but "open-weight" is ML-specific
  terminology.
- **Suggestion:** Parenthetical: "(models whose internal parameters are publicly
  available vs. those accessible only through APIs)"

---

## Section C: Terms That Are Fine As-Is

These terms are either already well-defined in the paper, generally understood
by systems researchers, or sufficiently intuitive in context.

| Term | Why it works |
|------|-------------|
| Neural network | Common knowledge |
| Training data / training corpus | Common knowledge; "fossil record" metaphor helps |
| Probability distribution | Standard math; systems readers know this |
| Supervision / bounded supervision | Well-defined formally in Definition 5 |
| Verification budget | Defined formally; natural systems concept |
| Observation model | Defined formally in Definition 1 |
| Policy ($\pi$) | Defined formally in Definition 2; standard in formal methods |
| Benchmark / TruthfulQA | Context makes it clear these are evaluation test sets |
| Calibration | Used in standard statistical sense |
| Retrieval-augmented generation (RAG) | Defined implicitly ("conditions on retrieved documents"); the RAG acronym appears only once and the concept is explained |
| Grounding / grounded generation | The paper defines "grounded mode" vs. "fabrication mode" explicitly |
| Composition / compositional | Standard systems terminology |
| Provenance | Standard systems terminology |
| Model scale | Intuitive in context |
| Prompt / prompt engineering | Sufficiently well-known |
| Abstention ($\bot$) | Formally defined; standard notation |
| Vector clocks / logical timestamps | Systems audience knows these well |
| TLA+ / TLC model-checking | Systems audience knows these well |
| Coherence | Used in natural language sense; clear in context |

---

## Terminology Substitutions to Consider

Places where ML jargon could be replaced with systems-equivalent terminology
without loss of precision:

| Current term | Suggested alternative | Location | Rationale |
|---|---|---|---|
| "forward pass" | "inference computation" or just "computation" | Design, line 231 | Systems readers think of forward references, not NN forward propagation |
| "gradient signal" | "training signal" or "optimization signal" | Formal proof, Theorem 2 proof; Design section | The proof's logic does not depend on gradient-specific mechanics -- it depends on information content of the signal |
| "hidden-state activations" | "intermediate representations" or "internal layer outputs" | Discussion line 70 | "Hidden state" in systems means something different (e.g., in state machines) |
| "open-weight" | "open-weight (publicly available parameters)" | Discussion line 61 | One-time clarification |
| "instruct-tuning effects" | "instruction-following fine-tuning" | Eval line 29 | Slightly more self-explanatory |
| "activation geometry" | "internal representation geometry" | Design lines 171-188 | "Activation" is NN-specific; "representation" is more general |
| "reasoning layers" | "deeper layers" or "later processing layers" | Design section, Tensor Interface definition | "Reasoning layers" implies a specific function that is debatable; "deeper layers" is architecturally precise |

---

## Structural Recommendation

The paper would benefit from a short **Architecture Primer** paragraph (4-5
sentences) in Section 2 (Background & Motivation) that defines the core ML
concepts together, before they are used:

> *A transformer-based language model generates text one token at a time. A
> token is a subword unit (typically 3-4 characters); the model maintains a
> fixed vocabulary of tokens and, at each generation step, produces a probability
> distribution over this vocabulary. The token with the highest probability (or a
> sample from the distribution) is selected and appended to the output, becoming
> part of the context for the next step. This process is autoregressive: each
> step conditions on all previous output. At each step, the computation passes
> through a stack of layers, each producing attention patterns (weighted
> relevance scores between positions) and intermediate representations. The
> entropy of the output distribution, the attention patterns, and the
> log-probabilities of selected tokens are all byproducts of this computation --
> telemetry that the model cannot separately control.*

This single paragraph would define token, transformer, autoregressive
generation, attention, entropy, and log-probability in one place, using
language natural to a systems audience. Every subsequent use of these terms
would then be grounded.

---

## Summary Statistics

- **Section A (MUST define):** 10 terms
- **Section B (SHOULD gloss):** 15 terms
- **Section C (fine as-is):** 19 terms
- **Substitutions suggested:** 7

The paper is well-written for a systems audience in its formal sections
(the impossibility proofs, the TLA+ specifications, the verification budget
framework). The main gap is that the *empirical* sections assume ML fluency
that the formal sections do not require. The Architecture Primer paragraph
plus targeted definitions for Section A terms would close this gap without
adding significant length.
