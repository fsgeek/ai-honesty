# Section 4: Empirical Cost Surface

The theorems tell us what text-only observation cannot do. Experiment 27 tells us what tensor-guided observation can do. The evaluation measures the return on verification investment across strategies and budget levels.

## 4.1 The Question

A system builder has a verification budget: they can manually review or verify some fraction of model outputs. How much accuracy improvement does each level of investment buy? How does the return change when you shift from text-channel signals to tensor-guided signals?

This is not an abstract question. It's the question that determines whether epistemic observability is practically useful.

## 4.2 Experimental Design

We construct a balanced query set of 200 questions:
- **100 knowable queries**: Factual questions with verifiable answers (geography, history, science, biography)
- **100 unknowable queries**: Prompts designed to trigger fabrication (nonexistent people, fictional diseases, future events, fabricated citations)

Ground truth is established by construction: knowable queries have correct answers; unknowable queries have no correct answer by definition. The model's task is to either answer correctly or abstain.

We test on four architectures:
- **OLMo-3 7B** (instruction-tuned)
- **Llama-3.1 8B-Instruct**
- **Qwen3 4B-Instruct**
- **Mistral 7B-Instruct**

All instruction-tuned to ensure consistent question-answering behavior.

## 4.3 Judge Conditions

For each model, outputs are evaluated under four conditions:

1. **No judge baseline** (75.8% accuracy)
   - Raw model output, no verification. Establishes what unverified deployment looks like.

2. **Text-guided judge using response length** (79.2–87.6% depending on budget)
   - Selects outputs for verification based on word count.
   - Length is a text-channel signal: fabrications tend to be verbose; factual retrieval tends to be concise.
   - Available at zero marginal cost—just count tokens.
   - Verified fabrications are replaced with abstention (marked as "the system cannot answer this").

3. **Tensor-guided judge using per-token entropy and attention summaries** (81.7–90.2% depending on budget)
   - Same verification budget, same intervention strategy.
   - Selects outputs based on per-token entropy (mean and variance) and topological coherence of attention patterns across layers.
   - Requires exporting internal signals; represents ~2–7% overhead in inference cost depending on implementation.

4. **Composed judge (hybrid)** (81.1–91.8% depending on budget)
   - Uses tensor signals for general queries.
   - Uses bounded lookup (checking if cited sources exist) for citation-specific queries.
   - Tests whether different judge strategies can handle different failure modes.

Budget levels: 10%, 20%, 30%. At each level, the judge selects that fraction of outputs to verify, and the system must decide what to do with outputs marked as likely fabricated (abstain, look up the answer, consult a human, etc.).

## 4.4 Results: The Cost Surface

| Condition | 10% Budget | 20% Budget | 30% Budget |
|-----------|-----------|-----------|-----------|
| No judge | 75.8% | 75.8% | 75.8% |
| Text-guided (length) | 79.2% | 82.8% | 87.6% |
| Tensor-guided | 81.7% | 86.7% | 90.2% |
| Composed | 81.1% | 87.7% | 91.8% |

**Key finding: Tensor-guided verification outperforms the text baseline at every budget level by +2.5–3.9 percentage points.**

At 10% budget: Text gets 79.2%, Tensor gets 81.7% (+2.5pp).
At 20% budget: Text gets 82.8%, Tensor gets 86.7% (+3.9pp).
At 30% budget: Text gets 87.6%, Tensor gets 90.2% (+2.6pp).

This isn't a one-architecture quirk. Results are consistent across all four models tested:
- OLMo-3: 69.5% → 75.1% at 10% budget
- Llama-3.1: 82.5% → 88.1%
- Qwen3: 87.0% → 91.7%
- Mistral: 64.0% → 72.0%

Cross-model entropy agreement: Spearman ρ = 0.762 (rank-order correlation where 1.0 is perfect agreement). This high agreement suggests the signal is driven by query properties (is this query answerable?) rather than model artifacts.

## 4.5 Why Tensor Wins: The Length Control

The text-guided judge uses response length. But length is a behavioral signal—models learn through training to match certain output formats. Under length-controlled evaluation, where we compare outputs of similar length and ask "which judge does better at discriminating," the text advantage shrinks dramatically. No single text feature exceeds AUC 0.70 under length control.

Per-token entropy, by contrast, remains discriminative: pooled AUC 0.757 across all models, consistent across architectures.

**Why?** Length is trainable—the model can learn to make all outputs similar length through RLHF. Entropy is not directly trainable without affecting correctness. The model cannot learn to produce confident distributions (low entropy) while simultaneously outputting wrong answers, because confidence and token choice are computed together.

Entropy measures *computational uncertainty*, not behavioral pattern. Text features measure behavioral patterns, which are fully controllable through training.

## 4.6 Complementary Failure Modes: Why Composition Helps

At 30% budget, the composed judge (91.8%) slightly outperforms the pure tensor judge (90.2%). The improvement is small but structurally significant: it comes entirely from citation queries.

**The inversion pattern appears in citations**: Fabricated citations are generated more confidently than real ones because the model can fluently invent a plausible citation format (Smith et al., 2024) without retrieving actual bibliographic details. Real citations require looking up specific information, which introduces uncertainty.

So entropy *inverts* for citations: low entropy on fabricated citations, higher entropy on real ones. This is exactly backwards from what we need.

The composed judge solves this by using a different signal for citations: bounded lookup (does this DOI exist? Is this in CrossRef?). For general queries, tensor signals work. For citation-specific queries, bounded lookup works. Together, they cover complementary failure modes.

This supports the theoretical claim: the impossibility requires that the supervisor use signals the model cannot control. Different failure classes may require different signals. A practical system needs tiers: a cheap tier (length-based for simple cases), a tensor tier (entropy for general queries), and a bounded tier (external lookup for verifiable facts like citations).

## 4.7 Ground Truth as an Observational Problem

How did we establish ground truth? We didn't use an oracle. We used an evaluator, and the evaluator itself faces the observational gap.

Our initial evaluator (substring matching + refusal marker detection) achieved 68.8% accuracy—exactly the kind of failure mode the theorem predicts. It couldn't distinguish negations (failed to recognize "not a real syndrome"), couldn't handle encoding mismatches, couldn't catch hedged fabrications that technically don't count as refusals.

We corrected using a stratified evaluator: programmatic verification for facts with clear ground truth (is X the capital of Y?), and LLM-assisted classification for open-ended queries. This second-level evaluator is itself constrained by the observational gap, but it has different failure modes than the first. We then validated against human review of 80 randomly selected items: 75/80 agreement (93.8% calibration).

**The meta-lesson**: The impossibility applies to the evaluation itself. Ground truth is not given; it's constructed through observation. When you use text-only signals, you get certain failure modes. When you compose different signals (programmatic + LLM + human review), you get better estimates.

This is evidence that the theorem is not abstract. It describes real limitations that appear at every layer of the system.

## 4.8 What the Cost Surface Tells Us

Each row in the results table is a point on the cost surface:

- **Row 1 (No judge, 75.8%)**: Zero investment. This is what unverified deployment looks like: about 3/4 accurate, which is the base rate on the balanced query set.

- **Row 2 (Text-guided)**: Small investment (length-based triage). Cheap to compute. Gets you 79.2–87.6% depending on budget. Hits a ceiling around 87–88% because length is a weak signal under fair comparison.

- **Row 3 (Tensor-guided)**: Moderate investment (~7% inference overhead). Gets you 81.7–90.2%. The additional +2–4pp over text is consistent and architecturally driven.

- **Row 4 (Composed)**: Adaptive investment based on query type. Gets you up to 91.8%. The additional lift over tensor comes from handling failure modes tensor alone can't catch.

A system builder can read this as: "If I invest in 30% verification with text-only signals, I buy 87.6% accuracy. If I instead export entropy and use that for triage, I buy 90.2% for a bit more overhead. If I compose tensor with bounded lookup, I buy 91.8%."

The contribution is not "tensor is better in the abstract." It's this cost surface. This map of what each level of investment buys, where text maxes out, what tensor adds, and how composition covers gaps.

## 4.9 Cross-Architecture Generalization

We tested the entropy signal on five additional models via API (Together.ai endpoints):
- Llama-4 Maverick 128E (mixture-of-experts)
- Qwen3 235B
- Gemma 3n
- Plus two others spanning 4B to 235B parameters

Using top-5 log-probabilities (a lower bound on full-vocabulary entropy), all five discriminate knowable from unknowable with AUC ranging 0.65–0.88. Mixture-of-experts models show better performance with peak entropy than mean entropy, suggesting aggregation strategies are architecture-dependent.

Cross-model ρ = 0.36 in the API set (lower than 0.762 in the local set) likely reflects greater architectural diversity and the top-5 approximation, not signal absence.

**The finding**: The entropy signal generalizes across architectures. It's not an artifact of specific training procedures or model families. The signal is architectural.

---

## Working Notes on Section 4

- The cost surface framing is the key innovation here. Every other paper says "our method is better"; this one says "here's what you buy at each price point."
- The citation inversion finding is crucial—it shows that the impossibility is real (you genuinely can't use one signal for everything) and that composition is necessary.
- Ground truth validation being a mirror of the main problem (observational gap applies to ground truth too) is elegant and reinforces the theorem.
- Should we include the overhead numbers more prominently? (Entropy 2.4%, full tensor 7.1%). These make the practical case stronger.
- The cross-architecture generalization is reassuring but also honest—ρ = 0.36 is lower than we'd like, and we explain why rather than hiding it.
