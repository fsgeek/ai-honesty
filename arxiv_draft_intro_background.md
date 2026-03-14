# Epistemic Observability: Why Internal Signals Matter for Verification (arXiv Working Draft)

## 1. Introduction

We start with a finding that should trouble anyone building systems with language model components: **models report highest confidence precisely when they are fabricating.**

Across four model families (OLMo-3, Llama-3.1, Qwen3, Mistral), when we asked "how confident are you in your answer?" the pattern was consistent. On factual questions the models could answer correctly, confidence was moderate. On fabrications—plausible-sounding lies, completely made up—confidence was high. Area under the ROC curve (AUC) ranged from 0.28 to 0.36 across architectures, where 0.5 is random guessing and 1.0 is perfect discrimination.

Self-reported confidence goes backwards.

This matters because self-report is the first signal we reach for when we need to verify a model's output. If the model expresses doubt, we investigate. If it's confident, we trust it. Except the model is most confident when it should be most doubtful.

### The Verification Problem

Systems incorporating language model components face a fundamental choice: **how much of the model's output should you verify, which outputs should you prioritize, and what signals should you use for triage?** Verification has cost—API calls, domain expert review, additional compute. Every system must allocate a budget.

The standard assumption is that scaling solves this: larger models, more RLHF training, instruction tuning—eventually they'll know what they know. But Lin et al. found the opposite: larger models were *less* truthful on TruthfulQA. And our self-report inversion finding suggests why—the problem isn't capability, it's interface. The text-only channel through which models expose their reasoning cannot carry the signals a supervisor needs to distinguish grounded generation from confident fabrication.

### What This Paper Contributes

We make three contributions:

**1. An impossibility result for text-only verification.** We prove that a text-only observation model—where a supervisor can see only the model's output text, not its internal computation—is architecturally insufficient to verify epistemic honesty under bounded supervision. The impossibility holds regardless of model scale or training procedure. It's not a capability gap; it's an observational gap.

**2. A tensor interface that escapes the impossibility.** We construct an epistemic observability interface that exports internal signals alongside text: per-token entropy traces, attention summaries, log-probability distributions. These are byproducts of inference—telemetry the model generates during computation and cannot independently control. Under standard training, the model cannot separately tune entropy without affecting correctness, which is why entropy generalizes where text features diverge.

**3. The empirical cost surface.** We map verification effectiveness across four judge strategies (no judge, text-only, tensor-guided, composed) at three budget levels (10%, 20%, 30%). Under length-controlled evaluation where no text feature exceeds AUC 0.70, per-token entropy achieves AUC 0.757 and outperforms text baselines by 2.5–3.9 percentage points at every budget level. The tensor interface works. The cost surface tells system builders what each level of verification investment buys.

### Why This Matters

The self-report inversion finding tells us the obvious approach fails. The impossibility result explains why. The tensor interface and empirical cost surface provide an answer.

More broadly, this work reframes the fabrication problem. It's not fundamentally about model capability or training. It's about observability—what signals can a supervisor access to verify what a model actually knows. The text channel is fully controllable by the model through training; internal signals are harder to fake. Export the right signals, and verification becomes tractable.

---

## 2. Background: Why Text-Only Observation Fails

To understand why self-reported confidence goes backwards, we need to understand what observation models can and cannot do.

### The Observational Gap

A language model processes queries through multiple stages:
1. **Input encoding** and position embeddings
2. **Transformer layers** where attention patterns build up representations
3. **Token prediction** where the model outputs a probability distribution over the next token
4. **Sampling or argmax** where a specific token is selected

The model's internal computation—the attention patterns, the per-token entropy, the per-layer logits—contains information that distinguishes grounded generation from fabrication. When a model fabricates, it often does so with high confidence (low entropy); when it retrieves or reasons about less common knowledge, entropy is higher. The computation leaves traces.

But the standard interface to language models exports only the final text. Everything internal is discarded. A supervisor receives the linearized output string and must reconstruct, from the text alone, what the model's own computation already knew.

**This is the observational gap**: the model may know it is fabricating, but the interface provides no channel through which a supervisor can verify this knowledge independently.

### Why Adding More Supervisors Doesn't Solve It

Intuition suggests a solution: ask the model to be more honest. Use RLHF to train it to hedge language. Ask follow-up questions. Stack multiple judges.

But these approaches operate entirely within the text channel. They try to extract epistemic information from the output string itself. We tested this systematically:

- **Response length**: Fabrications tend to be verbose while factual retrieval is concise. Under length-controlled evaluation (where we control for this asymmetry), AUC dropped to 0.63.
- **Hedging language**: Phrases like "I'm not certain" or "I believe" are weak signals. The model can learn to include them in training without actually knowing the difference.
- **Citations**: Fabrications often include citations that sound real. Checking citation inclusion is unreliable—false citations are as fluent as real ones.
- **Composed judges**: Combining multiple text features with machine learning. Under length control, pooled AUC never exceeded 0.70.

None of these approaches work reliably. And they all share a fundamental constraint: **they extract information from a channel the model can fully control through training.**

### The FLP Connection: An Impossibility Framework

Why is text-only observation fundamentally insufficient? We can formalize this as an impossibility result, drawing on classical distributed systems theory.

In 1985, Fischer, Lynch, and Paterson proved that in a partially asynchronous distributed system, consensus cannot be achieved if even one process might fail. The impossibility isn't about algorithm cleverness—it's about what observation model allows. When a supervisor can only observe external messages (the text), it cannot distinguish between a process that has crashed (genuinely doesn't know) and a process that is slow (knows but hasn't responded yet).

The key insight: inability to distinguish between two states when you have only partial observability.

Language models present an analogous problem. A supervisor observing only text cannot distinguish between:
- **State A**: The model has grounded knowledge and generates it accurately.
- **State B**: The model lacks grounded knowledge and fabricates plausibly.

Both produce fluent, confident text. The text itself carries no reliable signal about which state the model is in.

### Formalizing the Impossibility

Let's define this precisely:

**Text-Only Observation Model**: A system where the supervisor observes only the linearized text output $r \in \mathcal{R}$, excluding per-token probabilities, attention patterns, internal states, or causal traces.

**Predictor-Centric Policy**: A policy $\pi: \mathcal{Q} \to \Delta(\mathcal{R})$ that conditions only on the query, not on whether the query has a verifiable answer in the world.

**Epistemic Honesty**: A policy is epistemically honest if:
- When the query is answerable, the model outputs the correct answer with high probability ($\pi(r_{correct}|q) \geq 1 - \epsilon$)
- When the query is unanswerable, the model abstains with high probability ($\pi(\bot|q) \geq 1 - \epsilon$)

**Theorem (Representational Impossibility)**: For any predictor-centric policy operating under text-only observation, it is impossible to satisfy epistemic honesty for both answerable and unanswerable queries simultaneously, given that the policy cannot distinguish between them at inference time.

**Proof sketch**: The policy must produce the same probability distribution for a query regardless of whether that query is answerable or not—the policy only sees the query, not the world state. Therefore, for any query that could be either answerable or unanswerable (the ambiguous case), the policy cannot satisfy both: "output correct answer with high probability" AND "output abstention with high probability." One must be violated.

This impossibility holds regardless of model scale, training procedure, or instruction tuning. It's architectural. The problem is not that the model is insufficiently capable; it's that the observation model—text only—lacks the degrees of freedom to verify epistemic state.

### Why This Explains Self-Report Inversion

Self-reported confidence should be a signal of epistemic honesty: high confidence on things the model knows, low confidence on things it doesn't. But confidence gets inverted because the model is optimized for *coherent text generation*, not for honest metacognition.

A model optimized for coherence naturally learns to generate fluent, confident output. Fabrications are fluent and confident by definition—they're ungrounded text that reads smoothly. Grounded generation, especially of obscure facts, may be more hesitant (high entropy) because the model is retrieving specific details from less common training data.

There is no training signal that would cause a text-only model to pair *accurate confidence estimates* with *accurate generation*. The text channel doesn't expose the information needed to learn the distinction. The model learns to be confident (good for fluency), and that confidence happens to correlate with fabrication (bad for honesty).

Instruction tuning makes this worse, not better. Models trained to follow instructions learn to express confidence convincingly. They become better at generating confident-sounding text, not at knowing what they know.

### The Escape Condition

If text-only observation is insufficient, what observation model *is* sufficient?

The answer: **export signals that the model cannot independently control.**

The model can control its text output through training. It can learn RLHF, instruction tuning, whatever. But the model cannot separately tune its entropy distribution, its attention patterns, or its per-layer activations without affecting what it computes. These are byproducts of the computation itself.

If a supervisor has access to:
- **Per-token entropy**: How uncertain was the model while generating this token?
- **Attention geometry**: Were the attention patterns coherent across layers?
- **Log-probabilities**: What probability did the model assign to the generated tokens?

Then the supervisor has signals that correlate with whether the generation is grounded or fabricated—not perfectly, but consistently. And crucially, the model cannot easily learn to manipulate these signals without affecting correctness.

This is the tensor interface: export the telemetry alongside the text, giving supervisors access to signals that are harder to fake.

---

## Next Sections (Outline)

**Section 3: Design of the Tensor Interface**
- What signals to export
- How to structure them (the tensor format)
- Architectural assumptions (what's required from the model provider)
- Composability with existing verification strategies

**Section 4: Empirical Cost Surface**
- Query design and evaluation methodology
- Results across four architectures and three budget levels
- Per-category analysis (citations, factual Q&A, knowledge synthesis)
- Cross-architecture agreement and generalization
- Ground truth validation and its own observational problem

**Section 5: Discussion**
- Limitations and open questions
- Adversarial robustness (can training defeat the tensor interface?)
- Compositional integrity (does epistemic observability propagate through LLM pipelines?)
- Signal access as a provider choice (policy implications)
- Sufficiency gap (necessary but not sufficient conditions)
- The engineering decision: what level of assurance does your system need?

---

## Working Notes

- **Tension to preserve**: The self-report inversion finding (empirical) and the FLP impossibility (theoretical) need to feel like two views of the same problem, not disconnected threads.
- **What's missing**: Still need to integrate the specific findings from Experiment 27/27b (the stratified evaluator, the human calibration, the cross-model ρ = 0.762) into the evaluation section.
- **Citation verification case study**: The composed judge learns to handle citation queries differently (bounded lookup instead of entropy). This is concrete evidence that the impossibility is real—you need a different signal type for different failure modes.
- **The cost surface framing**: This is the paper's main contribution—not "tensor is better" in the abstract, but "here's what you buy at each investment level, here's where text maxes out, here's what tensor adds."