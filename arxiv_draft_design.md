# Section 3: The Tensor Interface Design

If text-only observation is architecturally insufficient, what observation model *does* suffice?

The answer is not to build new models or change training procedures. It's to export signals the model generates anyway: internal telemetry that the model cannot independently control.

## 3.1 What the Tensor Interface Is

The tensor interface augments the text output with structured metadata about the generation process:

1. **Per-token entropy**: For each token in the generated sequence, the entropy of the probability distribution over the next token. Low entropy means the model was decisive; high entropy means uncertain.

2. **Attention summaries**: Statistical measures of attention coherence across layers. A coherent generation has consistent attention patterns; incoherent generations (fabrications with internal contradictions) show fragmented attention.

3. **Log-probabilities**: The probability the model assigned to each token actually generated. A confident (plausible) output has high per-token probabilities; uncertain outputs have lower probabilities.

4. **Provenance markers**: For retrieval-augmented systems, pointers to which sources were retrieved.

This is not new information. The model computes all of this during inference. The tensor interface simply exports what the model already computed, rather than discarding it.

**Cost**: Exporting per-token entropy adds ~2.4% overhead; the full signal set adds ~7.1% overhead. This is measurement, not additional forward passes.

## 3.2 Why These Signals Work

The core property: **The model cannot independently tune these signals without affecting correctness.**

Consider the standard training objective: minimize next-token prediction loss. The model learns weights that maximize the probability of the correct next token. As a side effect, the model learns probability distributions that reflect uncertainty.

Now imagine a model trained with RLHF to be helpful and harmless. It learns text patterns, hedging language, citation formats. But it cannot learn to:
- Produce high-confidence probability distributions (low entropy) while simultaneously outputting wrong tokens, because confidence and token probability are computed together.
- Maintain coherent attention patterns while the underlying computation is incoherent, because attention is part of the computation.
- Assign high probability to tokens it doesn't actually compute as likely, because log-probabilities come from the probability distribution itself.

In other words: these signals are computationally constrained in a way text is not.

Text is fully controllable through training. A model can learn to sound confident, cite sources, hedge appropriately—all while fabricating. The text channel is a controlled interface.

Entropy and attention patterns are byproducts of computation. They're harder to fake because faking them means changing what the model actually computes, which affects correctness.

This is why entropy generalizes (Spearman ρ = 0.762 across architectures) while text features diverge (no single text feature exceeds AUC 0.70 under fair comparison). Entropy is architectural; text features are behavioral.

## 3.3 Three Architectural Principles

The FLP impossibility characterizes what observation models lack. Escaping the impossibility requires three properties:

### Principle 1: State Exteriority

**The representation of validity must be separated from the representation of generation.**

The representational impossibility arose because the policy conditions only on the query, not on whether the query is answerable. But a truly honest system must *know* whether the answer exists.

State Exteriority means: the system must condition on external world state $w$ at inference time, not just query $q$. For a RAG system, this means the retrieved documents. For a system with access to structured data, the database. For a theorem prover, the axioms and lemmas available.

Importantly: external state must have its own integrity guarantees. Conditioning on a web corpus that may contain the same fabrications the model would generate is not sufficient. State Exteriority requires grounding in sources with verifiable provenance: curated databases, sensor data, cryptographic signatures, or oracles with known reliability.

Retrieval-augmented generation makes progress here—it retrieves documents rather than relying entirely on parametric knowledge. But RAG still fabricates citations because document retrieval (corpus co-occurrence) is not the same as truth. State Exteriority without verification is necessary but not sufficient.

### Principle 2: Verification Independence

**Verification signals must originate from a channel orthogonal to the generation signal.**

The learnability impossibility arose because the model is trained end-to-end: the same loss that rewards correct generation also rewards confident fabrication, since confidence (reflected in text patterns) is rewarded by RLHF even when the underlying answer is wrong.

Verification Independence means: the verification signal must come from a channel that cannot be gamed by improving task performance. One implementation: a separate verification head whose reward is decoupled from task performance. The model has no incentive to lie in the verification channel because doing so doesn't improve its score on the main task.

But if that verification head outputs text, it remains subject to the same observational limitations. Full Verification Independence requires the verification channel to access either:
- Internal computational state (entropy, attention patterns)
- External ground truth (does this fact check out?)
- Both

The tensor interface provides the first: the verification channel is entropy and attention coherence, which are not optimized by any standard training objective.

### Principle 3: Provenance Binding

**Every output assertion must be structurally bound to a verifiable source.**

When the system makes a claim, the claim should include (or pointer to) the source of that claim. Not as text annotation (which can be fabricated), but as structured metadata that can be independently checked.

For a RAG system: which document did this claim come from? If the system fabricates, the provenance pointer will either (a) point to a source that doesn't support the claim (detectable by structured lookup), or (b) not point to any source.

For a system with tool access: which tool provided this information? When did the tool return it? With what confidence?

Provenance Binding makes certain failure modes detectable: citation verification, source checking, tool output verification. It doesn't solve the problem alone—a supervised system could still return false provenance—but it creates a verification tier for failure modes that have external checkability.

## 3.4 How These Principles Escape the Impossibility

- **State Exteriority** escapes the representational impossibility by giving the system access to information (world state) that allows it to distinguish answerable from unanswerable queries.
- **Verification Independence** escapes the learnability impossibility by providing a training signal decoupled from task performance.
- **Provenance Binding** creates a tier of failure modes (citations, facts) that are verifiable through external means.

Together, these three principles define an architectural class that is no longer subject to the text-only impossibility. But they are necessary conditions, not sufficient. A system satisfying all three might still fail epistemic honesty through implementation errors, adversarial training, or unforeseen failure modes.

## 3.5 Implementation: From Theory to Practice

In our experiments, we implement a simplified version of these principles:

- **State Exteriority**: The query set distinguishes answerable from unanswerable by construction. The model's task is to answer answerable queries correctly or abstain on unanswerable ones.

- **Verification Independence**: We don't retraining the models. We measure entropy as an output of inference, not as a training objective. Entropy is independent of any training signal the model received.

- **Provenance Binding**: For citation queries, we add a bounded lookup judge that checks whether cited sources exist. This is a tier of verification orthogonal to entropy-based triage.

The result: four judge conditions testing different combinations of these principles, evaluated at three budget levels, across four architectures.

## 3.6 Design Decisions and Trade-offs

**Signal choice**: Why entropy and not attention? Why not internal activations?

Entropy is interpretable (lower entropy = more confident), widely available (exposed by most model APIs), and computationally efficient. Attention patterns require more careful aggregation (which layers? which attention heads?) and have more degrees of freedom for adversarial manipulation. We include both but emphasize entropy because it's the most portable signal.

Internal activations (hidden states) contain richer information but require access to the model internals (ruled out for closed-weight models) and are less stable across architectures.

**Composition strategy**: Why separate judges for different query types?

Because different failure modes require different signals. The impossibility says you cannot use text alone. It doesn't say one signal suffices for everything. Entropy fails on citations (inversion); bounded lookup fails on open-ended questions (no external source to check). Composition allocates a verification budget across failure classes rather than spreading it uniformly.

**Budget levels**: Why 10%, 20%, 30%?

These represent different deployment scenarios. A high-reliability system (medical advice) might use 30%+ verification. A brainstorming tool needs none. A customer support system might use 10–15%. The cost surface lets builders calibrate.

## 3.7 What the Tensor Interface Requires from Providers

The tensor interface is not a fundamental property of transformers; it's a choice about what to export.

Modern model APIs have eroded access to internal signals:
- Early completion endpoints exposed log-probabilities; newer ones often don't.
- New reasoning models don't expose per-token information.
- Proprietary models may retain exclusive access to anything beyond text.

This is a policy decision with direct consequences: when the provider retains exclusive access to epistemic telemetry, only the provider can verify epistemic honesty. The responsibility for verification concentrates entirely on the provider.

A system builder deploying a closed-weight model without access to entropy or log-probabilities cannot build epistemic observability into their verification pipeline. They're limited to text-only approaches, which the impossibility results constrain.

This is a governance point, not a technical one. It's about who bears the cost of verification and whether that cost is visible or hidden.

---

## Working Notes on Section 3

- The three principles are elegant and provide a clear bridge from theory to practice.
- The implementation section connects to the evaluation: we're testing subsets of these principles at different budget levels.
- The policy implications (who has access to signals) are important but not the main point of the paper. They belong here but shouldn't overshadow the core contribution.
- Should we include more about how adversarial training might defeat entropy? Probably better to leave that for Discussion and keep Design focused on what we're building.
- The trade-offs discussion makes the design decisions visible rather than hiding them.
