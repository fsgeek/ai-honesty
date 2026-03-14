# Section 5: Limitations and Open Questions

We are explicit about what this paper does and does not claim.

## 5.1 What We Do NOT Claim

- **The tensor interface is unfakeable under adversarial training**: A model trained end-to-end to produce confident tensors while fabricating might learn to do so. Mechanistic interpretability has shown that internal signals are rich enough to be both attacked and defended. Whether adversarial training can force coupling between confident tensors and fabricated content remains open.

- **Composition preserves epistemic properties in recursive pipelines**: When LLM outputs feed into subsequent LLM inputs, the epistemic trace must be either preserved, regenerated, or deliberately discarded. Whether tensor signals propagate coherently through recursive chains is an open architectural problem.

- **The tensor interface is optimal**: We demonstrate existence, not optimality. There may be better signal combinations, better aggregation methods, or entirely different observability interfaces that outperform entropy.

- **These findings generalize to all architectures**: Our local experiments span four model families; API validation spans five additional families up to 235B parameters. Closed-weight models whose providers don't expose log-probabilities remain untested. The pattern is consistent, but coverage is incomplete.

- **Sufficient conditions for epistemic honesty**: The three architectural principles (State Exteriority, Verification Independence, Provenance Binding) are necessary for escaping the text-only impossibility. We do not prove they are sufficient. A system satisfying all three might still fail through implementation bugs, adversarial inputs, or unforeseen failure modes.

## 5.2 Future Work: Adversarial Robustness

**Can training defeat the tensor interface?**

The intuition is that entropy is hard to fake because faking requires changing what the model computes. But under end-to-end adversarial training, the model's objective includes "produce confident tensors for fabricated content." Can it learn to do so?

Preliminary analysis suggests linguistic and tensor confidence are decoupled in natural outputs: a model can express high confidence in text while producing low-entropy tensors. But whether adversarial training can *force* this coupling remains open.

Mechanistic interpretability work (Winninger et al. on craft attacks) shows that internal signals are rich enough to be attacked. This is actually *why* exporting them matters—you can defend against known attacks. But each new attack requires new defenses. Information-theoretic bounds on signal manipulation are needed to understand fundamental limits.

**Action for practitioners**: Entropy is more robust than text-only approaches, but it's not an oracle. Use it as one tier in a multi-tiered verification strategy, not as the single source of truth. Assume adversaries will find failures you haven't anticipated.

## 5.3 Future Work: Compositional Integrity

When a system uses LLM outputs as inputs to downstream LLMs, the epistemic trace must be preserved or explicitly discarded.

Example: A retrieval system returns documents. A summarizer LLM generates a summary from those documents. A question-answerer LLM answers questions about the summary. At each stage, a decision is made about whether to propagate the epistemic signals from the previous stage.

If the summarizer distorts the source material, should that be detectable in the entropy of the downstream answerer? The theory doesn't tell us. The compositionality of epistemic observability is an open problem.

**Why it matters**: Most real systems are pipelines, not single models. Epistemic signals that work in isolation might decay or invert through composition. Building compositionally-honest systems requires understanding how signals propagate.

## 5.4 Signal Access as a Policy Decision

The tensor signals we measure are byproducts of inference. Providers can expose them or withhold them. In practice, access has eroded:

- Early completion APIs exposed log-probabilities
- Recent reasoning models often don't
- Proprietary models may retain exclusive access

When a provider controls access to epistemic telemetry, only the provider can verify epistemic honesty. The cost of verification concentrates on the provider.

For a system builder, this creates a hard constraint:
- If you have access to log-probabilities and entropy, you can build multi-tiered verification.
- If you don't, you're limited to text-only approaches, which the impossibility results bound.

**Policy implication**: Responsibility concentration. Who bears the cost of verification? Is that cost visible or hidden?

A provider could:
- Export full epistemic telemetry (highest transparency, enables external verification)
- Export aggregated signals (moderate transparency, verification is guided but external)
- Export nothing beyond text (zero transparency, all verification responsibility falls on the provider)

This is a design choice, not a technological constraint. Different choices have different implications for trust and accountability.

## 5.5 Domain-Specific Failure Modes

The entropy signal measures generation confidence, not factual truth. This is an important distinction.

**Confident hallucinations**: A model can generate confident (low-entropy) nonsense. Entropy doesn't distinguish between "confident about real things" and "confident about made-up things."

**Epistemic refusals**: Models trained to decline unanswerable queries produce low-entropy refusals ("I don't know") that are entropically indistinguishable from confident correct answers. As models become more epistemically honest through training, the entropy signal's ability to detect dishonesty *degrades* because honest refusals and honest facts occupy the same region of entropy space.

**Citations**: Fabricated citations are generated fluently (low entropy) because the model invents plausible formats without retrieving specifics. Real citations require looking up details, introducing uncertainty (higher entropy). Entropy inverts on citation queries.

**Domain calibration**: In domains where confident answers are rare or where refusals are common, entropy thresholds need adjustment. The cost surface in Section 4 is calibrated to a balanced query set; in real deployment, base rates vary by domain.

Practitioners should:
- Validate entropy thresholds on domain-specific held-out data
- Use composed judges where entropy fails (e.g., citation lookup for citation queries)
- Monitor for drift as models improve or change

## 5.6 The Sufficiency Gap

The three principles are necessary but not sufficient. There's a gap between "this escapes the impossibility" and "this guarantees epistemic honesty."

Implementation can fail in many ways:
- Bugs in entropy calculation or aggregation
- Adversarial inputs designed to exploit the system
- Unforeseen failure modes we haven't tested

The papers provides the necessary conditions. Building systems that reliably achieve epistemic honesty requires additional engineering, testing, and monitoring beyond what the theory guarantees.

## 5.7 What This Paper Is Actually About

The core contribution is the cost surface: the empirical map of what each verification strategy buys at each investment level.

The theoretical contribution is the impossibility result: a formal proof that text-only observation is insufficient.

The design contribution is the tensor interface: a practical implementation that escapes the impossibility.

The meta-contribution is perspective: **epistemic honesty is not a capability problem, it's an observability problem.** Scaling models, RLHF, instruction tuning—these improve capability, not observability. To solve epistemic honesty, you need signals that the model cannot independently control. Exporting those signals is an architectural choice, not a technical achievement.

The engineering decision—what level of epistemic assurance does your system need?—is not a research question. A brainstorming tool needs none. A medical summary system needs substantial assurance. A legal research system needs the highest tier available.

This paper provides the cost surface for making that decision: what each level of verification investment buys, where the text-channel ceiling lies, what the tensor interface adds, and how content format modulates the tradeoff.

**The contribution is the map. The territory is the system you are building.**

---

## Working Notes on Section 5

- The "what we do NOT claim" section is crucial. It's where we demonstrate intellectual honesty and prevent overstatement.
- The adversarial robustness section acknowledges that our solution is not unbeatable, only better than text-only.
- The policy implications (signal access) are important because they connect the technical work to governance.
- The domain-specific failures show that entropy is a signal, not an oracle. This is honest and practical.
- The sufficiency gap explicitly separates necessary from sufficient conditions. Important for researchers building on this work.
- The final framing—"the contribution is the map"—is powerful and connects back to Tony's voice: empirical, practical, honest about limits.
