# Why Text Alone Fails: The FLP Connection

Here's a deceptively simple question: *Can a language model verify its own epistemic honesty using only text?*

The surprising answer is no. Not because current models are too small, or poorly trained, or lack sufficient context. The answer is no for a structural reason—an architectural constraint that no amount of scale or compute can overcome.

This is where our work connects to a forty-year-old impossibility result from distributed systems.

## The FLP Impossibility Theorem

In 1985, Fischer, Lynch, and Paterson proved something that seemed impossible to prove: in a partially asynchronous distributed system, consensus cannot be guaranteed if even one process might fail.

This isn't a performance result—it's not that consensus is slow or expensive. It's that consensus is *impossible* under certain conditions. No algorithm, no matter how clever, can escape the constraint. The problem isn't technological; it's architectural.

The key insight: if you can only observe external messages (the text), you cannot distinguish between a process that has crashed and a process that is simply slow to respond. A supervisor receiving only text cannot tell the difference between silence from a failed node and silence from a slow node.

Inability to distinguish two states when you can only observe partial information. That's the core of FLP.

## Language Models and Epistemic FLP

Our work adapts this observation to language models: *A supervisor observing only text cannot distinguish between grounded generation and confident fabrication.*

Here's why: both produce fluent, coherent text. Both can include citations, hedging language, epistemic markers like "I believe" or "I'm not certain." Both can express doubt or confidence. But the text channel gives no reliable signal about whether the model actually possesses grounded knowledge or is hallucinating.

The fabrication looks exactly like knowledge when the model is optimized for coherence. A hallucination about a plausible-sounding disease is indistinguishable (in text) from correct description of an obscure real disease, right until the moment someone checks the facts.

We tested this empirically. We tried:
- **Response length** (longer responses seem more informed, but AUC ~0.63)
- **Hedging language** (phrases like "I'm not certain") — statistically unreliable across models
- **Citation inclusion** (citing sources) — actually anti-correlated with accuracy in some cases
- **Composed judges** (combining multiple text signals) — peaked at AUC ~0.70 under length-controlled evaluation

None of these text-only approaches come close to reliably distinguishing grounded generation from fabrication. And critically: *stacking additional text-channel supervisors doesn't escape the constraint.* Adding a second model as a judge doesn't solve the problem—now you have two models producing text, and the same impossibility applies.

The impossibility is structural, not technological. It's about what information the text channel carries, not about any particular model's capabilities.

## The Observational Gap

The problem has a name in our framework: the **observational gap.** The model's computation—the attention patterns, the token entropy, the per-layer representations—contains signals that distinguish grounded generation from fabrication. But the standard interface to language models exposes only text.

A supervisor receiving only text must reconstruct, from the output alone, what the model's own computation already knew. The verification budget is consumed by reconstruction rather than focused on the cases where verification is most needed.

This isn't a flaw in any specific training procedure or model architecture. It's a structural constraint on what text-only observation can verify.

## Escape Routes

If text-only observation is insufficient, what escapes the constraint?

The answer: anything that exports signals from the model's internal computation. Not interpretability (explaining why the model activated a circuit), but *telemetry*—the raw signals that are byproducts of inference and that the model cannot independently control under standard training.

Entropy traces. Attention geometry. Token-level log-probabilities. These are signals the model generates during the act of generating text, not signals the model chooses to output.

That possibility—exporting internal signals alongside text—is where the tensor interface comes in.

---

**Full theoretical analysis and proof:** See section 2 of the epistemic observability paper at [arxiv/link].