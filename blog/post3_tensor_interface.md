# The Tensor Interface: How Internal Signals Beat Text for Verification

If text alone is insufficient, what actually works?

In our experiments across four model families, one signal consistently distinguished grounded generation from fabrication: **per-token entropy**—a measure of how uncertain the model was while generating each token.

This isn't a post-hoc interpretation or a proxy signal. It's a byproduct of the model's computation: when a model generates a token, it assigns probabilities to all possible next tokens. The entropy of that distribution is a direct measure of decision uncertainty.

Low entropy means the model was decisive—one token was much more likely than others. High entropy means the model was uncertain, spreading probability across many options.

Our key finding: *When a model fabricates, its entropy is **lower**, not higher.* The model is decisive, fluent, confident in its generation—but the content is made up. When a model generates grounded facts, especially weird facts that are less common in training data, the entropy is **higher**. The model is genuinely uncertain, internally hesitant, even when it outputs text confidently.

## The Cost Surface

We tested this across four model families (OLMo, Llama, Qwen, Mistral) and built a cost surface: what does each verification budget buy you in terms of detection accuracy?

Under a fair comparison where we controlled for response length:

- **Text-only features**: No single text feature exceeded AUC 0.70. Length-controlled, they're even weaker.
- **Per-token entropy**: AUC 0.757, consistent across all four architectures. This is the only signal that generalizes.
- **Tensor-guided triage** (using entropy to prioritize which outputs to manually verify): Outperformed text baselines at every verification budget level, by +2.5–3.9 percentage points on average.

The practical implication: if you have a verification budget—say, you can manually fact-check 20% of model outputs—using entropy to guide which 20% to check beats using any text-based heuristic.

## Why Entropy Wins

The signal works because of the asymmetry in what the model can control. Under standard training (next-token prediction loss), the model optimizes for correct token prediction. It cannot independently tune its entropy distribution without affecting its accuracy.

Text, by contrast, is highly controllable. Models are trained on instruction-following, RLHF, and other procedures that let them learn to express confidence, doubt, hedging, citations—all in text. A model can learn to sound uncertain while remaining fluent. It can learn to cite sources that don't exist, with perfect formatting.

The entropy signal is harder to game because it's not directly optimized in training. It's a byproduct of the computational decision—the probability distribution over the next token space.

This is the same principle that makes code plagiarism detection (MOSS) robust: code structure is harder to forge than code content. The model cannot independently tune its attention patterns without affecting what it computes.

## The Tensor Interface

This leads to our proposal: **epistemic observability** through a **tensor interface.**

Instead of the standard text-only interface to language models, export:
- The text output (unchanged)
- Per-token entropy traces
- Attention summaries (how coherent the attention patterns are across layers)
- Token-level log-probabilities

These are byproducts of inference. They add minimal computational overhead (~2.4% for entropy alone, ~7.1% for the full signal set). They don't require architectural changes to current transformers.

And they work: a supervisor using these signals alongside text outperforms text-only verification at every budget level.

## What's Actually Observable

The tensor interface answers a narrower question than full interpretability: *Is this generation grounded or fabricated?*

It doesn't explain *why* the model generated something, or *how* a particular circuit activated. It just exports the telemetry that the model cannot separately control, giving a supervisor access to what the model's own computation already knew.

This is the escape route from the FLP-style impossibility of text-only observation. The impossibility applies because text is fully controllable. Telemetry is not.

---

**Full empirical results, methodology, and cost surfaces:** See section 4 of the epistemic observability paper at [arxiv/link]. Supplementary material includes reproducibility instructions and per-category breakdown.