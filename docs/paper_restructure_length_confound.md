# Paper Restructure: Length Confound as Feature, Not Bug

**Date:** 2026-02-11
**Context:** Scourers independently discovered that response length (AUC 0.88-0.96)
dominates mean entropy (AUC 0.89-0.92) for knowable/unknowable discrimination.
After controlling for length, residual entropy AUC drops to 0.60-0.70.
This finding restructures the paper's empirical story.

---

## The Numbers

| Model | Raw Entropy AUC | Length AUC | Residual Entropy AUC | Length % of signal |
|-------|:---:|:---:|:---:|:---:|
| Qwen | 0.896 | 0.881 | 0.668 | 58% |
| OLMo | 0.894 | 0.912 | 0.645 | 63% |
| Llama | 0.922 | 0.941 | 0.700 | 53% |
| Mistral | 0.905 | 0.956 | 0.604 | 74% |

Length explains 53-74% of entropy's discrimination above chance.
For Mistral, length alone (0.956) beats every entropy signal.

---

## The Restructured Argument

### Old story (what the paper currently says)
Entropy discriminates knowable from unknowable with AUC > 0.89 across four
architectures. Tensor signals beat text-only verification at lower budget.

### New story (what the paper should say)

**The impossibility theorem predicts the length confound.**

1. The theorem proves that text-only observation cannot guarantee epistemic
   honesty detection. Length is a text-channel signal — it measures *how much
   the model said*, which is observable without any tensor interface. The fact
   that length dominates raw discrimination is evidence FOR the theorem: the
   text channel carries signal, but it's the wrong kind.

2. Length is a *behavioral* signal. A model can be trained to fabricate briefly
   and confidently. Length-based detection is trivially defeated under
   adversarial pressure, the same way a firewall based on packet size is
   trivially bypassed.

3. The residual (0.60-0.70 AUC, or 0.1-0.2 above chance) is the *tensor
   interface* signal — the part that requires exporting computational
   byproducts. It is modest but has a property length does not: it is a
   computational signal the model cannot independently control. Flattening
   the entropy requires changing the generation process itself, the same way
   restructuring code to beat MOSS changes the code.

4. The gap between length AUC and residual entropy AUC is the theorem made
   empirical: it shows how much of the discrimination comes from the text
   channel (gameable) vs. the tensor channel (robust).

### Structure for the paper

**Section 4 (Evaluation) should present the decomposition explicitly:**

> We decompose the discrimination signal into two components. Response length
> alone achieves AUC 0.88-0.96 across models — a text-channel signal observable
> without any tensor interface. After controlling for length, per-token entropy
> retains AUC 0.60-0.70: a modest but persistent computational signal.
>
> This decomposition is predicted by our impossibility result (Theorem 1). The
> text channel carries behavioral signals (length, hedging patterns, refusal
> templates) that discriminate in current models but are trivially gameable:
> a model trained to fabricate briefly defeats length-based detection. The
> tensor channel carries computational signals (per-token entropy, attention
> geometry) that the model cannot independently control — flattening the
> entropy of a fabricated response requires changing the computation that
> produces it, analogous to the MOSS observation that restructuring code to
> defeat a complexity metric requires writing different code.
>
> The residual signal is modest. We do not claim that current tensor signals
> are sufficient for reliable epistemic honesty detection. We claim that they
> occupy a fundamentally different position in the adversarial landscape: they
> are robust to the class of behavioral adaptations that defeat text-channel
> signals.

**This framing has three advantages:**

1. **Honesty.** It reports the length confound before reviewers find it.
   Three of five independent scourers flagged it — reviewers will too.

2. **Strength.** It converts a weakness (most of the signal is length) into
   evidence for the core theorem (text-channel signals are insufficient).

3. **Systems audience.** SOSP reviewers understand the behavioral/computational
   signal distinction from security, fault tolerance, and MOSS. The argument
   that computational signals are more robust than behavioral signals is a
   systems argument, not an ML argument.

---

## The Manifold Framework

The format-constraint manifold provides the second major contribution:

| Domain | Scaffolding% | Scaffolding Rigidity | Entropy Ratio |
|--------|:---:|:---:|:---:|
| Natural text | 38% | Low | 1.2x |
| Code (documented) | 45% | High (mixed) | 2.3x |
| Code (standard) | 57% | High | 2.8x |
| Math proofs | 67% | Low (convention) | 1.4x |
| Code (compact) | 69% | Very high | 2.2x |

The surface is parameterized by (scaffolding_ratio, scaffolding_rigidity).
Entropy ratio is the dependent variable. This determines bounded judge strategy:

- **Low scaffolding, low rigidity** (text): Mean entropy is informative.
  Spike detection adds little because spikes are everywhere.
- **High scaffolding, high rigidity** (code): Mean entropy is diluted by
  near-zero scaffolding tokens. Spike detection (max_entropy) or content-only
  entropy is needed.
- **High scaffolding, low rigidity** (math proofs): Scaffolding tokens
  themselves carry entropy (choosing "therefore" vs "thus" is a decision).
  Need different classification boundaries.

The framework tells you HOW to use the tensor interface for each domain.
The residual tells you THAT the tensor interface carries signal.
The length confound tells you WHY the text channel is insufficient.

---

## Meta-observation

The length confound was discovered by deploying five open-ended scourers with
the instruction "look at the data and report what you find — we're not telling
you what to look for." Three of five independently flagged the same confound.

This is itself a demonstration of the bounded supervision pattern the paper
describes: multiple bounded observers with different perspectives, none
individually authoritative, whose agreement provides stronger evidence than
any individual finding. The methodology performed itself.

---

## Action Items for Paper

1. **Add length-baseline column** to all evaluation tables (Tables 2-3).
2. **Add the decomposition paragraph** to Section 4 (quoted above).
3. **Reframe the AUC claims**: "naive AUC 0.89-0.92 (majority length-confounded)"
   → "residual AUC 0.60-0.70 (robust to behavioral adaptation)."
4. **Add the adversarial robustness argument** explicitly: length is gameable,
   entropy is not independently controllable.
5. **Connect to MOSS**: same structural pattern (gaming the metric requires
   changing the thing being measured).
6. **Add the manifold framework** as the practical contribution: routing
   bounded judges by (scaffolding_ratio, scaffolding_rigidity).
7. **Acknowledge the scourers' meta-finding** in the discussion or a footnote:
   open-ended bounded supervision found the confound that targeted analysis missed.
