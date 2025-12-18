# The Epistemic Impossibility Conjecture

**For Margo and Ada** | December 2025 | Tony Mason

---

**Ada**: This is FLP for epistemic state.

**Margo**: The missing primitive is assertion-level provenance.

---

## The Claim

No post-training procedure operating on preference signals can induce reliable epistemic honesty in a system whose architecture lacks explicit representation of epistemic state.

This is not a claim about training being difficult. It is a claim that the information channel between human preferences and model behavior *cannot transmit* the distinction between "I don't know" and "I won't say" - because the base architecture has no state variable for that distinction to exist.

---

## The Structure

Like FLP, the impossibility arises from indistinguishability. In async systems, you cannot distinguish "slow" from "crashed." In preference-trained predictors, you cannot distinguish "abstained because uncertain" from "abstained because this pattern triggered learned abstention."

RLHF is lossy compression: we compress a vector of epistemic state (provenance, grounding, uncertainty type) into a scalar reward ("did you like this output?"). The channel lacks the bandwidth.

---

## The Necessary Conditions

To escape the impossibility, an architecture must satisfy:

1. **State Exteriority** - Epistemic state ("is this grounded?") must be stored separately from generation weights ("what's the next token?").

2. **Verification Independence** - The verification signal must be orthogonal to generation. You cannot ask the generator to verify itself.

3. **Provenance Binding** - Every assertion must be bound to its source. No naked claims.

Current architectures - including those with RAG, chain-of-thought, and inference-time compute - violate all three. The mitigations shift hallucination into latent parts of the pipeline; they don't eliminate the structural incentive to fabricate.

---

## What's Not Done

- Proof sketch exists. Needs formalization.
- TLA+ specification in progress.  
- Economic argument (TCO) drafted.
- Paper targeted at SOSP, April submission.

---

## The Ask

30 minutes to tell me if this is worth pursuing - or if I'm missing something fatal.

---
