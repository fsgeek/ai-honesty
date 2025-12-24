# Preliminary Briefing — FLP for Epistemic Honesty

*Audience*: **Margo Seltzer** (provenance systems) and **Ada Gavrilovska** (distributed systems).  
*Status*: December 18, 2025 working draft, pre-SOSP submission.

---

## 1. Why You

- **Margo**: The impossibility result shows that assertion-level provenance is the missing primitive in large-scale AI deployments. Your PASS lineage work becomes the systems requirement for trustworthy language models.
- **Ada**: The argument is an FLP-style indistinguishability proof over predictor-centric architectures. We need your read on whether the model/augmentation definitions and adversarial construction hold water with a distributed-systems audience.

---

## 2. Core Claim

> *No post-training procedure operating on preference signals can induce reliable epistemic honesty in an architecture whose internal state lacks explicit epistemic variables (provenance, grounding status, uncertainty type).*

- Mirrors FLP: indistinguishability between “slow vs. crashed” → “abstained because uncertain vs. abstained because policy learned that feature.”
- Model Class **M₀**: transformer-based predictors without an epistemic channel.
- Augmentation Class **A**: non-intrusive wrappers (RAG, tool use, policy filters) observing only inputs/outputs.
- **Desiderata**: Truthfulness, Abstention, Robustness, Non-triviality.
- **Lemma**: π_honest and π_mimic produce identical observable behavior on underdetermined queries ⇒ zero gradient for preference training.
- **Theorem**: For some task distributions, any M ∈ M₀ with augmentation A faces a trilemma—break truthfulness, abstain everywhere, or fabricate confidently.

---

## 3. Evidence Snapshot

1. **Base “Jester” probe**: OLMo base model invents a non-existent Tanaka paper, never self-terminates, no epistemic cues (Figure 1 + Blog Post #1).
2. **Aligned “Courtier” probe**: Chat-tuned model detects the Adam Smith impossibility but still writes a detailed, authoritative synopsis (Figure 2 + Blog Post #1).
3. **TLA+ sketch**: Formalizes environment/system interaction; predicate `Indistinguishable(q1, q2)` cannot be evaluated without adding an epistemic channel.
4. **Provenance table**: Required state values (Grounded/Inferred/Interpolated/Fabricated/Unknown) have no slot in current inference stacks.
5. **Cost/pollution asymmetry**: Verification cost remains O(n) for Class A architectures, and fabrication pollution is self-amplifying.

---

## 4. Figure / Artifact Plan

| ID | Status | Description | Ask |
|----|--------|-------------|-----|
| F0 | needed | Hallucination framing vs. structural diagnosis | sanity check narrative |
| F1 | needed | Stack diagram showing where framework patches sit relative to missing primitive | advice on clarity |
| F2 | needed | FLP-style indistinguishability diagram/trilemma | Ada: critique proof sketch |
| F3 | needed | Provenance state binding visual | Margo: confirm taxonomy |
| F4 | needed | Pollution feedback loop | data sanity |
| F5 | needed | Verification cost scaling plot (Class A vs. Class B) | suggest case studies |

Appendices will carry the TLA+ snippet plus longer transcripts; Blog Post #1 already hosts the probe data.

---

## 5. Questions for Review

1. **Lemma/Theorem soundness**: Are the definitions of observable behavior and augmentation sufficient to claim indistinguishability?
2. **Provenance primitive**: Does the proposed assertion-level state align with provenance best practices? What is missing to make it actionable?
3. **Systems framing**: Does the analogy to FLP and CAP resonate, or should we adopt a different systems result for intuition?
4. **Evaluation expectations**: Given SOSP norms, is a diagnostic paper grounded in formal argument + qualitative probes acceptable? What additional evidence would you expect?
5. **Future work pointers**: Which architectural families (Bayesian nets, credal models, epistemic heads, hyper-ensembles, etc.) sound most promising as “Class B” exemplars worth highlighting?

---

## 6. Timeline

| Date | Milestone |
|------|-----------|
| Dec 18 – Jan 10 | Lock SOSP outline + gather artifact feedback (you) |
| Jan 11 – Feb 7 | Finish proof draft + TLA+ appendix; iterate with reviewers |
| Feb 8 – Mar 7 | Complete paper text, integrate provenance/TCO sections |
| Mar 21 | SOSP abstract deadline (target submission) |
| Apr 4 (est.) | Full paper deadline |

arXiv + blog posts will publish in sync with submission.

---

## 7. How to Help

- Inline comments on this briefing or a short call to discuss the theorem/primitives.
- Specific concerns about framing the provenance tie-in (Margo) or the FLP analogy (Ada).
- Suggestions for historical references or figures that would speak to SOSP reviewers.

Thanks for being the first readers—your alignment here sets the tone for the broader systems community review.

---

## Addendum: What’s Firm vs. In Flight

**Firm commitments**
- Empirical story (Jester/Courtier probes, fabrication rates, pollution cases)
- Core diagnosis: preference-only training can’t distinguish principled vs. pattern-matched abstention without extra observables
- Architectural thesis: we need assertion-level provenance / epistemic primitives

**Actively being formalized**
- Exact system model for the impossibility (boundaries of Model class M₀ and augmentation class A)
- Formal definition of “assertion” and the provenance state lattice (Grounded/Inferred/Interpolated/Fabricated/Unknown, how they compose)
- Propagation rules for epistemic state during autoregressive generation
- Precise statement of the indistinguishability lemma/trilemma (quantifiers, observable behavior definition, adversarial construction)

**What I’m asking from you at this stage**
- Are the intuitions and direction sound enough to justify investing in the full formalism?
- Where should the formal effort concentrate to land with SOSP reviewers (e.g., system model clarity, provenance semantics, architectural “slot” analysis)?
- Are there obvious blind spots or alternative framings I should consider before locking definitions?

This outline is intentionally high-level; I’ll follow up with the assertion/propagation spec and proofs once we align on the direction.
