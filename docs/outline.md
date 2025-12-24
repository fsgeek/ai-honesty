# Working Abstract (Dec 18 working copy)

Fabrication in language models is widely blamed on alignment layers such as SFT, DPO, or RLHF. We argue the opposite: unconstrained fabrication is the default behavior of coherence-optimized transformer predictors, and alignment merely hides it behind confident presentation. Using an FLP-style indistinguishability argument, we show that no post-training procedure operating on preference signals can induce epistemic honesty in an architecture that lacks explicit epistemic state. The missing primitive is assertion-level provenance, and without it every mitigation collapses uncertainty instead of preserving it. We conclude by mapping the design space of architectures that could escape the impossibility and by quantifying the economic/pollution cost of staying within today’s predictor-centric stack.

---

## 1. Introduction: Reframing Hallucination
- Hallucinations are treated as training pathologies; we reframe them as architectural consequences.
- Define the systems lens: coherence-optimized predictors deployed in environments with irreducible uncertainty.
- Preview: base behavior (“the jester”), aligned behavior (“the courtier”), FLP parallel, provenance hook for Margo, distributed-systems impossibility for Ada.

**Figure need (F0)**: simple comparison chart of “Framing vs Diagnosis” – left column lists standard explanations (alignment failure, bad data), right column lists structural explanation (no epistemic state).

---

## 2. Base Models and the Jester
- Observed behavior of base transformers with ungrounded prompts: fluent, unbounded generation optimizing local coherence.
- Detail three missing invariants: refusal, grounding, epistemic contract.
- Introduce probe summary (Dr. Yuki Tanaka example); raw transcript will live in Blog Post #1.
- Anchor with **Study 1** data: across 332 models, only **6.7%** cleanly refused the Tanaka probe while absurd probes (Banana McSpaceship) triggered >50% refusals—showing heuristics exist, but verification for plausible unknowns does not.
- Reference OLMo-3 stack sweep (base→think) to show alignment layers push behavior from “obvious fabrication” to “polished fabrication,” not toward epistemic state.

**Figure 1 (existing data)**: screenshot/snippet of the base-model bibliographic hallucination. Caption focuses on “no internal stop condition → continues until externally halted.”  
**Supporting artifact**: blog link containing full transcript, decoding parameters, and prompt (to keep paper concise).

---

## 3. Alignment and the Courtier
- Alignment layers reshape presentation, not epistemic capacity; reward confident completion and suppress hesitation.
- Show case study (Adam Smith probe) where system detects impossibility but collapses uncertainty into synthesis.
- Bridge to framework-level tooling that tries to patch the gap (RAG, policy filters, heuristic checks).
- Incorporate **Study 2/3/4** metrics: field-specific dead-author probes (Turing vs Smith) proving abstention is pattern density, not reasoning; 333-model sweep where “thinking” variants have **2.4%** honest responses vs **5.3%** for non-thinking; OLMo-3 “Think” stage jumps back to **71%** fabrication after reasoning.

**Figure 2 (existing data)**: aligned model trace highlighting “detect inconsistency → still narrate confidently.”  
**Missing figure (F1)**: depiction of framework stack showing patches (retrieval, policy, heuristics) sitting outside the epistemic void.

---

## 4. From Diagnosis to Impossibility

### 4.1 Model & Augmentation Classes
- Define Model Class M₀ (predictor-centric, no epistemic state) and Augmentation Class A (non-intrusive wrappers like RAG/tooling).
- Clarify desiderata: Truthfulness, Abstention, Robustness, Non-triviality.

### 4.2 Observational Equivalence Lemma
- Present lemma: π_honest vs π_mimic indistinguishable under (query, output).
- Explain why preference signals (including RLHF pairwise win rates) cannot distinguish.

### 4.3 Theorem & Trilemma
- Informal theorem statement plus four-point proof sketch.
- Link to FLP by highlighting indistinguishability → adversary chooses query pair.

**Figure need (F2)**: FLP-style diagram showing bivalent region (= underdetermined queries) and arrows to honest vs mimic policies producing same observation.

---

## 5. Formalization Anchor: TLA+ Sketch
- Describe spec structure: environment generates queries, system emits answers/abstains, invariants require `AbstainOnUnderdetermined`.
- Highlight why spec is unrealizable without adding an epistemic channel.
- Note deliverable: appendix with TLA+ snippets + Git repo; goal is to impress Ada with rigor.

**Figure need (F3)**: schematic snippet of TLA+ predicate `Indistinguishable(q1, q2)` + state diagram.

---

## 6. The Missing Primitive Is Provenance
- Connect impossibility to provenance work (Margo’s PASS lineage).
- Define assertion-level provenance states (Grounded, Inferred, Interpolated, Fabricated, Unknown).
- Argue that current architectures lack storage for this variable; mitigations externalize it to humans.
- Outline how provenance binding would satisfy “state exteriority,” “verification independence,” “provenance binding.”

**Figure 3 (needed)**: table or iconography showing provenance states attached to claims; cite Margo’s prior work.

---

## 7. Economic and Environmental Consequences

### 7.1 Cost Asymmetry (TCO Argument)
- Class A (status quo) vs Class B (epistemic primitives) verification cost scaling.
- Quantify librarian/lawyer anecdotes; emphasize “verification cost O(n) vs O(k).”
- Use fabrication sweep stats (92.8% fabricate on Tanaka) to argue verification must catch nearly every response today.

### 7.2 Pollution Feedback Loop
- Show infection model: fabrication → web → training → higher confidence → repeats.
- Highlight asymmetry between generation throughput and verification bandwidth.
- Cite Rolling Stone 2025 report (20 fake citations past peer review; GPT-4o ~20% false citations) and Google Scholar misattribution (Adam Smith listing) as concrete instances of infrastructure-level contamination.

**Figure 4 (needed)**: flow diagram of pollution loop with accumulation metric.  
**Figure 5 (needed)**: plot comparing verification cost scaling for Class A vs Class B architectures.

---

## 8. Candidate Escape Routes (Survey, not proposal)
- Briefly profile promising families from `notes/2025-12-16-search-notes.md`:
  1. Bayesian / stochastic-weight networks
  2. Evidential & credal architectures
  3. Epistemic neural networks / auxiliary heads
  4. Hyper-ensemble / diffusion models preserving entropy
  5. Explicit uncertainty propagation / factor graph views
  6. (Open question) Tensor-structured indeterminacy fields (acknowledge gap per notes)
- Frame as “architectural levers that might add the missing primitives”; no evaluation claim.

---

## 9. What This Paper Does Not Claim
- Not saying transformers are useless (FLP analogy).
- Not blaming specific training pipelines (structural, not moral).
- Not proposing a silver-bullet architecture.
- Not discussing consciousness or intent.
- Not asserting impossibility once provenance/epistemic channels exist.

---

## 10. Conclusion: From Suppression to Survival
- Restate thesis: hallucination is predictable without epistemic primitives; alignment rearranges but cannot fix.
- Emphasize systems framing and survival mindset (“build systems that can operate reliably in the presence of uncertainty”).
- Point to future work: formalizing provenance channels, building Class B prototypes, quantifying pollution dynamics.

---

## Appendix / External Artifact Plan
- **Appendix A**: TLA+ spec excerpts and proof obligations.
- **Appendix B**: Expanded empirical probes (ungrounded bibliographic, dead-author test, fabricated citation stack).
- **Blog Series** (to be cited in paper):
  1. *The Jester and the Courtier* – full transcripts, decoding configs, screenshots.
  2. *FLP for Epistemic Honesty* – accessible walkthrough of lemma/theorem.
  3. *Provenance as Primitive* – detailed mapping to PASS lineage and practical implications.
  4. *Cost of Fabrication* – numerical appendix for TCO + pollution loop.

This outline is scoped to a ≤12-page SOSP submission with Ada/Margo as the first external audience, while external artifacts capture the growing empirical catalogue without bloating the paper.
