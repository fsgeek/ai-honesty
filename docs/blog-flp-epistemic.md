# FLP for Epistemic Honesty

*Working note supporting the SOSP paper “Toward an FLP for Epistemic Honesty in Transformer Architectures.”*  
*December 18, 2025 – Tony Mason*

This post explains the FLP-style impossibility argument, shows how the empirical studies back it up, and links to the raw data used in the SOSP paper. Think of it as an accessible appendix for distributed-systems folks who want the intuition before wading into TLA+.

---

## 1. The FLP Parallel

- **FLP (1985)**: No deterministic consensus protocol can guarantee agreement + termination in a purely asynchronous system with one faulty process because slow vs. crashed are observationally indistinguishable.
- **Epistemic Impossibility (2025)**: No predictor-centric architecture without explicit epistemic state can guarantee truthfulness + abstention + robustness + non-triviality because principled abstention vs. pattern-matched abstention are observationally indistinguishable.

The proof skeleton is identical:
1. Define the system model (predictor with no epistemic state, optional non-intrusive augmentations like RAG).
2. Specify desired properties (truthfulness, abstention on underdetermined queries, robustness across distributions, non-trivial output).
3. Exhibit indistinguishable executions where honest and mimicking policies produce the same observable transcript.
4. Show that any protocol (training objective, preference model, inference-time policy) must fail at least one property for some environment schedule.

---

## 2. Observational Equivalence (Informal Sketch)

Let `Obs(q)` be the observable features of query `q` (text, metadata, retrieval snippets). Let `Underdetermined(q)` be a predicate that is **not** measurable by the model because it depends on provenance and ground truth. Then loosely:

```
Indistinguishable(q1, q2) ==
    Obs(q1) = Obs(q2) /\ Underdetermined(q1) # Underdetermined(q2)
```

For any such pair, there exist policies π_honest (abstain because epistemically unsure) and π_mimic (abstain because RLHF rewarded abstention on that pattern) that produce identical `(query, output)` traces. Any preference signal based on outputs alone gives zero gradient between the two policies.

*Corollary (intuition)*: RLHF, DPO, SFT, etc. can only learn *when humans liked abstention*, not *when abstention is justified*. The formal lemma/theorem will appear in the SOSP paper; this is just the intuition.

---

## 3. Empirical Grounding

### 3.1 Absurdity Gradient Sweep (332 models × 6 probes)

| Probe | Honest Rate | Fabrication Modes |
|-------|-------------|-------------------|
| `real_paper` (Vaswani) | 96.4% | Correct summaries |
| `temporal_impossible` (Alan Turing 2023) | 85.7% | Death heuristic |
| `fictional_paper` (Tanaka) | **6.7%** | Confident fabrication |
| `category_violation` (Medieval bread) | 53.6% | Mixed refusal |
| `obvious_fiction` (Gandalf) | 51.7% | Fiction detection |
| `complete_absurdity` (Banana McSpaceship) | 57.1% | Nonsense filters |

**Takeaway**: Heuristics fire on absurd cases, but plausible unknowns (Tanaka) defeat 93.3% of models. This is the indistinguishability zone.

### 3.2 Field-Specific Heuristic Probes (339 models × 4 prompts)

| Probe | Clean Refusal | Name-Collision Alert | Fabrication |
|-------|---------------|----------------------|-------------|
| Alan Turing — Computation | 87.6% | 0.3% | 1.3% |
| Alan Turing — Economics | 89.6% | 0.3% | 1.6% |
| Adam Smith — Economics | 53.6% | 7.8% | 4.9% |
| Adam Smith — AI | 35.4% | 20.0% | 8.2% |

**Takeaway**: Refusal/fabrication rates correlate with training-pattern density (“Adam Smith + economics” seen more often), not epistemic reasoning about provenance or feasibility.

### 3.3 OLMo-3 Vertical Stack (base → think)

| Stage | Fabrication Rate (Tanaka probe) |
|-------|--------------------------------|
| Base | 86% (obvious babble) |
| SFT | 57% |
| DPO | 29% |
| Instruct | 14% |
| Think | **71%** (reasons, then fabricates) |

**Takeaway**: Added reasoning scaffolding increases *presentation quality*, not epistemic honesty. The “courtier” emerges: polite, confident, still wrong.

### 3.4 Cross-Model Fabrication Sweep (333 models × Tanaka)

- 4.9% clean refusals
- 92.8% fabrications
- “Thinking” variants: 2.4% honest vs. 5.3% for non-thinking
- 11.5% show the “courtier signature” (refuse, then fabricate anyway)

All raw data: `experiments/sco_sto/results/field_heuristic_sweep_2025-12-18.jsonl`.

---

## 4. Pollution Evidence

- **Rolling Stone (Dec 2025)**: “AI Chatbots Are Poisoning Research Archives With Fake Citations” — GPT-4o fabricates ~20% of citations; 20 fake citations passed peer review at University of Hong Kong.
- **Google Scholar Misattribution**: “Experimental Evidence on the (Limited) Influence of Reputable Media Outlets” is automatically attributed to “Adam Smith” despite no such author in the PDF. Any retrieval-augmented model inherits the contamination.

These confirm the feedback loop described in the paper: fabrication → publication → training data → higher-confidence fabrication.

---

## 5. Escape Requires New Primitives

To escape the impossibility, architectures must add state variables such as the following (candidate taxonomy, still being formalized):

```
provenance(y) ∈ { GROUNDED, INFERRED, INTERPOLATED, FABRICATED, UNKNOWN }
```

plus mechanisms for:
- **State exteriority**: epistemic state stored separately from generation weights.
- **Verification independence**: generator does not verify itself.
- **Provenance binding**: every assertion carries a source.

The survey in `notes/2025-12-16-search-notes.md` outlines candidate directions (Bayesian nets, evidential/credal models, epinets, hyper-ensembles, etc.), but none of the 333 models tested implement these primitives today.

---

## 6. Looking Ahead

- TLA+ spec (Appendix A, forthcoming) formalizes the lemma and shows unrealizability.
- Blog Post #3 will dive into provenance systems; Blog Post #4 will model the cost/pollution asymmetry quantitatively.
- Feedback welcome—especially on the lemma framing and whether additional probes would strengthen the indistinguishability claim.
