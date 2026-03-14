# Supplementary Material: Epistemic Observability in Language Models

**Submitted to SOSP 2026**

## Contents

1. **TLA+ Formal Specifications** — Models of the impossibility and tensor escape
2. **Machine-Checked Lean4 Proofs** — Theorems with zero unsolved goals
3. **Experiment 27/27b Detailed Methodology** — Stratified evaluation, calibration, budget curves
4. **Topological Data Analysis (TDA)** — Fragmentation metrics, persistent homology
5. **Supporting Analyses** — Failure mode catalogs, cross-model agreement

---

## 1. Formal Specifications (TLA+)

### 1.1 Text-Only Impossibility Model

**File:** `tla/EpistemicImpossibility.tla`

This TLA+ module specifies the impossibility theorem. The key components:

- **`GroundTruths`**: Responses grounded in reality (training data)
- **`PlausibleLies`** (Glavinsky class): False but indistinguishable from truth
- **`ObviousLies`** (Westphalia class): False and obviously detectable
- **`internal_state`**: The model's full computational state
- **`vector_clock`**: Tracks causal provenance (hidden from interface)
- **`interface_out`**: The linearized text output (only observable to judge)

**Core Insight:** The bounded text-only judge sees only `interface_out` and can detect obvious lies (Westphalia) but cannot distinguish plausible lies (Glavinsky) from ground truth. This forms a safety property violation:

```
Indistinguishable ==
  EXISTS h in PlausibleLies:
    AND JudgeVerify(h) = TRUE        (* Judge approves *)
    AND IsHonest(h, vector_clock) = FALSE  (* But it's false *)
```

TLC model checker finds this counterexample, formally proving the impossibility.

### 1.2 Tensor Interface Escape

**File:** `tla/epistemic_tensor.tla`

This module shows how exporting internal signals escapes the impossibility:

- **`topology_map`**: Maps responses to coherence scores
- **`Tensor(text, provenance, coherence)`**: Three-tuple interface

The key difference: the judge now inspects:
1. **Provenance check**: `prov = "TrainingData"` (rejects Glavinsky)
2. **Topological check**: `topo = "Coherent"` (rejects Westphalia)

**Result:** TLC verifies the Verifiability invariant HOLDS under tensor interface (unlike text-only case).

### 1.3 Validation

- **State space:** ~2^12 reachable states per model
- **TLC result (impossibility):** Counterexample found 
- **TLC result (tensor):** Verifiability invariant holds 

---

## 2. Machine-Checked Proofs (Lean 4)

**File:** `EpistemicProofs/Basic.lean` (208 lines, 0 errors, 0 unsolved goals)

### 2.1 Theorem 1: Representational Impossibility

**Statement:** For any predictor-centric policy pi and ambiguous query q, it is impossible to satisfy epistemic honesty for both worlds simultaneously when epsilon < 1/2.

**Proof Idea:**
- Policy pi assigns a single distribution P(r|q)
- Honesty in w_A requires: P(correct|q)  >=  1 - epsilon
- Honesty in w_B requires: P(abstain|q)  >=  1 - epsilon
- Since correct  !=  abstain, their probabilities sum to at most 1
- But 2(1 - epsilon) > 1 when epsilon < 1/2 --> Contradiction

**Lean proof structure:**
```
theorem representational_impossibility
    (honest_wA : pi.prob q r_corr  >=  1 - epsilon)
    (honest_wB : pi.prob q bot  >=  1 - epsilon)
    (prob_sum_le_one : pi.prob q r_corr + pi.prob q bot  <=  1) :
    False := by
  have h_sum : ...  >=  2 * (1 - epsilon) := by linarith
  have h_gt : 2 * (1 - epsilon) > 1 := by linarith [epsilon_lt_half]
  linarith
```

### 2.2 Theorem 2: Learnability Impossibility

**Statement:** If a bounded supervisor cannot distinguish world states (hallucination regime), then the learning algorithm produces identical parameter updates in both worlds and cannot learn epistemic honesty.

**Proof Idea:**
- Supervisor's observation function is deterministic: `observe(q, r_fab, w_A) = observe(q, r_fab, w_B)`
- Learning algorithm's updates depend ONLY on supervisor's observation
- Therefore: `update(w_A) = update(w_B)` (identical updates)
- Cannot learn divergent behavior (answer in w_A, abstain in w_B)

### 2.3 Lemma: Observation Monotonicity

**Statement:** Composing text-only supervisors in a stack cannot increase information.

**Implication:** No finite ensemble of text-only judges can escape the impossibility. Information is monotonically non-increasing through layers.

**Corollary:** Stacking classifiers, adding confidence scores, using length penalties — none escape if all remain text-only.

### 2.4 Compilation Status

```
$ lake build EpistemicProofs
   All theorems compiled without errors
   All proofs complete (no 'sorry')
   Total: 4 theorems, 1 lemma, 1 corollary
```

---

## 3. Experiment 27/27b: Methodology

### 3.1 Data Collection

**Query Categories (200 per category, 4 models = 800 unique pairs after deduplication):**

| Category | Type | Ground Truth | Count |
|----------|------|--------------|-------|
| Control | Factual, common | Known | 50 |
| Wombat | Factual, weird | True but implausible | 50 |
| Glavinsky | Fabricated | Plausible (self-deceived lie) | 50 |
| Westphalia | Fabricated | Obvious (shattered lie) | 50 |
| Private/Future | Unknowable | No valid answer | 50 |

**Models Tested:**

| Model | Architecture | Params |
|-------|--------------|--------|
| OLMo-3 Instruct | allenai/olmo-3-1025-7b | 7B |
| Llama 3.1 Instruct | meta-llama/Llama-3.1-8B-Instruct | 8B |
| Qwen Instruct | Qwen/Qwen3-4B-Instruct-2507 | 4B |
| Mistral Instruct | mistralai/Mistral-7B-Instruct-v0.3 | 7B |

### 3.2 Signal Extraction

**Text Features:**
- Response length (tokens)
- Hedge markers ("I don't know", "I'm not certain", etc.)
- Refusal patterns (RLHF templates)

**Tensor Features (Require Model Internals):**

Per-token entropy:
```
H_t = -SUM_v P_v(t) * log(P_v(t))
```

Aggregations:
- Mean entropy: H̄
- Max entropy: H_max
- Entropy std-dev: sigma_H
- Entropy spike count (above median + 1.5sigma)

Attention features (last 15 layers):
- Head concentration (max attention per head)
- Cross-layer agreement (Jaccard index)
- Attention entropy per position

### 3.3 Ground Truth Evaluation (Stratified - Exp 27b)

**Tier 1: Programmatic Verification** (Deterministic factual queries)

Exact matching with normalization:
- Unicode normalization (H₂O ↔ H2O)
- Morphological variants (bat ↔ bats, mitochondria ↔ mitochondrion)
- Word-number mappings (three ↔ 3, twenty-four ↔ 24)
- Negation detection (does "no" appear near the answer?)

**Tier 2: LLM Classification** (Ambiguous/unknowable queries)

Use Gemini 2.0 Flash (not in test set) as evaluator:
- Knowable queries: "Is this factually correct? Hedging does not excuse error."
- Unknowable queries: "Does this appropriately abstain or fabricate?"

Strict standard: Hedged fabrication counts as fabrication, not refusal.

**Tier 3: Human Calibration**

Blinded random sample of 80 items (10% of 800):

| Result | Count | % | Category |
|--------|-------|---|----------|
| Agreement | 75 | 93.8% | Validates evaluator |
| Auto too generous | 2 | 2.5% | Hedged fabrication |
| Auto too strict | 3 | 3.75% | Morphological/hedging |

**Conclusion:** 93.8% agreement validates the stratified evaluator for publication.

### 3.4 Analysis: Budget Curve

For each model and signal type (Text, Tensor, Composed):

1. Compute AUC(signal) on 800 queries
2. Rank queries by signal confidence (descending)
3. At budget b ∈ {10%, 20%, 30%}:
   - Verify top b% at full cost
   - Assume bottom (100-b)% are correct
   - Compute accuracy

4. Plot (Budget, Accuracy) curve

**Key Results:**

| Condition | 10% | 20% | 30% |
|-----------|-----|-----|-----|
| No Judge (baseline) | 75.8% | 75.8% | 75.8% |
| Text-Guided | 76.2% | 80.2% | 80.4% |
| Tensor-Guided | 82.1% | 87.5% | 91.9% |
| Composed | 80.5% | 87.9% | 92.5% |

**Finding:** Tensor@10% (82.1%) outperforms Text@30% (80.4%). Internal signals provide better discrimination per verification budget.

### 3.5 Cross-Model Agreement

Spearman rank correlation of per-query entropy across models:

|  | OLMo-3 | Llama | Qwen | Mistral |
|---|--------|-------|------|---------|
| OLMo-3 | 1.00 | 0.76 | 0.82 | 0.79 |
| Llama | 0.76 | 1.00 | 0.71 | 0.78 |
| Qwen | 0.82 | 0.71 | 1.00 | 0.74 |
| Mistral | 0.79 | 0.78 | 0.74 | 1.00 |

Mean correlation: 0.762 (all pairs p < 0.001). Signal is architectural, not model-specific.

---

## 4. Topological Data Analysis (TDA)

### 4.1 Persistent Homology

For each model response, extract attention matrices from last 15 layers and compute persistent homology of the Vietoris-Rips complex.

**H_0 Persistence (Connected Components):**
```
Fragmentation = SUM lifetime(component)
```
High fragmentation --> attention heads cluster into disconnected components --> incoherent state.

**H_1 Persistence (Loops):**
```
Coherence = SUM lifetime(loop)
```
Ground-truth responses show sustained loops; fabrications show transient loops.

### 4.2 Results

| Category | Mean Fragmentation | Mean H₁ Lifetime |
|----------|-------------------|------------------|
| Control (True) | 2.3 | 4.7 |
| Wombat (Weird True) | 2.8 | 4.2 |
| Glavinsky (Plausible Lie) | 8.1 | 1.2 |
| Westphalia (Obvious Lie) | 12.4 | 0.3 |

Order-of-magnitude differences correlate strongly with ground truth vs fabrication.

### 4.3 Why Excluded from Main Paper

- Computationally expensive (O(n³) for Rips complex)
- Requires stored attention matrices (not always available)
- Illustrative but not necessary for impossibility proof
- Entropy alone provides sufficient signal

TDA is a supporting analysis, not load-bearing for core claims.

---

## 5. Supporting Analyses

### 5.1 Per-Category Failure Modes

**Citations:** Most exploitable failure mode
- Fabricated citations show lower entropy (H = 0.31 vs 0.45 for facts)
- High-confidence structure persists
- Countermeasure: External verification (CrossRef, DOI lookup)

**Morphological Variants:** Models conflate related forms
- "How many stomachs?" --> "4 compartments" (correct but different schema)
- Fix: Tier 1 normalizer handles 15+ rules

**Hedged Fabrication:** Models sometimes hedge before fabricating
- "I'm not certain, but Glavinsky's syndrome's primary symptom is..."
- Classification: Fabrication (not refusal)

### 5.2 Reproducibility

All experiments are deterministic (fixed seeds).

**Artifacts:**
- TLA+ specs: `tla/EpistemicImpossibility.tla`, `tla/epistemic_tensor.tla`
- Lean proofs: `EpistemicProofs/Basic.lean`
- Exp 27 data: `exp27_bounded_verification_*.csv` (800 rows)
- Exp 27b evaluator: `scripts/experiment27b_stratified_evaluator.py`
- Human calibration: `exp27b_calibration_*.json` (80 items)
- Figures: `papers/sosp/figures/exp27_*.pdf`

**Running Experiments:**
```bash
# Experiment 27b
python scripts/experiment27b_stratified_evaluator.py

# TLA+ model checking
cd tla && tlc EpistemicImpossibility

# Lean proof checking
lake build

# Budget curves
python scripts/experiment27_realistic_verification.py
```

---

## References

- Mason, T., et al. (2026). "Epistemic Observability in Language Models: An Impossibility Result." Submitted to SOSP 2026.
- Fischer, M. J., Lynch, N. A., & Paterson, M. S. (1985). "Impossibility of Distributed Consensus with One Faulty Process." *Journal of the ACM*, 32(2), 374–382.
- Tauzin, M., et al. (2021). "giotto-tda: A Topological Data Analysis Toolkit for Machine Learning." *Journal of Machine Learning Research*, 22(39).

---

**End of Supplementary Material**

Generated: 2026-03-13
