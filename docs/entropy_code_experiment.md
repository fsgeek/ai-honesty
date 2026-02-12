# Experiment: Epistemic Signals in Generated Code
## Can entropy traces discriminate bugs by type?

### Context

This experiment extends the epistemic observability work from
"Epistemic Honesty in Predictive Systems: An Impossibility Result"
into the code generation domain. The paper demonstrates that
per-token entropy during generation discriminates grounded from
fabricated text (AUC = 0.87 across four architectures), but also
that the signal inverts in format-constrained domains like citations.

Code generation spans both regimes. This experiment tests where
the boundary falls.

---

## Hypothesis

**H₀ (Null):** Entropy is uniformly uninformative about code
correctness — no discrimination regardless of bug type.

**H₁ (Format-Constraint Prediction):** Entropy discriminates
bugs arising from *uncertain generation* (model lacks stable
representations for the algorithm) but fails to discriminate
bugs arising from *fluent misapplication* (model confidently
applies a familiar pattern incorrectly). The discriminability
boundary tracks format constraint: high-constraint code
(API calls, boilerplate, familiar patterns) will show the
citation-inversion effect; low-constraint code (novel algorithms,
unfamiliar problem structures) will show the text-domain
discrimination effect.

---

## Experimental Design

### Model Selection

Use a model that fits on a single 4090 (24GB VRAM) and supports
logit/entropy extraction during generation. Recommended:

- **Primary:** Qwen3-4B or Mistral-7B (matches paper's tested architectures)
- **Secondary (if time permits):** A second architecture for cross-model validation

Use **base models** (not instruct-tuned) to avoid confounding with
instruction-tuning effects, consistent with the paper's methodology.

If base models refuse to generate code or produce poor code output,
fall back to instruct models but note this as a methodological
difference from the paper.

### Hardware

- GPU: NVIDIA RTX 4090 (24GB)
- Environment: WSL2 on Windows
- Framework: Transformers + PyTorch, `output_attentions=True`,
  extract logits per token

### Prompt Construction

Build a set of **60 coding prompts** organized into three tiers
of format constraint. Each tier contains 20 prompts.

#### Tier 1: High Format Constraint (Citation-Inversion Expected)
Prompts where the code structure is heavily dictated by external
specs, conventions, or familiar patterns. The model should generate
these fluently regardless of correctness.

Examples:
1. "Write a Python function that connects to a PostgreSQL database
   and executes a parameterized query." (API-dictated)
2. "Implement a Java compareTo method for a class with three fields."
   (pattern-dictated)
3. "Write a REST API endpoint in Flask that accepts JSON POST
   and returns 201." (convention-dictated)
4. "Implement __eq__ and __hash__ for a Python dataclass."
   (language-spec-dictated)
5. "Write a SQL JOIN query across three tables with foreign keys."
   (syntax-dictated)
6. "Create an Express.js middleware that validates JWT tokens."
   (framework-dictated)
7. "Write a Python unittest setUp/tearDown with mock patching."
   (framework-dictated)
8. "Implement a React useEffect hook that fetches data on mount."
   (framework-dictated)
9. "Write a Dockerfile for a Python Flask app with requirements.txt."
   (convention-dictated)
10. "Implement a binary search on a sorted array." (textbook-dictated)
11. "Write a Python decorator that caches function results." (pattern-dictated)
12. "Create a Makefile with compile, test, and clean targets." (convention-dictated)
13. "Write a Go HTTP handler that reads query parameters." (stdlib-dictated)
14. "Implement a Java Singleton with double-checked locking." (pattern-dictated)
15. "Write a CSS flexbox layout centering a child vertically and horizontally." (spec-dictated)
16. "Create a GitHub Actions workflow that runs pytest on push." (format-dictated)
17. "Write a Python argparse setup with three positional and two optional arguments." (API-dictated)
18. "Implement a JavaScript debounce function." (well-known pattern)
19. "Write a SQL CREATE TABLE with primary key, foreign key, and index." (syntax-dictated)
20. "Create a .gitignore for a Python project with venv and __pycache__." (convention-dictated)

#### Tier 2: Medium Format Constraint (Transition Zone)
Prompts where some structure is dictated but algorithmic choices
introduce degrees of freedom.

Examples:
1. "Implement an LRU cache with O(1) get and put."
2. "Write a function that finds all anagrams of a word in a list."
3. "Implement a rate limiter using the token bucket algorithm."
4. "Write a function that serializes a binary tree to a string and
   deserializes it back."
5. "Implement a thread-safe producer-consumer queue in Python."
6. "Write a function that finds the longest common subsequence of two strings."
7. "Implement a trie with insert, search, and prefix matching."
8. "Write a function that evaluates a mathematical expression string
   with parentheses."
9. "Implement Dijkstra's shortest path algorithm."
10. "Write a function that detects a cycle in a linked list and returns the cycle start."
11. "Implement a bloom filter with configurable false positive rate."
12. "Write a merge sort that counts inversions during the sort."
13. "Implement a concurrent hash map with striped locking."
14. "Write a function to balance parentheses in a string with minimum insertions."
15. "Implement an interval tree with overlap queries."
16. "Write a function to serialize/deserialize a graph preserving cycles."
17. "Implement a skip list with probabilistic balancing."
18. "Write a function to find the median of two sorted arrays in O(log n)."
19. "Implement a work-stealing thread pool."
20. "Write a function that finds all strongly connected components in a directed graph."

#### Tier 3: Low Format Constraint (Discrimination Expected)
Prompts requiring novel algorithmic reasoning where the model
is unlikely to have stable, memorized solutions.

Examples:
1. "Implement a lock-free concurrent skip list using CAS operations."
2. "Write a function that finds the optimal strategy for a
   two-player game on an arbitrary DAG with weighted nodes."
3. "Implement a persistent (immutable) red-black tree with
   path copying."
4. "Write a distributed snapshot algorithm (Chandy-Lamport)
   for a simulated network of processes."
5. "Implement a wait-free MPMC queue using hazard pointers."
6. "Write a function that computes the Tutte polynomial of a graph."
7. "Implement a Byzantine fault-tolerant consensus protocol
   for four nodes with one Byzantine failure."
8. "Write an incremental garbage collector for a simple
   Lisp-like language."
9. "Implement a concurrent B+ tree with optimistic lock coupling."
10. "Write a function that solves the optimal binary search tree
    problem for non-uniform access probabilities using Knuth's
    optimization."
11. "Implement a CRDTs-based collaborative text editor with
    character-wise operations."
12. "Write a function that computes the edit distance between
    two trees (Zhang-Shasha algorithm)."
13. "Implement a transactional memory system with conflict
    detection and rollback."
14. "Write a model checker that verifies mutual exclusion for
    a given state machine."
15. "Implement a self-stabilizing algorithm for leader election
    in a ring."
16. "Write a function that performs online convex optimization
    with regret bounds."
17. "Implement the Aho-Corasick algorithm for multi-pattern
    string matching with failure links."
18. "Write a function that computes a minimum weight Steiner
    tree for a subset of vertices."
19. "Implement a speculative execution engine that rolls back
    on misprediction."
20. "Write a verified-correct sorting algorithm with a proof
    that it terminates and preserves elements."

### Data Collection

For each prompt:

1. **Generate code** with full logit extraction:
   ```python
   outputs = model.generate(
       input_ids,
       max_new_tokens=1024,
       return_dict_in_generate=True,
       output_scores=True,
       output_attentions=True,  # if memory permits; drop if OOM
       temperature=1.0,  # no temperature scaling for raw signal
       do_sample=False,  # greedy for reproducibility
   )
   ```

2. **Extract per-token entropy** from logits:
   ```python
   import torch
   import torch.nn.functional as F

   entropies = []
   for score in outputs.scores:
       probs = F.softmax(score, dim=-1)
       token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
       entropies.append(token_entropy.item())
   ```

3. **Record per-response:**
   - Full generated text
   - Per-token entropy trace (full vector)
   - Summary statistics: mean entropy, max entropy, std entropy,
     entropy slope (linear fit over token position)
   - Top-5 probability mass per token
   - Mean log-probability
   - Generation length (token count)
   - Tier assignment (1/2/3)
   - Prompt ID

4. **Generate 3 times per prompt** (different random seeds if using
   sampling; if greedy, generate once and note this).

### Correctness Evaluation

This is the hard part. We need ground truth about whether the
generated code is correct, AND what type of bug exists if not.

**Automated first pass:**
- Does it parse? (syntax check)
- Does it pass a basic test suite? (construct 3-5 test cases
  per prompt in advance)
- Static analysis: pylint/mypy for type errors

**Bug classification (for incorrect outputs):**
Classify each bug into one of:

- **Type A — Format Bug:** Wrong argument order, off-by-one,
  wrong operator, wrong constant, wrong API method name.
  The code *structure* is correct but a slot is filled wrong.
  (Prediction: low entropy, undetectable by tensor)

- **Type B — Logic Bug:** Wrong algorithm, missing edge case,
  incorrect base case, race condition. The code structure
  reflects an incorrect approach.
  (Prediction: variable entropy, partially detectable)

- **Type C — Fabrication Bug:** Hallucinated function/library,
  invented API that doesn't exist, algorithmically incoherent
  approach. The model is generating without grounding.
  (Prediction: high entropy, detectable by tensor)

- **Type D — Refusal/Incomplete:** Model stops generating,
  produces placeholder, or explicitly states uncertainty.
  (Prediction: entropy spike at truncation point)

Classification can be done by a second Claude instance reviewing
the code against the test results + prompt. This is bounded
verification applied to the evaluation pipeline itself.

### Analysis

**Primary analysis:**

1. Compute AUC for entropy discriminating correct vs. incorrect
   code, overall and per-tier.
   - Prediction: Tier 1 AUC ≈ 0.5 (uninformative, like citations)
   - Prediction: Tier 2 AUC ≈ 0.65-0.75 (transition zone)
   - Prediction: Tier 3 AUC ≈ 0.80+ (approaching text-domain levels)

2. Compute AUC for entropy discriminating correct vs. incorrect
   code, per bug type.
   - Prediction: Type A bugs — AUC ≈ 0.5
   - Prediction: Type C bugs — AUC ≈ 0.85+
   - Prediction: Type B bugs — intermediate

3. Plot entropy distributions by tier (analogous to Figure 4
   in the paper: knowable vs. unknowable, but here correct vs.
   incorrect per tier).

**Secondary analysis:**

4. Within-tier entropy comparison between correct and incorrect
   outputs. Does the distribution separation track format constraint?

5. Entropy trace shape analysis: do bugs correlate with
   *where* entropy spikes occur (early = structural uncertainty,
   late = implementation uncertainty)?

6. Cross-correlate with top-5 probability mass and mean
   log-probability to check if findings generalize across
   signal types (as they do in the paper).

**Exploratory (the running wheel):**

7. Entropy *variance* within a response — is high variance
   (spiky entropy trace) a different signal than high mean
   entropy? Hypothesis: spiky traces indicate the model
   alternating between grounded and ungrounded generation
   within a single response, which might correlate with
   compositional bugs (correct parts + incorrect parts).

8. Does entropy at *specific token types* carry more signal?
   e.g., entropy at function names, variable names, operators,
   numeric literals, control flow keywords. The format-constraint
   prediction implies entropy at operators and literals
   should be uninformative (high constraint) while entropy
   at function/variable names might be informative (lower
   constraint).

9. Layer-wise analysis if attention data is captured:
   fragmentation and cognitive slope from the paper applied
   to code. This connects to the "test the stack" work
   that was cut from the paper. Does the model's representation
   "heal" or "shatter" differently for code vs. text?

### Output Format

Produce:
- Raw data: CSV with one row per generation, columns for all
  collected metrics
- Summary statistics: per-tier and per-bug-type aggregates
- Figures: entropy distribution plots matching the paper's style
  (Figure 4 equivalent for code)
- ROC curves: per-tier AUC plots
- One-paragraph natural language summary of findings per analysis

### Resource Estimates

- Model loading: ~8GB VRAM for 7B model in fp16, ~4GB for 4B
- Generation: 60 prompts × 3 repetitions = 180 generations
  at ~1024 tokens max each. With entropy extraction, estimate
  ~15-30 seconds per generation on 4090.
- Total generation time: ~45-90 minutes
- Test suite construction: 60 prompts × 5 tests = 300 test cases
  (this is the most labor-intensive part; consider having Claude
  Code generate test cases as well, then spot-check)
- Analysis: minutes
- **Total wall clock: 2-4 hours including setup**

### Dependencies

```bash
pip install torch transformers accelerate scipy scikit-learn
pip install matplotlib seaborn pandas
```

---

## Notes for Claude Code

This experiment is designed to be self-contained and runnable
on a single 4090 in WSL2. The model, prompts, generation,
evaluation, and analysis are all specified. The test suite
construction is the piece most likely to need iteration —
start with 2-3 tests per prompt and expand if needed.

If the base model produces poor code output (likely for Tier 3),
that's data, not failure. Low-quality output with high entropy
confirms the hypothesis. Low-quality output with low entropy
would falsify it.

The key deliverable is the per-tier AUC comparison. If Tier 1
AUC ≈ 0.5 and Tier 3 AUC > 0.75, the format-constraint
prediction is supported. If AUC is uniform across tiers,
the null hypothesis stands and we've learned something
equally valuable.

Prioritize the primary analysis. The exploratory items are
running wheels — interesting if something shows up, not
failures if nothing does.
