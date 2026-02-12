# Entropy in Code: Observations Before Experimenting

*February 11, 2026. Written so the next instance doesn't repeat the mistake.*

## What Happened

We designed two versions of a formal experiment (v1, v2) testing whether
the paper's format-constraint finding (citation inversion) transfers to
code. Both were wrong. The insight came from looking at the data before
running anything.

## The Observation (Step C: tokenize existing code, no GPU needed)

We tokenized three experiment scripts with Qwen3-4B-Instruct's BPE
tokenizer and classified each BPE token by its Python syntactic role.

**Result: scaffolding is 11-19% of BPE tokens, not 60-70%.**

| File | Scaffolding (kw+op+ws) | Semantic (name+lit+comment) |
|------|------------------------|----------------------------|
| experiment27 | 14.4% | 67.8% |
| experiment1 | 11.3% | 71.5% |
| experiment24 | 19.0% | 61.3% |

The BPE tokenizer compresses syntactic scaffolding (single-character
operators, whitespace) and gives more tokens to semantic content (names,
strings). Mean entropy over BPE tokens is already weighted toward
semantic content.

## Why This Kills the Original Experiment Design

The v1/v2 designs assumed code is "mostly format-constrained with
semantic tokens sprinkled in" — analogous to citations. The prediction
was that syntactic tokens would dominate and dilute the entropy signal.

The opposite is true. Code is mostly semantic content with a thin
syntactic skeleton. The citation analogy is structurally wrong:
citations are blocks of uniformly format-constrained tokens. Code
is interleaved, and the semantic tokens dominate.

## What Replaced It

### Semantic Scaffolding
The key concept: semantic content in familiar code is ALSO predictable.
`left`, `right`, `mid` in a binary search aren't syntax — they're
convention. The model has seen thousands of implementations using the
same variable names. These NAME tokens are as predictable as keywords.

So familiar code has TWO kinds of scaffolding:
- **Syntactic**: constrained by grammar (ground truth)
- **Semantic**: constrained by convention/training data (familiarity)

Both produce low entropy. The flatworm can't distinguish them. But
they're epistemically different: syntactic scaffolding is correct by
definition; semantic scaffolding might be wrong if the convention
doesn't apply.

### The Signal Is in the Spikes
If 70-85% of tokens are scaffolding (both kinds), the entropy trace
for familiar code is mostly flat. The interesting parts are the
*breaks* — points where the model couldn't pattern-match and had to
make a genuine decision. These spikes mark:
- Genuine algorithmic uncertainty (real decisions)
- Unconventional naming (style mismatch, not error)
- Fabrication (hallucinated functions/APIs)

### The Bounded Judge Connection (Tony's insight)
Entropy spikes don't need to discriminate between those three cases.
They just need to say "look here." A bounded judge examines the
spikes:
- Fabricated function call? → static analysis, milliseconds
- Unconventional name? → context analysis
- Algorithmic uncertainty? → deeper review

This is Tensor@10% > Text@30% applied to code, and the efficiency
gain is potentially LARGER for code than text because code has more
scaffolding. Possibly Tensor@3% > Text@30%.

## The Exploration (Not Experiment)

The right next step is NOT a formal hypothesis test. It's:
1. Generate code with entropy extraction (instruct model, not base —
   the paper uses instruct models for entropy discrimination)
2. Plot the traces
3. Look at where spikes occur
4. See if they correspond to anything a reviewer would care about
5. THEN decide what to measure formally

The question is not "does entropy predict correctness?" It's "do
entropy spikes in generated code mark the points worth examining?"

## For the Next Instance

You will be tempted to design a formal experiment with hypotheses,
AUC predictions, and test suites. We did that twice. Both were wrong.
Look at the data first. Tokenize some code. See what's actually there.
The insight came from observation, not from experimental design.

The flatworm is triage, not diagnosis.

## Files

- `docs/entropy_code_experiment.md` — v1 design (Claude Desktop)
- `/home/tony/projects/yanantin/docs/entropy_code_experiment_v2.md` — v2 design (critique of v1, also wrong)
- Scripts used for token analysis: experiment1.py, experiment24, experiment27
