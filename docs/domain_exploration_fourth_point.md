# Fourth Domain Exploration for Format-Constraint Manifold

**Date:** 2026-02-11
**Source:** Desktop (flatworm persona) brainstorm + Claude Opus fold
**Decision:** Mathematical proofs (LaTeX) as fourth domain, minified code as perturbation test

## Current Manifold (three measured points)

| Domain | Scaffolding % | Semantic % | Entropy Ratio | Spike Target |
|--------|--------------|------------|---------------|--------------|
| Plain text | ~15% | ~85% | 5.7x | Everywhere |
| Citations | ~45% | ~50% | 2.5x | SEMANTIC tokens |
| Code | 52.5% | 47.5% | 3.1x | SEMANTIC spikes (61.4%) |

## Selected: Mathematical Proofs (LaTeX)

- **Predicted scaffolding ratio:** ~60-65%
- **Predicted entropy ratio:** ~3.5x
- **Bounded judge analog:** Proof assistants (Lean, Coq, Isabelle)
- **Why it fits SOSP:** Formal verification is systems language
- **Prediction:** Spikes on proof-step content (assertions, lemma claims),
  not on logical connectives or notation

## Selected: Minified Code (perturbation test)

- **Concept:** Same domain as code, scaffolding stripped by minification
- **Prediction:** Entropy signal concentrates (doesn't disappear) as
  scaffolding is removed
- **Why it matters:** Shows the manifold is a continuous curve, not four
  dots connected by optimism. Reviewers can't dismiss as curve-fitting.

## Full Domain Space (for future ML/AI paper)

### High probability (p > 0.2)
1. **Mathematical proofs (LaTeX)** — scaffolding ~60-70%. SELECTED.
2. **API documentation / structured technical writing** — scaffolding ~35-40%.
   Fills the gap between text and citations.

### Low probability (0.2 >= p > 0.01)
3. **Legal contract language** — scaffolding ~65-75%, but semantic tokens carry
   enormous consequence weight. Tests whether scaffolding ratio alone predicts
   judge strategy or whether semantic token *consequence* matters.
4. **Musical notation (ABC/LilyPond)** — almost entirely scaffolding with
   creative decisions compressed into pitch/rhythm choices.
5. **Multilingual text (code-switching)** — BPE fragmentation becomes unstable
   across interleaved languages. New failure mode for the theory.

### Oddballs (0.01 >= p > 0.001)
6. **Chess PGN** — nearly 100% format-constrained. Extreme end of manifold.
   Bounded judge = Stockfish (deterministic, free, millisecond evaluation).
7. **Cooking recipes** — mid-range scaffolding, universally understood.
8. **DNA/protein sequences (FASTA)** — zero scaffolding, every token is content.
   Anti-code. Tests the other extreme.
9. **Spreadsheet formulas (Excel)** — high scaffolding, but semantic content is
   referential (cell references as choices).

### Crazy ideas (0.001 >= p > 0.0001)
10. **Emoji sequences** — BPE tokenization inconsistent. Scaffolding/semantic
    distinction may not be definable. Tests framework boundary condition.
11. **Compiler error messages (LLM-generated)** — text *about* code.
    Meta-scaffolding. Where does it sit?
12. **Liturgical text / prayer** — high repetition, strong format constraints,
    variations carry theological weight. Ancient template problem.
13. **Git commit messages** — tiny, constrained, semantic content maps to
    external state (the diff). Tests self-contained token assumption.
14. **Social media posts with hashtags** — hashtags as metadata scaffolding
    interleaved with semantic text.

### Wildcards (0.0001 > p) — theory-breakers
15. **Generated ASCII art** — spatial, not linguistic. Tests whether framework
    applies to non-textual token sequences.
16. **Lossy speech transcriptions ("um", "uh")** — fillers are linguistic
    scaffolding but HIGH entropy (unpredictable). Inverts the scaffolding-entropy
    correlation the manifold depends on. Most interesting potential theory-breaker.
17. **Obfuscated/minified code** — SELECTED as perturbation test.
18. **Bilingual code (Mandarin comments, Python code)** — double jeopardy for BPE.
19. **Fictional languages (Elvish, Klingon)** — tokenizer has no training
    distribution. Every token becomes high-entropy.
20. **This conversation** — self-referential evaluation of epistemic honesty
    about epistemic honesty evaluation. The scaffolding is academic framing.

## Key Insight from the Exploration

Several domains form natural pairs/transformations:
- Code -> Minified code (scaffolding removal)
- Text -> Legal text (high-consequence semantic tokens)
- Citations -> API docs (fills the ~35-40% gap)
- Code -> Chess PGN (extreme scaffolding end)
- Scaffolding-low-entropy assumption -> Speech fillers (inversion test)

The strongest future paper tests the manifold as a *surface* (scaffolding ratio x
semantic consequence x domain familiarity) rather than a curve. The current paper
tests the curve; the surface is next.
