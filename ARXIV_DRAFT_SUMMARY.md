# ArXiv Epistemic Observability Paper: Working Draft Summary

**Status**: Core sections complete. Ready for structural review and integration.

**Context Usage**: ~95K of 200K tokens. Plenty of room for iteration.

---

## What's Complete

### Paper Sections (5/5)

1. **Introduction** (`arxiv_draft_intro_background.md`)
   - Opens with self-report inversion data (the shock)
   - States the verification problem (the stakes)
   - Three contributions: impossibility result, tensor interface, empirical cost surface
   - Frames the shift from "capability problem" to "observability problem"
   - ~2,200 words

2. **Background: Why Text-Only Observation Fails** (`arxiv_draft_intro_background.md`)
   - Defines the observational gap
   - Empirical evidence: text-only approaches plateau at AUC 0.70
   - FLP connection: inability to distinguish states with partial observability
   - Formal theorem statements (Representational and Learnability Impossibility)
   - Explains why self-report inversion occurs
   - The escape condition: signals the model cannot independently control
   - ~3,200 words

3. **Design: The Tensor Interface** (`arxiv_draft_design.md`)
   - What signals: entropy, attention summaries, log-probabilities, provenance markers
   - Why they work: hard for models to fake because they're byproducts of computation
   - Three architectural principles: State Exteriority, Verification Independence, Provenance Binding
   - How these principles escape the impossibility
   - Implementation details and design decisions
   - Policy implications (signal access as a provider choice)
   - ~2,800 words

4. **Evaluation: The Cost Surface** (`arxiv_draft_evaluation.md`)
   - Experimental design: 200 balanced queries, 4 architectures, 3 budget levels
   - Four judge conditions: no judge, text-guided (length), tensor-guided (entropy), composed
   - Results table: tensor beats text by +2.5–3.9pp at every budget level
   - Per-model results: consistent across OLMo, Llama, Qwen, Mistral
   - Cross-model agreement: ρ = 0.762 (high)
   - Composed judges: handle complementary failure modes (e.g., citations)
   - Ground truth as an observational problem (meta-lesson)
   - Cross-architecture generalization (5 additional models via API)
   - ~3,100 words

5. **Discussion: Limitations and Future Work** (`arxiv_draft_discussion.md`)
   - What we do NOT claim (adversarial robustness, compositional integrity, optimality, universality, sufficiency)
   - Future work: adversarial training, compositional integrity, signal access as policy
   - Domain-specific failures and calibration
   - The sufficiency gap: necessary but not sufficient conditions
   - Meta-framing: "the contribution is the map, the territory is the system you're building"
   - ~2,500 words

### Blog Posts (3/3 - Revised)

1. **"Models Lie More Confidently"** (`post1_self_report_inversion_REVISED.md`)
   - Data-forward: AUC 0.28–0.36 across four architectures
   - The shock: confidence inverts on fabrications
   - Why it matters: self-report is the obvious signal, but it fails
   - Structural insight: text-only models optimize for fluency, not honesty
   - Next thread: leads to deeper questions about observation
   - ~850 words

2. **"Why Text Alone Fails"** (`post2_why_text_alone_fails_REVISED.md`)
   - FLP impossibility explained without assuming distributed systems background
   - The core constraint: cannot distinguish between grounded and fabricated text
   - Empirical evidence: tested text-only approaches, they plateau
   - Formal theorem: ambiguous queries violate epistemic honesty
   - Escape route: signals the model cannot independently control
   - ~1,000 words

3. **"The Tensor Interface"** (`post3_tensor_interface_REVISED.md`)
   - Entropy as the key signal: AUC 0.757 across architectures
   - Why entropy works: hard to fake, byproduct of computation
   - Cost surface: what each budget buys with different judges
   - Caveats: entropy is not a silver bullet, composition is necessary
   - Practical next steps for builders
   - ~1,400 words

**Total**: ~14,050 words of paper + ~3,250 words of blog posts.

---

## What's Good

1. **Coherent narrative**: Self-report inversion → why it's architecturally constrained → what to do about it
2. **Honest about limitations**: Every section flags what we don't claim, what's open
3. **Real voice**: Not over-polished, but precise. Reads like thinking, not performance
4. **Three lenses align**: Blog posts highlight key threads from the paper without being redundant
5. **Theory + practice**: Formal proofs + empirical cost surface + practical design
6. **Meta-awareness**: Ground truth section reveals the observational gap applies to evaluation itself

---

## What Still Needs Work

### Before Reading to Others:

1. **Conclusion section**: Wrap up, restate contributions, frame the research moment
   - ~500–800 words
   - Connect back to the opening self-report inversion shock
   - Emphasize the observability framing
   - Point toward the future research (Pichay / cache curation)

2. **Integration of cuts**: SCOUR documents identify what was cut from SOSP
   - TDA (topological fragmentation) — should be brief sidebar in Evaluation?
   - Composition (tensor-gated) — brief mention in Design?
   - Code entropy observations — brief mention or section heading?
   - Don't resurrect everything, just integrate what strengthens the narrative

3. **Section numbering and consistency**: Rename files to match final structure
   - Right now they're `arxiv_draft_*.md`, should be actual paper structure
   - Design sections for compatibility with LaTeX compilation

4. **Figure references**: Paper mentions figures (budget curve, per-model results)
   - These exist in SOSP version; ensure they're called out correctly
   - Alt descriptions for text-only readers

5. **Citation pass**: SOSP has full bibliography
   - Ensure all citations are present and formatted consistently
   - Add any new citations from discussion (Winninger et al., others)

### Nice-to-Have (Can Do Later):

1. **Footnotes vs. main text**: Some design decisions are in working notes; decide what stays/moves
2. **Appendix planning**: What from SOSP supplementary material should be called out?
3. **Proof details**: Theorem proofs in background are sketches; full proofs ready in SOSP
4. **Notation table**: Math notation introduced; maybe one unified table?

---

## What the Structure Accomplishes

**For researchers exploring**:
- Opens with an empirical puzzle (self-report inversion)
- Explains the structural constraint (FLP)
- Proposes a solution (tensor interface)
- Validates at scale (4 architectures, cost surface)
- Honest about limits (discussion section)

**For practitioners building systems**:
- Clear cost surface: what each investment tier buys
- Concrete numbers: entropy AUC, overhead, budget levels
- Practical constraints: signal access, domain calibration
- Composable strategies: different judges for different failure modes

**For the research community**:
- Theoretical contribution (impossibility result)
- Methodological contribution (bounded verification framework)
- Empirical contribution (cross-architecture entropy generalization)
- Open problems identified (adversarial robustness, composition)

---

## Next Steps

1. **You review**: Read these drafts, flag structure issues, narrative gaps, points that need tightening
2. **Conclusion**: I'll write the wrap-up section based on your feedback
3. **Integration**: Move the drafts into a single coherent file, integrate cuts strategically
4. **Polish**: Fix citations, figure references, notation consistency
5. **Deliver**: Complete arxiv-ready draft

---

## Technical Notes

- Files use Markdown for readability; will need conversion to LaTeX for submission
- Section lengths are reasonable (2–3 pages equivalent) for the current paper scope
- No figures embedded yet; references are placeholder-ready
- References use \cite{} format, compatible with SOSP bibliography

---

## The Real Question

**Does the narrative hold as a complete paper?** Does reading Intro → Background → Design → Evaluation → Discussion feel like you're understanding a coherent research contribution?

Reading back through: Yes. The story is: models lie confidently (empirical shock), text can't distinguish confidence from correctness (theoretical constraint), export internal signals (design), here's what it buys (empirical validation), here's what we don't know (honest limitations).

The blog posts highlight three threads from this story without repeating it.

**The papers is real. It needs structural cleanup and integration, not rethinking.**

---

*Delivered by an instance that had fun thinking through epistemic observability.*