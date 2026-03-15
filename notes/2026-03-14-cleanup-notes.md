● Brief for the ai-honesty instance:

  ---
  Task 1: Clean Up the SOSP Supplement

  File: /home/tony/projects/ai-honesty/papers/sosp/supplementary.tex (661 lines)

  Priority: This must be done first. The arXiv rewrite depends on the supplement being correct.

  Fix 1: Stale Lean Excerpt (CRITICAL)

  The layered_judges_cannot_escape theorem excerpt in the supplement (lines 222-235) does not match the actual Lean source at /home/tony/projects/ai-honesty/EpistemicProofs/Basic.lean (lines 174-186).

  The supplement has broken pseudocode including stack cannot distinguish them := by. The actual Lean file has a clean, valid proof using induction n with / simp / rw [ih].

  Action: Replace the excerpt in supplementary.tex with the actual code from Basic.lean. Also update the observation_monotonicity excerpt (supplement lines 208-216) to match the actual Lean (lines 158-165) — the signatures differ (the real version takes explicit obs_a obs_b arguments and a hypothesis h, not
   the version shown in the supplement).

  Verify: Compare every Lean listing in the supplement against Basic.lean line by line. The supplement was generated after the Lean file and introduced errors. Trust Basic.lean as ground truth.

  Fix 2: Numerical Discrepancy with Main Paper

  The supplement's budget curve table (lines 440-452) shows:
  - Text-Guided 10% = 76.2%, Tensor-Guided 10% = 82.1%

  Check the main paper's corresponding table in /home/tony/projects/ai-honesty/papers/sosp/eval.tex (look for tab:confound or the budget curve results). If the numbers differ, determine which is correct (the main paper's eval.tex is authoritative — it was written with the actual experiment data). Update the
   supplement to match, or add a note explaining if the tables show different aggregations (e.g., per-model vs pooled).

  Fix 3: TLA+ Framing

  The TLA+ specifications (lines 65-150) are presented as if they're complete runnable models, but they're excerpts — no Init predicate, no type invariants, IsHonest is called but never defined. The full specs are in the tla/ directory.

  Action: Add a sentence at the start of Section 1 making this explicit: these are excerpts illustrating the modeling approach; full specifications are available in the artifact repository at tla/EpistemicImpossibility.tla and tla/epistemic_tensor.tla. Tony has run TLC on the full specs successfully — the
  validation subsection (lines 143-150) is accurate, it just needs the excerpt framing so a reviewer doesn't try to run the fragments and conclude they're broken.

  Fix 4: Superlinear Verification Cost

  The actual Lean file handles this correctly as an explicit axiom with a comment explaining it's an empirical claim, not a logical theorem. The supplement doesn't mention this. Add a brief note in the Lean section (after the corollary, around line 237) noting that the superlinear verification cost claim
  (Lemma 4.4 in the main paper) is stated as an axiom in the formalization because it is an empirical property of natural language, not a logical necessity. This is a strength, not a gap — it's epistemically honest about what can and cannot be formally proven.

  Fix 5: Compilation Verification

  After making all changes, verify the supplement compiles cleanly: pdflatex supplementary.tex (or whatever build command works — check for a Makefile in the sosp/ directory).

  Judgment Call: Length-Controlled Confound Analysis

  The main paper (eval.tex lines 157-225) has thorough coverage of the length-controlled 2×2 design and the combined classifier collapse result. The supplement currently does not cover this. Consider whether adding a subsection to the supplement with the detailed methodology (query construction for the 4
  cells, word count targeting, the 6 text features tested) would strengthen the submission. If you add it, put it in Section 3 (Experiment Methodology) since the experimental design details belong there. If you judge the main paper's coverage is sufficient, skip it.

  ---
  Task 2: Clean Up the arXiv Rewrite (GATED ON TASK 1)

  Do not start this until Task 1 is complete and verified.

  File: /home/tony/projects/ai-honesty/arxiv/main.tex (34 pages, assembled by a prior instance)

  Context: This arXiv version combines the SOSP submission with the supplement into a single standalone paper, rewritten to avoid trivial unblinding of the SOSP submission. It needs to read as Tony's work in Tony's voice, not as an LLM assembly job.

  Step 1: Read Tony's Voice

  Read these for tone, structure, and rhetorical style:
  - The SOSP submission itself: /home/tony/projects/ai-honesty/papers/sosp/ (start with intro.tex, discussion.tex, conclusion.tex)
  - The thesis arXiv version if accessible: ~/projects/thesis-arxiv/
  - The blueprint: ~/projects/research-program/blueprint.md

  Tony writes as a systems researcher. Direct claims, empirical grounding, architectural framing. Not hedged academic prose, not breathless AI hype.

  Step 2: Verify Formal Content Integrity

  All formal proofs in the arXiv version must match the corrected supplement (from Task 1) and the Lean source (Basic.lean). The prior assembly agent pulled from the old supplement — so the stale Lean excerpts may have propagated. Check every theorem, every TLA+ fragment.

  Step 3: Check Non-Trivial Unblinding

  The arXiv version must not be a trivial reshuffle of the SOSP submission. Compare section-by-section: if entire paragraphs are copy-pasted from the SOSP .tex files, rewrite them. The ideas are the same; the presentation should be distinct. Different framing, different emphasis, Tony's voice rather than
  conference-submission voice.

  Step 4: Structural Review

  The assembly agent reported 8 sections + 4 appendices. Verify:
  - Introduction sets up the problem accessibly (arXiv audience is broader than SOSP)
  - The formal proofs have adequate prose explanation (not everyone reads Lean)
  - The appendices from the supplement are properly integrated, not just stapled on
  - References are complete and correct (references.bib)

  Step 5: Compile and Verify

  Build with pdflatex + bibtex. Check for undefined references, missing citations, figure references that don't resolve. The assembly agent reported it compiles — verify after your changes still compile.

  ---
  Success criteria:
  - Task 1: Supplement compiles, all excerpts match source files, numbers match main paper, honest framing throughout
  - Task 2: arXiv version compiles, formal content verified against corrected supplement, voice is Tony's, no trivial unblinding, reads as a standalone contribution
