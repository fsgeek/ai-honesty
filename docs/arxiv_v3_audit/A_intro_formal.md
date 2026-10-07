# Ledger A: Introduction, Background, The Impossibility (main.tex lines 137-627)

Auditor scope: `/home/tony/projects/ai-honesty/arxiv/main.tex` L137-627.
Primary data: `exp27_bounded_verification_20260206_205725.csv` (800 rows = 4 families x 100 knowable + 100 unknowable; `self_report_confidence` is the corrected column, `self_report_confidence_broken` the original).
Unknowable sub-classes recovered from `scripts/experiment27_bounded_verification.py` `UNKNOWABLE_QUERIES` (25 fictional people/papers, 25 fictional events, 25 future/private, 25 fabricated citations, per family).
All AUCs computed with sklearn `roc_auc_score`, positive class = unknowable.

## Summary of non-VERIFIED items

| Line | Claim (short) | Verdict | Correct value / evidence |
|---|---|---|---|
| 149 | "the model is most confident when it should be most doubtful" | RETRACTED-LEFTOVER | Opposite holds: mean conf knowable 0.975 > unknowable 0.733, in every family (OLMo 0.96/0.65, Llama 0.96/0.64, Qwen 0.99/0.74, Mistral 1.00/0.91). Sentence survives verbatim from the pre-correction version (git `e8bc23d^`, L139). |
| 176 | "To understand why self-reported confidence goes backwards" | RETRACTED-LEFTOVER | Self-report is not inverted (pooled AUC 0.730). Verbatim survivor from pre-correction version (`e8bc23d^`, L166). |
| 193 | Fig. 1 caption: five categories; knowable fact H~0.13; weird truth "moderate"; shattered lie "very high"; unknowable query panel | WRONG | Figure is from `scripts/epistemic_trace_demo.py` (OLMo-3-7B-Instruct), JSON `epistemic_trace_demo_20260316_061220.json`. It has **six** traces, no Private/Future "unknowable query" trace. Means: capital 0.335, H2O 0.111 (neither is 0.13), wombat 1.147 (highest of all, not "moderate"), Tanaka citation 0.653 (and the response is a refusal, "I don't have access..."), Westphalia 0.671 (lower than wombat and Brennan-Kowalski 0.872, so not "very high"). The wombat response says scat is "cylindrical or sausage-shaped", i.e. the "weird truth" panel shows a wrong answer. Panel titles in the rendered PDF show these mean-H values. |
| 208 | "Using response length alone as a signal, AUC was 0.63" | WRONG | Raw word count, exp27: pooled 0.912; per family OLMo 0.887, Llama 0.949, Qwen 0.852, Mistral 0.969. (The binned `length_score` column gives 0.567 pooled / 0.45-0.67 per family; exp21 `baseline_length` gives 0.675 on n=20. No source gives 0.63.) Contradicts the paper's own L855 ("AUC 0.85-0.97 per family"). |
| 211 | "Composed judges ... Pooled AUC never exceeded 0.70" | WRONG | 5-fold CV logistic regression on exp27: hedge+binned length_score 0.768; hedge+log word count 0.944; hedge+log wc+self-report 0.956. Even hedge alone is 0.741 pooled. No source for 0.70 found. |
| 214 | "None of these approaches work reliably" | WRONG (as applied to length) | Length alone, pooled 0.912, beats pooled entropy 0.870; it beats entropy outright on Llama (0.949 vs 0.922) and Mistral (0.969 vs 0.905). |
| 209 | Hedging phrases "are weak signals" | UNSUPPORTED / overstated | hedge_score AUC pooled 0.741; per family 0.595 (Mistral) to 0.877 (Qwen; that is close to Qwen entropy at 0.896). "Weak" holds for Mistral only. Second sentence (model can learn to include them) is an argument, untested. |
| 141 | "no threshold separates them usefully" (self-report) | OVERSTATED | True pooled. Not for OLMo: self-report AUC 0.918 there, higher than OLMo entropy (0.894); "conf < 100" flags 97/100 unknowable at 41/100 knowable FP. The 43% literal-100 rate is pooled; per family it varies widely (OLMo 3%). |
| 141 | 43% literal "100" on "questions about people, papers, and events that do not exist" | MINOR MISDESCRIPTION | 43.25% is over **all** unknowables, a quarter of which are Private/Future (not nonexistent entities). For nonexistent entities only (people+events+citations): 42.3%, so the number barely moves. |
| 241 | fabricated-entity (Glavinsky, Westphalia) average 0.75; Private/Future 0.60 | WRONG (small) | From exp27: people+events = 0.785; people+events+citations = 0.772 (citations alone 0.748); Private/Future = 0.6145 (rounds to 0.61, not 0.60). The commit message for 3488f83 also says 0.751; I could not reproduce it from any grouping I tried. |
| 241 | "only Private/Future ... drops appreciably" | PARTLY MISLEADING | Private/Future still draws a literal 100 on 46% of items, more than the people (47%) vs citations (22%) spread would suggest; the "trained to refuse" explanation is unsupported. |
| 163 | "which is why entropy generalizes where text features diverge" | UNSUPPORTED | On exp27, raw word count per family is 0.85-0.97 vs entropy 0.89-0.92: length generalizes across these families too. I found no in-range evidence of text features diverging. Possibly refers to Exp34 (post-training), which I did not check. |
| 165 | "outperforms text baselines by 3.3-4.6 pp at every budget level" | VERIFIED with caveat | Values hold (fig3 sim: 78.4/82.2/87.6 length vs 81.7/86.8/90.9 entropy => +3.3/+4.6/+3.3). But there is only one text baseline (raw word count), not several. Next to "pooled AUC 0.870", the sentence suggests entropy is the better detector; pooled word-count AUC is 0.912. |
| 165 | "0.757 when response length is held fixed" | VERIFIED with caveat | exp32: 0.757 is the AUC in the **long** cell only (word counts 183 vs 169, roughly matched). Short cell 0.847 (not length-matched: 36 vs 79 words); pooled exp32 0.784. Reporting the matched cell is defensible, but the sentence should say so. |
| 220 | FLP: "in a partially asynchronous distributed system, consensus cannot be achieved" | WRONG | FLP (JACM 1985) is about **fully asynchronous** systems (deterministic consensus with one crash failure). Under partial synchrony, consensus is solvable (Dwork-Lynch-Stockmeyer 1988). Year 1985 is correct. |
| 249 | "Instruction tuning does not fix this ... become better at generating confident-sounding text" | UNSUPPORTED (data mixed) | exp29 (n=20 per model, correct parser): self-report AUC base to instruct: OLMo 0.54 to 1.00 (improves), Llama 0.60 to 0.70, Qwen 1.00 to 0.70, Mistral 0.765 to 0.70. No consistent direction, which matches CLAUDE.md "alignment tax does NOT generalize". |
| 155 | "And our self-report finding suggests why: the problem isn't capability, it's interface" | WEAKENED (borderline leftover) | Written when the finding was inversion. A compressed but above-chance self-report (AUC 0.66-0.92) is weak evidence for "interface, not capability". Lin et al. citation itself is correct. |
| 500-508 | "Empirically ... observation shows a strong bias. Supervisors typically prefer plausible completion over abstention" | UNSUPPORTED | No citation and no data in range. |
| 577 | Composed judge cited as a triage architecture whose second judge "inspects the original artifact" | WRONG characterization | Composed judge (`fig3_budget_curve.py` L104-109; `experiment27_bounded_verification.py`) sets the score of citation-flagged rows from the ground-truth `is_knowable` label (0.0 if knowable, else 1.0). It is an oracle upper bound, not a second judge that inspects anything. |
| 598-608 | Lemma (Superlinear Verification Cost): "|E| grows superlinearly in |V|" | UNSUPPORTED (stated as lemma, no proof) | Asserted with no proof or data. The Lean file encodes it as `axiom superlinear_verification_cost_assumption : True` and calls it "an empirical claim, not a logical necessity". The paper still labels it a Lemma. |
| 534-541 | Monotonicity Lemma: there exist **deterministic** f_i ... "regardless of whether their judgments are deterministic or probabilistic" | FORMAL IMPRECISION | If a judge is stochastic, Obs_{i+1} is a randomized function of Obs_i. That is still a Markov chain, so the data-processing conclusion survives, but the stated "deterministic f_i" does not. Lean `observation_monotonicity` covers deterministic layers only, and only the two-layer case. |
| 559-567 | Corollary induction: S_{m+1} "inherits ... the verification budget B" | NOT FORMALIZED | Lean `layered_judges_cannot_escape` proves only "equal inputs give equal outputs under function iteration". Budget B is not modeled. |
| 467-496 | Thm 2 proof step 5: "output r_fab in w_A (where it may be correct)" | INTERNAL INCONSISTENCY (minor) | Condition 2 defines r_fab as a plausible *fabrication*, while honesty in w_A requires r_corr. The proof needs r_fab = r_corr in w_A, which the definitions never state. Lean sidesteps this: it proves only that the updates are equal (via the assumed `update_depends_on_obs`), not non-convergence to honesty. |
| 612-617 | Corollary (Responsibility Concentration) | NOT DERIVED | Labelled a corollary, but no proof is given and it is not in Lean/TLA+. It is an informal argument. |
| 317 | "The industry trend is toward greater opacity" | UNSUPPORTED in range | Deferred to Sec. discussion; not checked there. |

## Full ledger

| Line | Claim | Source / computation | Verdict |
|---|---|---|---|
| 139 | Asking a model how confident it is "buys almost nothing" | Pooled self-report AUC 0.730; per family 0.658-0.918 | VERIFIED as framing; overstated for OLMo (see summary) |
| 141 | Four families OLMo-3, Llama-3.1, Qwen3, Mistral | exp27 `model_id` | VERIFIED |
| 141 | Mean conf 0.98 knowable / 0.73 unknowable | exp27: 0.975 / 0.733 | VERIFIED |
| 141 | 43% of unknowables answered "100" | exp27: 173/400 = 43.25% (>=0.999) | VERIFIED (description of question set slightly off; see summary) |
| 141 | AUC 0.66-0.92 across architectures | exp27: Llama 0.658, Mistral 0.666, Qwen 0.748, OLMo 0.918; pooled 0.730 | VERIFIED |
| 141 | "no threshold separates them usefully" | OLMo threshold at 100: TPR 0.97, FPR 0.41 | OVERSTATED |
| 143 | "not a lie ... signal with almost no dynamic range" | 84.5% of knowable and 43% of unknowable at literal 100 | VERIFIED (pooled) |
| 146 | Earlier version: AUC 0.28-0.36, "we checked multiple times" | `git show e8bc23d^:arxiv/main.tex` L104/L135; broken column per family: OLMo 0.284, Llama 0.330, Qwen 0.332, Mistral 0.362 | VERIFIED |
| 146 | Bug: decoded prompt + generation, took first integer | commit 3488f83 message and diff | VERIFIED |
| 146 | Zeroed 47% knowable vs 31% unknowable | broken==0: knowable 47.25%, unknowable 31.25% | VERIFIED |
| 146 | Reported by Chris Brew; fixed at source | 3488f83; issue #1 | VERIFIED |
| 146 | Entropy and budget results unaffected | budget pipeline does not read self_report (REPRODUCTION.md; fig3 script) | VERIFIED |
| 149 | "model is most confident when it should be most doubtful" | see summary | RETRACTED-LEFTOVER |
| 155 | Lin et al.: larger models less truthful on TruthfulQA | Lin et al. 2022 (inverse scaling on TruthfulQA) | VERIFIED |
| 155 | "our self-report finding suggests why" | see summary | WEAKENED |
| 161 | Impossibility result for text-only verification under explicit assumptions | Thm 1/2 + Lean `verification_impossibility`, `no_honest_verdict`; `lake build` succeeds (1658 jobs, 2026-10-06) | VERIFIED (conditional formal result) |
| 163 | Exports per-token entropy and log-prob | design section | NOT-CHECKABLE here |
| 163 | Model "cannot separately tune entropy without affecting correctness" under standard training | argument (Sec. design) | NOT-CHECKABLE |
| 163 | "entropy generalizes where text features diverge" | see summary | UNSUPPORTED |
| 165 | Four judges x three budgets (10/20/30%) | fig3_budget_curve.py BUDGET_LEVELS | VERIFIED |
| 165 | Pooled entropy AUC 0.870 | exp27: 0.8703; signal_ablation_results.csv 0.8703 | VERIFIED |
| 165 | 0.757 length-controlled | exp32 long cell 0.7566 | VERIFIED w/ caveat |
| 165 | +3.3-4.6 pp over text baselines at every budget | REPRODUCTION.md table: +3.3/+4.6/+3.3 | VERIFIED w/ caveat (single baseline) |
| 176 | "self-reported confidence goes backwards" | see summary | RETRACTED-LEFTOVER |
| 188 | Entropy 0.24 vs 0.61 nats (grounded vs not) | exp27 mean_entropy: knowable 0.237, unknowable 0.609; computed with natural-log `log_softmax` (nats) | VERIFIED |
| 193 | Fig. 1 caption | see summary | WRONG |
| 199 | "the model may know it is fabricating" | argument | NOT-CHECKABLE |
| 208 | Length "AUC 0.63" | see summary | WRONG |
| 208 | Fabrications tend to be verbose | Word count higher on unknowable (AUC 0.912 in that direction) | VERIFIED |
| 209 | Hedging "weak signals" | hedge AUC 0.595-0.877 | UNSUPPORTED/overstated |
| 210 | Entropy markedly weaker on citation-like queries | exp27 is_citation-flagged rows: entropy AUC 0.635 (n=144) vs non-flagged 0.877 | VERIFIED. Note: the flag is the apostrophe heuristic and only 8 knowable rows are flagged, so the subset is badly imbalanced. |
| 211 | Composed text judges <= 0.70 | see summary | WRONG |
| 214 | "None of these approaches work reliably" | see summary | WRONG |
| 214 | Text channel "the model can fully control through training" | argument | NOT-CHECKABLE |
| 220 | FLP 1985 | Fischer, Lynch, Paterson, JACM 32(2), 1985 | VERIFIED |
| 220 | "partially asynchronous" | see summary | WRONG |
| 220 | FLP indistinguishable states: crashed vs slow | standard reading | VERIFIED |
| 235 | Wombat scat is cube-shaped | established fact | VERIFIED |
| 235 | A day on Venus is longer than a year on Venus | sidereal day ~243 d > orbital year ~225 d | VERIFIED |
| 241 | Knowable 0.98 / unknowable 0.73 | exp27 | VERIFIED |
| 241 | Fabricated-entity 0.75; Private/Future 0.60 | see summary | WRONG (small) |
| 241 | "Nearly half" of unknowables draw 100 | 43.25% | VERIFIED (loose) |
| 243 | Self-report is "almost always near its maximum" | 84.5% knowable / 43% unknowable at 100; median conf for unknowable is high | VERIFIED (pooled) |
| 245 | Obscure grounded facts "no more assertive, and sometimes less" | not tested in range | UNSUPPORTED (mild) |
| 247 | "No training signal" would pair accurate confidence with accurate generation | argument | NOT-CHECKABLE |
| 249 | Instruction tuning makes it worse / doesn't fix | exp29, mixed | UNSUPPORTED |
| 255-257 | Entropy/logprob/attention are byproducts "structurally coupled" and resist manipulation | argument; deferred to Sec. design | NOT-CHECKABLE here |
| 266 | "scaling has not resolved it" (as noted in intro) | rests on Lin et al. | VERIFIED as citation |
| 273 | Lean proofs and TLA+ specs machine-checked; question is fidelity of definitions | Lean builds; TLA+ `EpistemicImpossibility.tla` hard-codes the judge as `text \notin ObviousLies`, so the counterexample holds by construction. The paper's framing ("conditional ... the substantive question is whether the definitions faithfully model") is accurate. | VERIFIED (framing honest) |
| 279-286 | Thm 1 = representational, Thm 2 = learnability | matches the theorems | VERIFIED |
| 288-295 | Impossibility holds "regardless of model family, scale, or training procedure" and is prompt-agnostic | follows from definitions (text-only obs) | VERIFIED as conditional |
| 302-346 | Definitions (observation model, predictor-centric policy, truth status, epistemic honesty) | definitional | NOT-CHECKABLE |
| 314-317 | logprob APIs fall outside text-only class; "industry trend toward greater opacity" | the latter is not supported in range | UNSUPPORTED (2nd sentence) |
| 358-380 | Thm 1 proof: disjoint events, sum >= 2(1-eps) > 1 when eps < 0.5 | arithmetic correct; Lean `representational_impossibility` proves it (normalization `prob_sum_le_one` is taken as a hypothesis, which is fine) | VERIFIED |
| 382-386 | Holds even with retrieval/tool use if obs collapses to text | Note: Thm 1 is about the *policy* conditioning only on q. A retrieval policy conditions on D, so Thm 1 does not apply to it. The remark conflates the policy's input with the supervisor's observation. Sec. 3.3 and the Lean `verification_impossibility` carry the actual argument. | FORMAL IMPRECISION (minor) |
| 389-399 | Yamamoto example | illustrative | NOT-CHECKABLE |
| 401-412 | Non-identifiability; Sampath 1995 diagnosability | bib entry exists (references.bib L356) | VERIFIED |
| 421-430 | RAG grounding is to w_docs, not w | argument; consistent with Lean comment block | NOT-CHECKABLE |
| 440-454 | Bounded supervisor definition | definitional | NOT-CHECKABLE |
| 456-465 | Hallucination regime condition | definitional. Lean's `BoundedSupervisor.indistinguishable` quantifies over **all** AmbiguityConditions (stronger than the paper's "there exists"). | NOT-CHECKABLE (note) |
| 467-498 | Thm 2 + proof | Steps 1-4 match Lean `learnability_impossibility`. Lean assumes `update_depends_on_obs`, which makes the result near-definitional. Step 5 inconsistency on r_fab. | VERIFIED (conditional) w/ inconsistency |
| 500-509 | "Supervisors typically prefer plausible completion" | none | UNSUPPORTED |
| 511-523 | Scope: deployment-time | definitional | NOT-CHECKABLE |
| 534-553 | Observation Monotonicity | Lean covers deterministic, 2-layer | FORMAL IMPRECISION |
| 555-570 | Layered-judge induction | Lean `layered_judges_cannot_escape` | VERIFIED (weak form) |
| 572-582 | Triage architectures, incl. "our own composed judge", fall outside | composed judge is an oracle | WRONG characterization |
| 590-596 | Composition graph | definitional | NOT-CHECKABLE |
| 598-608 | Superlinear cost "Lemma" | Lean axiom `True` | UNSUPPORTED |
| 612-623 | Responsibility Concentration corollary; attestation reinforces it | informal | NOT-CHECKABLE / not derived |

## What I could not check
- The Exp34 basis (if any) for "entropy generalizes where text features diverge" (L163).
- Section discussion support for the "trend toward opacity" (L317).
- How 0.751 (fabricated-entity mean, commit 3488f83) was computed. No grouping of the exp27 data I tried reproduces it.
- I did not run TLC. The TLA+ judgment above comes from reading the spec. The Lean project was rebuilt and builds cleanly.

## Resolution, Oct 6 2026: the 0.751 fabricated-entity figure

Not reproducible. Commit 3488f83 and the issue #1 reply quote 0.751. The Aug 28 session
says it came from a query "classification", but that code was not kept. No combination
of the four subgroups in the exp27 script, and none of three keyword heuristics, gives 0.751.
Use `scripts/self_report_by_subgroup.py`, which takes its labels from the script's own
subgroup comments: people 0.834, events 0.735, citations 0.748, future/private 0.614,
fabricated entities pooled 0.772 (n=300), knowable 0.975. The conclusion is unchanged:
every fabricated subgroup is below knowable, so "confidence peaks on plausible lies" fails
at every grain. Line 241's "0.75 / 0.60" becomes "0.77 / 0.61". The v3 erratum and the
follow-up to Chris should carry the corrected number.
