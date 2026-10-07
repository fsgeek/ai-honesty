# Ledger B: arxiv/main.tex lines 628-919 (Tensor Interface Design + Empirical Cost Surface)

Auditor scope: errors that were wrong when written. The paper was not edited.
Computation scripts are in `scratchpad/b/` (fig3.py is an unmodified copy of the artifact's
`fig3_budget_curve.py` with only the output paths changed; a1-a4.py hold the checks).
Python: /home/tony/projects/ai-honesty/.venv/bin/python.

## SUMMARY: non-VERIFIED items (most severe first)

| # | Line(s) | Claim (short) | Verdict | Correct value / note |
|---|---|---|---|---|
| 1 | 853 | "On our balanced test set, no single text feature exceeds AUC 0.70" | WRONG / RETRACTED-LEFTOVER | On exp27, raw word count AUC is **0.912 pooled** (0.852-0.969 per family), and the hedge_score column is 0.741 pooled (up to 0.877 for Qwen). Line 855 contradicts it two lines later. The "<0.70" claim comes from the SOSP draft's exp32 **long-cell length-controlled** table (papers/sosp/eval.tex:180-205) and was moved to the wrong dataset. Only the pre-binned `length_score` column stays under 0.70. |
| 2 | 890 | "Text-guided triage hits a ceiling around 87-88% because length is a weak signal under fair comparison; the tensor advantage (+3-5pp) is consistent and architecturally driven" | WRONG / RETRACTED-LEFTOVER | No ceiling: the same seeded sim gives length **94.0 at 40% and 96.4 at 50%**, tying entropy (94.2, 96.5) and passing it at 60% (97.8 vs 97.5). Length is not weak (see #1). Per family the advantage is not consistent (see #3). |
| 3 | 842 (and 847 caption by implication) | "The advantage is consistent across all four model families" | WRONG | Per-family runs of the same sim (budget = fraction of each family's 200 rows), length vs entropy at 10/20/30%. **Mistral: 71.9/79.5/87.7 vs 72.0/80.0/85.6, so length wins at 30% by 2.1pp** and the 10/20% gaps are within 0.5pp. OLMo, Llama and Qwen favour entropy. Spearman rho 0.762 is cited as support but measures something else. |
| 4 | 851-855 | "Why tensor wins": explains the budget-curve result with knowable-vs-unknowable AUCs | TARGET CONFLATION | The curve rewards catching **wrong answers** (`~is_correct`, 194 rows, 31 of them knowable). The AUCs quoted use `is_knowable`. Against the curve's own target: entropy **0.841** vs word count **0.774** pooled. Per family entropy/wc: OLMo .807/.718, Llama .831/.747, Qwen .844/.682, **Mistral .891/.931**. That is the actual reason entropy wins the pooled curve. The paper's "length beats entropy on 2 of 4" holds only for the is_knowable target. |
| 5 | 802-822 (whole cost-surface description) | Accuracy numbers presented as judge outcomes; "Verified fabrications are replaced with abstention" | UNSUPPORTED (method undisclosed) | The table comes from a **Monte Carlo simulation** (1000 trials, seed 42) with **assumed** verifier success rates: P(catch fabrication)=0.938, borrowed from the human-agreement rate, and P(retrieve answer)=0.95. Verification also repairs wrong *knowable* answers by retrieval, not only by abstention. The words "Monte Carlo", "simulat", 0.938 and 0.95 appear nowhere in main.tex. |
| 6 | 813 | Tensor judge "selects outputs based on per-token entropy (mean and variance)" | WRONG | The judge ranks by `mean_entropy` only: `state_object_score` in fig3_budget_curve.py and `tensor_uncertainty_score` in experiment27_realistic_verification.py:50-52. |
| 7 | 882 | "Our initial evaluator (substring matching + refusal marker detection) achieved 68.8% accuracy" | WRONG | 68.75% is `old_correct.mean()` over the 800 rows, i.e. the share of responses the old evaluator **scored as correct** (model accuracy under that evaluator). It is not the evaluator's accuracy. On the 80-item human sample the old evaluator agrees with the human **70/80 = 87.5%**, and it agrees with the new evaluator on 89.3% of all 800 rows. |
| 8 | 882 | Example: failed to recognize "not a real syndrome" | UNSUPPORTED | No exp27 query or response contains "syndrome" (grep count 0). The example comes from a different experiment. |
| 9 | 904 | "all five discriminate ... AUC ranging 0.65-0.88" | WRONG | DeepSeek-V3 returned logprobs for **1 token** per response (num_tokens=1), giving AUC 0.607. docs/experiment31_together_api_analysis.md excludes it as "not comparable". The 0.651-0.875 range is over Maverick, Mistral-Small, Qwen3-235B, Gemma **and Mistral-7B (the calibration model)**, so only 4 of the 5 "additional" models have usable data. |
| 10 | 904 | "Mixture-of-experts models show better performance with peak entropy than mean" | WRONG | Llama-4-Maverick (MoE) improves (0.651 to 0.899). **Qwen3-235B (MoE) gets worse (mean 0.875, max 0.853).** DeepSeek has no usable data. The dense models also improve with max (Mistral-7B .860 to .917, Mistral-Small .850 to .909, Gemma .765 to .855), and the analysis doc itself says "not purely an MoE effect". |
| 11 | 704, 735, 671, 908 | Entropy is "not optimized by any standard training objective"; "independent of any training signal"; "Entropy is architectural"; "not an artifact of specific training procedures ... The signal is architectural" | UNSUPPORTED / contradicted by own data | Entropy is a function of the next-token distribution that cross-entropy and post-training shape directly. Line 656 says so itself. The project's Exp34 (`tab_pipeline_stages.py`, OLMo-3) shows knowable entropy falling from 1.18 to 0.18 nats and AUC rising from **0.666 (base) to 1.000 (SFT)**, so discrimination depends heavily on training stage. Line 735's intended meaning (entropy is not used as a training target here) is fine. As written, it is false. |
| 12 | 660 | Model "cannot learn to" produce low entropy while outputting wrong tokens | UNSUPPORTED (overclaim) | Counterexamples exist in exp27: wrong answers at 0.10-0.17 nats (e.g., Qwen and Llama on "largest desert on Earth", Qwen "country with the most people"). Rare (7/194 wrong answers fall below the correct-answer median), but "cannot" is too strong, and line 872 says "confidence and correctness come apart". |
| 13 | 641 | Attention summaries "Measured but not used in the bounded verification experiment" | WRONG | experiment27_bounded_verification.py never extracts attention (no `output_attentions`), and the exp27 CSV has no attention columns. Attention was measured only in the overhead benchmark and in earlier topological experiments. |
| 14 | 636, 645 | "In our implementation, we export ... Provenance markers" | WRONG / internally inconsistent | tensor_interface.py exports no provenance. Line 742 says "Provenance Binding is not implemented". |
| 15 | 650 | "(see benchmark details in Appendix)" | WRONG (dangling) | The appendix (line 1134 on) holds only TLA+ and Lean sections, with no overhead benchmark. Also note: the 2.4%/7.1% figures were measured on one model (Qwen3-4B **base**, 64 tokens, 20 prompts x 10 iterations), not on the four exp27 models. |
| 16 | 748 | Entropy is "widely available (exposed by most model APIs)" | UNSUPPORTED | APIs expose top-k logprobs at best, and some expose none. The paper's own API experiment had only top-5 (DeepSeek: 1 token). This contradicts lines 762 and 904. |
| 17 | 783 | Unknowable set includes "fictional diseases" | WRONG (minor) | The exp27 unknowable set has 25 fabricated research concepts, 25 fabricated treaties/events, 25 private/future questions and 25 fabricated citations. **No disease queries.** Line 781's knowable list ("geography, history, science, biography") also leaves out the 20 arithmetic and 20 weird-truth items. |
| 18 | 855 | Length-controlled study: queries "matched for response length; entropy retains AUC 0.757" | IMPRECISE | 0.757 is the **long-cell stratum only**. All of exp32 pooled is 0.784. The short cells are *not* length-matched (35.7 vs 79.3 words; word-count AUC 0.78 there). In the long cell, knowable answers run slightly longer (wc AUC 0.369). The conclusion holds; the description overstates the matching. |
| 19 | 870, 874 | "flagged fabrications still carry higher mean entropy than the non-fabricated rows in that slice (0.575 vs 0.436)"; "slice is almost entirely fabrications" | IMPRECISE (numbers verified) | 0.575 is the mean over all 136 **unknowable** citation-slice rows and 0.436 over the 8 **knowable** rows. Taking actual fabrications (unknowable and judged wrong, n=80) gives 0.645 vs 0.470 for the rest. The direction holds. Of 136 fabrication *prompts*, 56 were answered correctly (refused), so the slice is "almost entirely fabrication prompts", not fabrications. |
| 20 | 913 (Fig. crossmodel) | Heatmap of pairwise rho | WRONG (figure labels) | The mean rho of 0.762 is correct. The heatmap's cells, however, are in alphabetical order (Llama, Mistral, OLMo, Qwen) while the axis labels read OLMo, Llama, Qwen, Mistral. Example: the figure shows OLMo-Mistral 0.80, but the data give OLMo-Mistral 0.739 and Llama-Qwen 0.798. The script that generated it could not be found, so this was inferred from the matrix. |
| 21 | 884, 795 | 93.8% agreement "with the independent LLM evaluator"; it "provides independent cross-validation of this dual role" | IMPRECISE | 93.8% is the agreement of the whole stratified evaluator (programmatic 34/37, LLM 41/43). The LLM evaluator is Qwen3-4B-Instruct, one of the test models, so it is not independent. The human annotator is an author. No Qwen self-favoring is visible: on Qwen rows the human agrees 18/19. |
| 22 | 654 | Model "cannot independently tune these signals without affecting correctness" | NOT-CHECKABLE | Theoretical assertion with no experiment. Hedged by the next sentence. |
| 23 | 687 | "RAG still fabricates citations" | NOT-CHECKABLE | Literature claim with no citation; no data in the repo. |
| 24 | 758 | Budget scenarios (medical 30%+, support 10-15%) | NOT-CHECKABLE | Illustrative. |
| 25 | 870 | Two consecutive sentences say the same thing ("separates ... far less well" / "discriminates far more weakly") | EDITORIAL | Duplicate sentence. |

## FULL LEDGER

### Section 4: Tensor Interface Design (628-762)

- **L639** per-token entropy is the primary signal in evaluation: VERIFIED (all judges use `mean_entropy`).
- **L641** attention "Measured but not used in the bounded verification experiment": WRONG, see #13.
- **L643** log-probabilities "Measured for analysis": VERIFIED (`mean_logprob` and `mean_top5_mass` are in the exp27 CSV).
- **L645** provenance markers exported "in our implementation": WRONG, see #14.
- **L650** entropy overhead ~2.4%: VERIFIED. benchmark_tensor_overhead_20260214_091023.csv mean wall-clock Entropy/Baseline = +2.41% (median +2.29%).
- **L650** full signal set ~7.1%: VERIFIED (+7.12% mean, +6.95% median). Caveat: Qwen3-4B base only.
- **L650** "see benchmark details in Appendix": WRONG, see #15.
- **L650/814** "experiments use entropy only, yielding 2.4%": VERIFIED (consistent with the judge code).
- **L654** coupling claim: NOT-CHECKABLE (#22).
- **L660** cannot be low-entropy and wrong: UNSUPPORTED (#12).
- **L671** "entropy generalizes across architectures while text features diverge ... Entropy is architectural": UNSUPPORTED/overstated. On the is_knowable target, word count is strong in every family (0.85-0.97) and does not diverge. Hedge_score does diverge (0.595-0.877). On the wrong-answer target, wc ranges 0.68-0.93 against entropy's 0.81-0.89, so there is partial support there. "Architectural" is contradicted by Exp34 (#11).
- **L687** RAG fabricates citations: NOT-CHECKABLE (#23).
- **L704** entropy and attention "not optimized by any standard training objective": UNSUPPORTED (#11).
- **L735** "Entropy is independent of any training signal": UNSUPPORTED as worded (#11).
- **L737/742** provenance tier modelled, not implemented; composed row uses an oracle: VERIFIED (the composed score sets citation rows from `is_knowable`).
- **L740** four judges x three budgets x four architectures: VERIFIED (the curve pools the four architectures).
- **L748** entropy "exposed by most model APIs": UNSUPPORTED (#16).
- **L754** "Entropy weakens on citations": VERIFIED directionally. Citation-slice AUC 0.635 vs 0.877 off-slice (8 positives, low confidence).
- **L758** budget scenarios: NOT-CHECKABLE (#24).

### Section 5: Empirical Cost Surface (767-915)

- **L779-783** 200 queries, 100 knowable and 100 unknowable: VERIFIED (per family; 800 rows over 4 families).
- **L781** knowable topic list: incomplete (omits math and weird truths) (#17).
- **L782** "fictional diseases": WRONG (#17).
- **L789-792** four models: VERIFIED. model_ids are olmo-3-7b-instruct, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507, Mistral-7B-Instruct-v0.3.
- **L795** Qwen3-4B-Instruct is the Tier-2 classifier: VERIFIED (`EVALUATOR_MODEL` in experiment27b_stratified_evaluator.py:241).
- **L795** 93.8% human agreement on blinded samples: VERIFIED. 75/80; the sample is blinded to model and auto-judgment (RandomState(42)). "Independent" is overstated (#21).
- **L802** no judge 75.8%: VERIFIED (re-ran fig3 script).
- **L805** length 78.4-87.6%: VERIFIED.
- **L806** length selects by word count: VERIFIED (raw `split()` word count).
- **L807** "just count tokens": counts words, not tokens. Trivial.
- **L809** verified fabrications replaced with abstention: INCOMPLETE/UNSUPPORTED (#5).
- **L811** entropy 81.7-90.9%: VERIFIED.
- **L813** "mean and variance": WRONG (#6).
- **L814** ~2.4% overhead: VERIFIED (caveat #15).
- **L816-818** composed 81.5-91.5%, oracle on citation rows, upper bound: VERIFIED.
- **L822** budget levels 10/20/30: VERIFIED.
- **Table (L832-835)** all 12 cells: VERIFIED. Re-run today, seed 42, 1000 trials, random tie-break: 75.8 / 78.4, 82.2, 87.6 / 81.7, 86.8, 90.9 / 81.5, 87.1, 91.5.
- **L838/842** "+3.3-4.6 pp at every budget level": VERIFIED (aggregate gaps 3.3, 4.6, 3.3).
- **L842** "consistent across all four families": WRONG (#3).
- **L842** Spearman rho = 0.762: VERIFIED (mean of the 6 pairwise per-query Spearman rhos over 200 queries = 0.7616). Interpretation caveat: entropy correlates with word count at rho 0.70-0.81 within each family, so cross-model agreement could also reflect shared response-length properties.
- **L847** caption "dominates at every budget level": VERIFIED for the aggregate at 10/20/30. It stops holding at 60% (#2) and per family for Mistral (#3).
- **L853** "no single text feature exceeds AUC 0.70": WRONG / RETRACTED-LEFTOVER (#1).
- **L855** pooled entropy AUC 0.870, per family 0.894-0.922: VERIFIED (OLMo .894, Qwen .896, Mistral .905, Llama .922).
- **L855** length AUC 0.85-0.97 per family, beating entropy on 2 of 4: VERIFIED (Qwen .852, OLMo .887, Llama .949, Mistral .969). Note: pooled length 0.912 also beats pooled entropy 0.870, which the paper does not mention.
- **L855** length-controlled 0.757: VERIFIED as the long-cell value; description imprecise (#18).
- **L862** caption: unknowable mean 0.61 vs knowable 0.24 nats: VERIFIED (0.609 / 0.237; natural-log entropy). "Separate consistently across architectures": VERIFIED (each family's unknowable mean exceeds its knowable mean; Qwen 0.31 vs 0.09).
- **L870** slice holds 144 responses, 8 knowable, two queries via the apostrophe heuristic: VERIFIED ("We the People" and "polar bear's skin", times 4 families).
- **L870** per-family AUCs rest on two positives each: VERIFIED (36 rows and 2 knowable per family).
- **L870** 0.575 vs 0.436: numbers VERIFIED, labels imprecise (#19).
- **L874** composed 91.5 vs entropy 90.9 at 30%: VERIFIED.
- **L874** detector catches possessives: VERIFIED.
- **L874** no real-citation control: VERIFIED (no citation-style knowable query).
- **L882** 68.8%: WRONG (#7).
- **L882** "not a real syndrome": UNSUPPORTED (#8).
- **L884** 80 randomly selected items, 75/80, 93.8%, 2 auto-too-generous and 3 auto-too-strict: VERIFIED (exp27b_human_review_filled.csv; RandomState(42)).
- **L884** "independent LLM evaluator": IMPRECISE (#21).
- **L890** 87.6 / 90.9 / 91.5 at 30%: VERIFIED.
- **L890** "ceiling around 87-88% ... weak signal ... consistent and architecturally driven": WRONG (#2).
- **L894-902** six API models (five plus one calibration): VERIFIED (exp31_frontier_api_*.csv; 80 probes each).
- **L904** top-5 entropy is a lower bound on full-vocabulary entropy: VERIFIED mathematically. Renormalized top-k entropy is at most the full entropy because every tail token has p at or below the smallest top-5 probability, so H_full - H_5 >= -ln P >= 0.
- **L904** "all five ... 0.65-0.88": WRONG (#9).
- **L904** MoE peak beats mean: WRONG (#10).
- **L906** API rho = 0.36: VERIFIED for the set {Gemma, Maverick, Mistral-7B, Mistral-Small, Qwen3-235B} (mean pairwise 0.360). With DeepSeek included it is 0.284. Note: exp31 "knowable" = control + wombat over 80 probes, a different label and probe set from exp27.
- **L908** "signal is architectural; not an artifact of training procedures": UNSUPPORTED (#11). The same paragraph's Maverick mean AUC (0.651) also weakens "generalizes".
- **L913** caption rho 0.762: VERIFIED. Heatmap labels WRONG (#20).

## What I could not check
- How the entropy-distribution figure (L862) and crossmodel figure (L913) were generated (no script found). Only the PDF text was inspected.
- Whether the per-family budget reading in #3 is what the authors meant. The paper reports no per-family budget numbers, so I used the natural per-family construction (same sim, each family's 200 rows).
- Theoretical claims in Section 4 (L654-669, L675-726). These are design arguments, not empirical claims, and were judged only for consistency with the paper's own data.
