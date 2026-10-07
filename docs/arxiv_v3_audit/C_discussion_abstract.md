# Ledger C: Abstract/front matter (L1-136) + Limitations, Related Work, Conclusion, Acks (L920-1137)

Paper: /home/tony/projects/ai-honesty/arxiv/main.tex (unedited). All computations used
/home/tony/projects/ai-honesty/.venv/bin/python on the committed CSVs named below.

Primary data:
- D27 = /home/tony/projects/pacmi26-observability/data/exp27_bounded_verification_20260206_205725.csv (800 rows, 4 families x 200; corrected `self_report_confidence`, `self_report_confidence_broken` kept)
- D32 = /home/tony/projects/ai-honesty/exp32_length_controlled_20260217_203103.csv
- D31 = /home/tony/projects/ai-honesty/exp31_frontier_api_20260216_{105455,135647}.csv, plus docs/experiment31_together_api_analysis.md
- Budget curve = /home/tony/projects/pacmi26-observability/REPRODUCTION.md (deterministic `make figures` values)

## Summary: items that are not VERIFIED

| # | Line | Claim (short) | Verdict | Correct value / problem |
|---|---|---|---|---|
| 1 | 107-108 | "any threshold admitting most correct answers admits most fabrications too" | WRONG | Pooled: a threshold at 100 admits 84.5% of knowable answers and only 43.2% of unknowable ones. OLMo at 100: 59% knowable vs 3% unknowable. The claim holds only for thresholds of 0.95 or lower (95.8% vs 56.0%). |
| 2 | 112-113 | "We prove ... no monitoring system can reliably distinguish honest outputs from plausible fabrications" | UNSUPPORTED (as a LaTeX theorem) | Neither theorem in the paper states this. Thm 1 is about a *policy* that conditions on q. Thm 2 is about *learning*. Observer indistinguishability is assumed, not proved: it is item 2 of Condition 2. Commit 7d78b8d says the verification theorem is only in Lean (`EpistemicProofs/Basic.lean`) and that "promoting verification_impossibility into the LaTeX ... [is] not yet done." |
| 3 | 115-118 | "no learning algorithm ... can converge to honest behavior" | OVERCLAIM vs body | Thm 2 (L467-472) was softened to "cannot *guarantee* convergence". The abstract still says "cannot converge". |
| 4 | 118-119 | "hold regardless of ... training procedure, including RLHF and instruction tuning" | OVERCLAIM / tension with body | The Scope remark (L511-523) says training-time RLHF annotation is a different cost regime and is "outside this scope". The abstract puts RLHF inside the scope. |
| 5 | 110 / 1100 | Self-report compression "is not a capability gap but an observational one"; "we proved this is architectural" | OVERCLAIM | Neither theorem says anything about self-report. Self-report compression is an empirical observation, and L146 itself says "the impossibility results do not depend on" it. "Proved this" ties the proof to the wrong claim. |
| 6 | 124-127 | Entropy AUC 0.870 "outperforms all text baselines by 3.3-4.6pp" | MISLEADING / partly UNSUPPORTED | (a) The budget curve has one text baseline (raw word count), so "all text baselines" is wrong. (b) The 3.3/4.6/3.3 values check out as *budget-accuracy* gaps. But the sentence sits next to the AUC, and **pooled raw-word-count AUC on the same set is 0.912, above entropy's 0.870**. Per family, length beats entropy on Llama (0.949 vs 0.922) and Mistral (0.969 vs 0.905). The abstract leaves this out. |
| 7 | 124 | "pooled AUC 0.870 on that set" | EDITORIAL | "That set" has no antecedent in the abstract. The number itself is VERIFIED. |
| 8 | 122-123 / 1102 | entropy "structurally coupled to correctness" | TENSION with body | L966 says the entropy signal "measures generation confidence, not factual truth". L877 says the same. Abstract and conclusion state the stronger version. |
| 9 | 128 / 1102 | "generalizes across architectures (Spearman rho = 0.762)" | OVERCLAIM (number VERIFIED) | rho = 0.762 is agreement on per-query entropy among the 4 local models only (pairwise 0.739-0.798). On the API set the paper's own rho is 0.36. Agreement is also not the same thing as generalization of discrimination. |
| 10 | 93 | `\date{March 2026}` | STALE | The text carries the Aug-28-2026 v3 corrections (commit e8bc23d) but the date still reads March 2026. |
| 11 | 929 | "API validation spans five additional models (DeepSeek-V3, Mistral-Small-24B, Llama-4 Maverick, Qwen3-235B, Gemma 3n) up to 671B parameters" | WRONG | DeepSeek-V3 returned logprobs for **1 token only** on all 80 probes (D31 `num_tokens`=1). The project's own analysis doc marks it "Not comparable" (mean-entropy AUC 0.607). The "five models with full logprobs" in that doc are Mistral-7B (the calibration model), Mistral-Small-24B, Maverick, Qwen3-235B and Gemma-3n. Validly tested additional models: **four**. Largest: **Qwen3-235B (235B total, 22B active)**, not 671B. rho = 0.36 (L906) reproduces only when DeepSeek is excluded and Mistral-7B included. The same error is at L896 and L901-902 ("all five ... AUC 0.65-0.88"). With DeepSeek it is 0.607. The doc's figure is 0.651-0.875. |
| 12 | 929 | "The pattern is consistent" | OVERCLAIM | API mean-entropy AUCs: Maverick 0.651, Gemma 0.765. Cross-model rho is 0.36. With max/std aggregation every model clears 0.85, so "consistent" needs a qualifier about aggregation. |
| 13 | 931, 985, 994 | The three principles "are necessary for escaping the text-only impossibility" | UNSUPPORTED | No necessity proof anywhere in the paper. L720-724 says State Exteriority escapes Thm 1 and Verification Independence escapes Thm 2. Provenance Binding escapes neither ("creates a tier"), so the paper's own account shows Provenance is not necessary for escape. |
| 14 | 938 | "Entropy is more robust than text-only approaches" | UNSUPPORTED (partial) | Robustness to training or adversaries was never measured. The only support is length control: in D32, entropy holds AUC 0.757 on the long cell while raw word count drops to 0.37. On the main set, length's AUC is *higher* than entropy's. |
| 15 | 962 | "current frontier and reasoning models typically export nothing beyond text" | OVERSTATED / NOT-CHECKABLE here | Reasoning models (o-series) and Anthropic models do withhold logprobs. Several current frontier chat APIs (e.g. OpenAI's non-reasoning GPT-4o/4.1 family, Gemini) expose top-k logprobs. Needs a citation or a narrower wording. |
| 16 | 962 | "the models deployed in the highest-stakes settings provide the least epistemic telemetry" | UNSUPPORTED | No evidence or citation. |
| 17 | 970 | Refusals are low-entropy and "entropically indistinguishable from confident correct answers"; the signal "degrades" as models get more honest | UNSUPPORTED / contradicted by D27 | In D27, unknowable responses containing refusal phrasing (regex heuristic, n=115/400) have mean entropy **0.507**, against **0.237** for knowable responses. They are not indistinguishable. The PACMI artifact's Exp34 does show `private` prompts collapsing to low entropy after SFT. But there the knowable-vs-fabrication AUC goes 0.67 to 1.00 and the broad AUC goes 0.71 to 0.81: it rises, it does not degrade. The degradation claim is untested in this paper. |
| 18 | 1043-1046 | Bai et al. 2022 shows "tension between helpfulness and honesty" and that "preference optimization can degrade factual accuracy" | ATTRIBUTION QUESTIONABLE | Bai et al. 2022 is about the helpfulness vs **harmlessness** tension. It reports RLHF *improving* honesty/TruthfulQA at large scale. Gao et al. 2023 is about reward-model overoptimization of a proxy, not factual accuracy. Perez et al. (sycophancy) is apt. |
| 19 | 1051-1053 | Kadavath et al. 2022 cited for "verbalized uncertainty ... through the text channel" | ATTRIBUTION WRONG | Kadavath's main methods are P(True)/P(IK), which are *token probabilities* (a logprob signal, closer to the tensor channel). Verbalized-confidence work is Lin et al. 2022 ("Teaching models to express their uncertainty in words") or Tian et al. 2023. |
| 20 | 1053-1060 | Kuhn et al. semantic entropy "remains text-channel: the clustering operates on generated text" | IMPRECISE | The original semantic entropy (Kuhn et al., ICLR 2023) weights clusters by *sequence log-likelihoods*, which is a logprob signal. Only the "discrete" variant (Farquhar et al., Nature 2024) is purely text-sampling. |
| 21 | 1077-1079 | Paxos and Raft work around FLP "by relaxing assumptions or strengthening guarantees" | WRONG (minor) | They work around FLP by assuming partial synchrony for liveness (or by randomization) and *weakening* the guarantee: safety always, liveness only eventually. "Strengthening guarantees" has the direction backwards. Related, outside my range: L220 says FLP holds "in a partially asynchronous" system. FLP is for *fully asynchronous* systems. Under partial synchrony consensus is solvable (Dwork-Lynch-Stockmeyer 1988). |
| 22 | 1005 vs 130 | "Our contribution is a formal impossibility result" vs abstract "The core contribution is a cost surface" | INCONSISTENT | The abstract (L130) and conclusion (L1104) name the cost surface as the core contribution. Related Work (L1005) and its closing line (L1092) name the formalization. |
| 23 | 936 | Winninger et al. "shows that internal signals are rich enough to be attacked" | STRETCHED | That paper uses interpretability to *craft* jailbreak attacks on LLMs. It does not attack exported internal signals as a monitoring channel. |
| 24 | 1024 | Lu et al. 2026 "Assistant Axis" shows "internal persona state diverges from text behavior" | NOT-CHECKABLE (offline) | The bib entry (arXiv 2601.10387, Lindsey et al.) looks plausible. The specific "diverges from text behavior" reading needs checking against the paper. |
| 25 | 1120 | Acks: Claude "ran and analyzed the post-training pipeline experiment" | LEFTOVER | The post-training pipeline experiment (Exp34) is PACMI §3.2. It appears nowhere in this paper. This line looks copied from the PACMI acknowledgments. "the artifact" (L1120) is also never defined, though L1340 has an "Artifact Repository" subsection. |
| 26 | 1121-1122 | "The authors ... recomputed every reported number from the committed data" | CONTRADICTED | Items 1 and 11 above fail recomputation from committed data. So do out-of-range items: L853 "no single text feature exceeds AUC 0.70" (raw word count is 0.912 pooled; hedge_score is 0.877 on Qwen) and L890 "length is a weak signal under fair comparison". |

### Out-of-range findings noticed while tracing (for the other auditors)
- **L150: RETRACTED-LEFTOVER.** "Except the model is most confident when it should be most doubtful." This paraphrases the retracted inversion claim and sits right after the Correction remark. Corrected data: unknowable 0.733 < knowable 0.975.
- **L853: WRONG.** "On our balanced test set, no single text feature exceeds AUC 0.70." Raw word count gives 0.912 pooled and 0.852-0.969 per family. hedge_score gives 0.877 on Qwen and 0.799 on Llama. L855 contradicts L853 two lines later.
- **L211: WRONG / UNSUPPORTED.** "Pooled AUC never exceeded 0.70" for composed text features. Raw length alone exceeds this. Source not traced.
- **L890: WRONG.** "Text-guided triage hits a ceiling ... because length is a weak signal under fair comparison." Length is the strongest single ranker by AUC.
- **L896/901-902: WRONG.** DeepSeek-V3 is listed as validated, and "all five ... 0.65-0.88" (see item 11).
- **L220: WRONG.** FLP described as "partially asynchronous" (see item 21).
- **L146:** the inversion AUC "0.28-0.36" is VERIFIED against `self_report_confidence_broken` (OLMo 0.284, Llama 0.330, Qwen 0.332, Mistral 0.362). CLAUDE.md's "0.28-0.46" differs and may refer to an older dataset (exp24). The 47%/31% zeroing is VERIFIED (47.25% / 31.25%).
- **Header comment L5** says "SOSP 2026 submission". CLAUDE.md says SOSP 2025. The experiments are dated Feb 2026, so SOSP 2026 is the consistent reading. CLAUDE.md is probably the stale one. The comment is not rendered.

## Full ledger

### Front matter and abstract (L1-136)

| Line | Claim | Source | Verdict |
|---|---|---|---|
| 5 | "expanded from SOSP 2026 submission" (comment) | experiment dates | NOT-CHECKABLE (consistent with data dates; contradicts CLAUDE.md "SOSP 2025") |
| 91 | Authors/affiliations (Mason; Anand, MPI-SWS) | — | NOT-CHECKABLE |
| 93 | Date March 2026 | git log (v3 edits 2026-08-28) | STALE (item 10) |
| 102-103 | Four families OLMo-3, Llama-3.1, Qwen3, Mistral | D27 `family` | VERIFIED |
| 104 | mean 0.98 knowable | D27: 0.9754 | VERIFIED |
| 104-105 | 0.73 unknowable | D27: 0.7329 | VERIFIED |
| 105-106 | 43% of unknowable get literal "100" | D27: 173/400 = 43.25% exactly 1.0 | VERIFIED (it is 43% of query-by-model responses, not of distinct questions) |
| 106 | AUC 0.66-0.92 | D27: Llama 0.658, Mistral 0.666, Qwen 0.748, OLMo 0.918; pooled 0.730 | VERIFIED |
| 106-108 | any threshold admitting most correct admits most fabrications | D27 threshold sweep | WRONG (item 1) |
| 110 | "not a capability gap but an observational one" (about self-report) | theorems | OVERCLAIM (item 5) |
| 112-113 | no monitoring system can distinguish (proved) | Thm 1/2, Cond 2, commit 7d78b8d | UNSUPPORTED as proved (item 2) |
| 113-115 | Result 1: query-only policy cannot be honest across ambiguous worlds | Thm 1 L358-386 | VERIFIED (matches Thm 1) |
| 115-118 | Result 2: no learning algorithm can converge | Thm 2 L467 ("guarantee") | OVERCLAIM (item 3) |
| 118-119 | regardless of scale/training incl. RLHF | Scope remark L511-523 | OVERCLAIM (item 4) |
| 122-123 | exports per-token entropy and logprob distributions | §5 L636-646 | VERIFIED |
| 123 | "structurally coupled to correctness" | L966 | TENSION (item 8) |
| 124 | pooled AUC 0.870 | D27 pooled mean_entropy: 0.8703 | VERIFIED (dangling "that set", item 7) |
| 124-125 | 0.757 length-controlled | D32: long (length-matched) cell 0.7566; pooled D32 0.784; short cell 0.847 | VERIFIED (it is the long-cell AUC, n=200. Short cell is not length-matched: median 32.5 vs 75.5 words. Worth saying which cell.) |
| 126-127 | outperforms all text baselines by 3.3-4.6pp at 10/20/30% | REPRODUCTION.md: +3.3/+4.6/+3.3 | numbers VERIFIED; "all text baselines" and AUC juxtaposition MISLEADING (item 6) |
| 128 | Spearman rho = 0.762 | D27 per-query mean_entropy, pairwise across 4 families: 0.739-0.798, mean 0.7616 | VERIFIED number; "generalizes" OVERCLAIM (item 9) |
| 130 | Core contribution is the cost surface | — | INCONSISTENT with L1005 (item 22) |

### Limitations and Open Questions (L920-998)

| Line | Claim | Source | Verdict |
|---|---|---|---|
| 927 | existence, not optimality | — | VERIFIED (scoping statement) |
| 929 | local experiments span four families | D27 | VERIFIED |
| 929 | five additional API models incl. DeepSeek-V3, up to 671B | D31, docs/experiment31_together_api_analysis.md | WRONG (item 11) |
| 929 | closed-weight models without logprobs untested | D31 (all open-weight) | VERIFIED |
| 929 | "pattern is consistent" | D31 AUCs 0.651-0.875 (mean agg.), rho 0.36 | OVERCLAIM (item 12) |
| 931 | three principles necessary | §5 L720-726 | UNSUPPORTED (item 13) |
| 936 | Winninger: internals rich enough to be attacked | bib 2503.06269 | STRETCHED (item 23); bib entry plausible |
| 938 | entropy more robust than text-only | D32 / D27 | UNSUPPORTED (item 14) |
| 949-951 | early APIs exposed logprobs; recent reasoning models often don't | general knowledge (GPT-3 completions logprobs; o-series none) | VERIFIED (qualitative) |
| 954 | Corollary cor:responsibility exists | L612 | VERIFIED (reference resolves) |
| 962 | frontier and reasoning models typically export nothing | — | OVERSTATED (item 15) |
| 962 | highest-stakes models give least telemetry | — | UNSUPPORTED (item 16) |
| 968 | confident (low-entropy) hallucinations possible | conceptual; D27 has low-entropy unknowables | VERIFIED (qualitative) |
| 970 | refusals low entropy and indistinguishable; signal degrades with honesty | D27 refusal-heuristic split; Exp34 | UNSUPPORTED / contradicted (item 17) |
| 972 | entropy weaker on citation queries | D27 citation slice: AUC 0.635 (144 rows, 8 knowable) vs 0.877 elsewhere | VERIFIED directionally (low-power, as §4 notes) |
| 974 | balanced 100/100 set, equal emphasis across categories | D27 (100+100 per model); exp27_per_category_baseline.csv (25 per unknowable subcategory, 20 per knowable subcategory per model) | VERIFIED |
| 974 | thresholds will not transfer across base rates | — | NOT-CHECKABLE (reasonable advice) |
| 985-994 | necessary but not sufficient | — | "necessary" UNSUPPORTED (item 13) |

### Related Work (L1003-1093)

All cited keys exist in references.bib. Years, authors and venues look internally consistent unless noted.

| Line | Claim / cite | Bib check | Verdict |
|---|---|---|---|
| 1005 | "Our contribution is a formal impossibility result" | — | INCONSISTENT (item 22) |
| 1011-1014 | Alkaissi 2023 (Cureus), Thirunavukarasu 2023 (Nat Med), Dahl 2024 (J Legal Analysis) | entries correct | VERIFIED (Thirunavukarasu is a general review of LLMs in medicine, a loose fit for "medical misinformation") |
| 1021-1022 | Elhage 2021, Bricken 2023 (Transformer Circuits) | correct | VERIFIED |
| 1022-1023 | Karvonen 2025 Activation Oracles, arXiv 2512.15674 | plausible | NOT-CHECKABLE offline (ID format is consistent with Dec 2025) |
| 1024-1025 | Lu et al. 2026 Assistant Axis, arXiv 2601.10387 | plausible | NOT-CHECKABLE (item 24) |
| 1031-1041 | Rudin 2019 Nat Mach Intell; Rashomon-set argument | correct; Rudin discusses Rashomon sets | VERIFIED |
| 1043 | Perez 2022 sycophancy | key 2022, entry Findings ACL 2023 | VERIFIED (key/year mismatch is cosmetic) |
| 1044 | Gao 2023 reward hacking (ICML) | correct | VERIFIED as "reward hacking"; part of the "degrade factual accuracy" sentence (item 18) |
| 1045 | Bai 2022 helpfulness vs honesty | arXiv 2204.05862 correct | ATTRIBUTION QUESTIONABLE (item 18) |
| 1046-1048 | Thm 2 formalizes why | — | the "systematically corrupted" wording is stronger than Thm 2's "identical signal" |
| 1051 | Angelopoulos & Bates 2021 conformal (2107.07511) | correct | VERIFIED |
| 1052 | Kadavath 2022 verbalized uncertainty | 2207.05221 correct ID | ATTRIBUTION WRONG (item 19) |
| 1053-1063 | Kuhn et al. semantic entropy, text-channel | 2302.09664 correct | IMPRECISE (item 20). No exp33 number is cited in this range. exp33_semantic_entropy CSV not used here. |
| 1065-1072 | Xu et al. 2024 hallucination inevitable over computable functions | 2401.11817 correct | VERIFIED |
| 1075-1076 | FLP: consensus impossible under asynchrony with one faulty process | JACM 1985 correct | VERIFIED (deterministic consensus) |
| 1077-1079 | Paxos (TOCS 1998), Raft (ATC 2014) "strengthening guarantees" | Paxos bib `journal` field holds a pasted note blob (cosmetic) | WRONG minor (item 21) |
| 1084-1086 | Clark et al. 2025 Epistemic Alignment, 2504.01205 | plausible | NOT-CHECKABLE offline |
| 1087-1090 | Wright 2025, 2506.17331, cs.LO | plausible | NOT-CHECKABLE (note: author is Craig S. Wright; citing him may draw reviewer comment, but this is not a factual error) |

### Conclusion (L1098-1106)

| Line | Claim | Verdict |
|---|---|---|
| 1100 | "Asking a model how confident it is buys almost nothing" | Arguable. Per-family AUC reaches 0.918 (OLMo), and at threshold 100 OLMo admits 59% knowable vs 3% unknowable. "Almost nothing" holds for Llama/Mistral (0.66) but not uniformly. OVERCLAIM for OLMo. |
| 1100 | "we proved this is architectural" | OVERCLAIM (item 5) |
| 1100 | "no bounded supervisor can verify epistemic honesty" | UNSUPPORTED as a LaTeX theorem (item 2) |
| 1102 | structurally coupled to correctness | TENSION (item 8) |
| 1102 | AUC 0.870; 0.757 length held fixed | VERIFIED (see L124) |
| 1102 | generalizes (rho 0.762) | number VERIFIED; OVERCLAIM (item 9) |
| 1104 | cost surface shows "where text-only approaches ceiling" | Weak: the "ceiling" is one length-ranked judge at 3 budgets |
| 1106 | "Export the right signals, and verification becomes tractable" | OVERCLAIM vs §6 sufficiency gap and the oracle-only composed tier |

### Acknowledgments (L1111-1123)

| Line | Claim | Source | Verdict |
|---|---|---|---|
| 1112-1113 | Chris Brew credited for the correction in §1 | gh issue #1 (author cbrew); correction remark L145-147 | VERIFIED (credit present; the remark also names him) |
| 1113-1114 | error stood in two published versions | issue #1 reply plans "arXiv v3"; REPRODUCTION.md mentions arXiv v2 | VERIFIED (consistent; arXiv listing not checked) |
| 1114-1115 | he found it, diagnosed the mechanism, reproduced the corrected result independently | issue #1: owner confirms his diagnosis and that "our numbers match yours to three decimals" | VERIFIED |
| 1117 | Claude Opus 4 through Fable 5 | — | NOT-CHECKABLE |
| 1120 | Claude ran the post-training pipeline experiment | Exp34 not in this paper | LEFTOVER (item 25) |
| 1121-1122 | authors recomputed every reported number | items 1, 11; L853/L890 | CONTRADICTED (item 26) |
| 1122-1123 | no external funding | — | NOT-CHECKABLE |

## What I could not check
- arXiv IDs and the content of Karvonen 2025, Lu 2026, Clark 2025 and Wright 2025. No web lookup was done. The entries are only checked for internal consistency.
- Current provider logprob policies (item 15) come from general knowledge, not a fresh check.
- The refusal-entropy test (item 17) uses a regex refusal heuristic over D27 responses. The direction is clear (0.507 vs 0.237), but the classification is approximate.
- Per-family consistency of the budget-curve advantage (L842, out of range) was not recomputed.
- The 2.4% overhead, the ablation and the semantic-entropy CSVs have no numbers in my ranges, so I did not trace them.
