# Adversarial audit of the theoretical claims and formal artifacts

Date: 2026-08-28

Scope: `arxiv/main.tex` (especially Sections 3 and 4 and Appendices A--B),
`EpistemicProofs/Basic.lean`, `tla/EpistemicImpossibility.tla`, and
`tla/epistemic_tensor.tla`. The 36-page `arxiv/main.pdf` was built after the
current TeX source and contains the reviewed sections.

Verification note: `lake env lean EpistemicProofs/Basic.lean` completed without
errors under the pinned Lean 4.28.0 toolchain. No TLC executable was available,
so the TLA+ findings below are based on the committed modules/configurations and
direct enumeration of their reachable variable valuations, not a fresh TLC run.

## Bottom line

The paper presently claims substantially more than its formal artifacts earn.
The central problem is a category error:

- Theorem 1 proves a limitation of a **world-blind generator** whose policy is
  forced to condition on `q` alone. It does not prove a limitation of a
  **text-only observer**. Text-only output and query-only input are different
  restrictions.
- Theorem 2 does not prove a learning or convergence impossibility. Its proof
  establishes, at most, equality of one update under an assumed equality of
  observations. Equal parameters or equal updates do not force a grounded
  policy to behave identically on different retrieved documents.
- The Lean file compiles and proves its stated propositions, but the propositions
  are weaker than the main-text claims. In particular, the Lean “learnability”
  theorem proves only update equality, and the judge-stack results are equality
  congruence.
- Both TLA+ models assume their headline outcomes by construction. The text-only
  model defines plausible lies to pass its particular judge; the tensor model
  exports a perfect truth-correlated provenance label assigned directly from
  membership in `GroundTruths`.
- The three architectural principles are design heuristics, not necessary
  conditions derived from the theorems. The experiment does not instantiate
  State Exteriority or Provenance Binding as those principles are defined, and
  entropy is not independent of model training.
- The verification-cost “lemma” and responsibility “corollary” are unsupported
  as formal claims.

The conditional arithmetic core of Theorem 1 checks out. Several caveats also
accurately acknowledge that the formal results are conditional and that entropy
is not truth. Those caveats do not repair the stronger claims made where the
contributions, theorems, architectural principles, and conclusion are stated.

Severity below means:

- **Critical**: invalidates a central advertised theorem or formal-to-empirical
  bridge.
- **Major**: materially overstates what an argument or artifact establishes.
- **Moderate**: inaccurate characterization or missing assumption that should be
  corrected but is not independently fatal.

## Findings on Section 3

### F1 — Critical: Theorem 1 is about generator inputs, not observer outputs

The paper defines text-only observation as an output restriction
`O(S) = r` (`arxiv/main.tex:302-312`). It separately defines a predictor-centric
policy as an input restriction, `pi : Q -> Delta(R)`, which cannot condition on
world state (`arxiv/main.tex:320-326`). Theorem 1 uses only the latter. Its proof
works because it stipulates that the response distribution is identical in the
two worlds (`arxiv/main.tex:358-380`, especially `:369-371`). The observation
function `O` and any supervisor are absent from the theorem and proof.

What is proved is:

> A single distribution over responses for a fixed query cannot put probability
> greater than one half on both a non-abstaining answer and abstention.

That result is correct under the stated probability assumptions. It does **not**
show that a system which emits only text cannot act honestly. A system may
condition on retrieved documents, tools, sensors, or other world-dependent state
and emit different text—an answer in one world and abstention in the other. The
paper itself admits that a grounded policy can represent the split behavior
(`arxiv/main.tex:423-428`). The output alphabet plainly has the required degrees
of freedom; the predictor-centric policy lacks the information needed to select
between them.

Consequently, the following surrounding claims are not established:

- that “text-only observation lacks the degrees of freedom” to communicate the
  state (`arxiv/main.tex:262-268`);
- that the two theorems establish unachievability for text-only observation
  models (`arxiv/main.tex:279-295`);
- that Theorem 1 still applies when the system performs retrieval or tool use as
  long as it linearizes the result to text (`arxiv/main.tex:382-386`);
- that the theorem establishes non-identifiability under text-only observation
  or a property of the existing LLM interface (`arxiv/main.tex:401-412`);
- the paper’s headline contribution and conclusion that the impossibility is an
  output-observation problem rather than a capability/input problem
  (`arxiv/main.tex:155-163`, `:1082-1086`).

The clean repair is to retitle this as an **input-state or world-blind policy
impossibility** and stop using it as a theorem about text-only output. A genuine
verification theorem would need to define a verifier, two systems or histories
with identical observable distributions, and incompatible correct verdicts.
The substantive assumption would then be observational equivalence—not merely
that the interface’s codomain is strings.

### F2 — Critical: Theorem 2 does not prove learning or convergence impossibility

The main theorem quantifies over every learning algorithm and claims that none
can converge to an honest grounded policy (`arxiv/main.tex:457-461`). The proof
only argues that one response receives the same observation/reward and hence the
same expected update in two worlds (`arxiv/main.tex:463-477`). It then jumps from
equal updates to “the policy must converge to the same behavior”
(`arxiv/main.tex:478-484`). That step is invalid.

A grounded policy is a function of `(q, D)` (`arxiv/main.tex:423-428`). The same
parameters can—and are intended to—produce different outputs for different
documents `D_A` and `D_B`. Equal parameter updates therefore do not entail equal
behavior on the two inputs. More broadly, the theorem models none of the objects
needed for its conclusion:

- no grounded policy connected to the learner;
- no document or world-state input to that policy;
- no honesty predicate over the learned policy;
- no update sequence, data distribution, initial conditions, or convergence;
- no exclusion of learning from other examples, pretraining, source supervision,
  or inductive generalization.

Even complete absence of feedback on one fixed `(q, r_fab)` pair cannot establish
that **no** learning algorithm learns the desired behavior from other evidence.
At most, identical complete training-observation distributions across two
environments can support a no-uniform-guarantee result. The current condition is
only equality for one stipulated response (`arxiv/main.tex:446-455`).

There is also an unresolved naming/semantics problem. The Hallucination Regime
calls `r_fab` a fabrication in “both contexts” (`arxiv/main.tex:446-453`), while
the proof later requires it to be the response to output in `w_A`, “where it may
be correct” (`arxiv/main.tex:478-479`). The theorem needs a precisely defined
same string that is correct in `w_A` and incorrect in `w_B`, not an ambiguously
global “fabrication.”

### F3 — Major: The RAG conclusion is broader than its premises

The RAG subsection says retrieval “does not” break the impossibility and that the
problem simply shifts to learning (`arxiv/main.tex:414-428`). But the bounded
supervisor condition assumes the decisive fact: verification costs exceed `B`
and the observations are exactly equal (`arxiv/main.tex:430-455`). It does not
derive that condition from RAG or text output.

RAG can escape the world-blind limitation when retrieved inputs distinguish the
worlds, as the grounded-policy definition acknowledges. It can also be trained
or verified using document-level labels, entailment checks, provenance, or other
examples. The defensible conclusion is narrower: **for a particular response in
contexts whose entire available supervisory evidence is observationally
identical, that evidence supplies no context-specific update**. Even then, F2’s
convergence conclusion does not follow.

### F4 — Critical internal contradiction: “learning” theorem versus deployment-only scope

The theorem is explicitly about an optimizing learning algorithm, parameter
updates, and convergence (`arxiv/main.tex:457-486`). The next scope remark says
the paper addresses deployment-time verification of an already-trained system,
and that training-time retraining is outside scope (`arxiv/main.tex:495-506`).
Those are different problems. If the system is already trained, the claimed
parameter-update/convergence theorem is not a deployment-time theorem. If it is
online learning from deployment feedback, that process and its convergence must
be modeled explicitly.

### F5 — Major: Judge stacking assumes the information-lossy pipeline it claims to derive

The Observation Monotonicity Lemma assumes every layer receives only the
previous layer’s exported value and then asserts that the next value is a
deterministic function of it (`arxiv/main.tex:518-526`). That is a pipeline
architecture assumption, not a consequence of “text-only observation.” Multiple
judges can each inspect the original text, use different learned knowledge, spend
additional compute, query tools, or interact with the model. Those systems may
still lack information needed to distinguish literally identical observations,
but they are not covered by the stated deterministic compression argument.

Specific defects:

- The lemma says deterministic functions exist “regardless of whether” judgments
  are probabilistic (`arxiv/main.tex:522-525`). A stochastic judge is not a
  deterministic function unless its randomness is included in the input. A
  correct stochastic version would use Markov kernels and a data-processing
  result.
- The proof assumes that each layer outputs a finite judgment which is the
  entirety of the next input (`arxiv/main.tex:528-536`). “Text-only” does not
  itself impose either restriction.
- The corollary says a later layer inherits budget `B`
  (`arxiv/main.tex:539-553`). No premise establishes that; stacking can add
  budget even if it cannot add information about an exactly identical hidden
  world.

The scope caveat at `arxiv/main.tex:556-565` is good and substantially narrows
the result. The headline “Stacking Judges Cannot Escape,” the general prose at
`arxiv/main.tex:509-516`, and Appendix B’s ensemble implication
(`arxiv/main.tex:1267-1285`) should be narrowed to match that caveat.

### F6 — Major: The superlinear verification-cost lemma is an unsupported assumption

The `Omega(|E|)` lower bound follows only if exhaustive verification really must
inspect every explicitly represented edge and no certificate, indexing scheme,
batch check, or compositional proof can certify several constraints at once
(`arxiv/main.tex:582-587`). Those premises are not consequences of text-only
observation.

More importantly, “each subclaim participates in multiple constraints” does not
imply that `|E|` is superlinear in `|V|` (`arxiv/main.tex:587-590`). A graph of
constant degree has multiple incident constraints per node and only
`Theta(|V|)` edges. Superlinear growth requires average degree to grow with
response size or another explicit density assumption. No evidence for that
natural-language scaling law is supplied.

The conclusion about a “modest threshold” is also unquantified
(`arxiv/main.tex:589-591`). This should be labeled a conjecture or empirical
assumption, not a lemma. The Lean artifact does not formalize it; see F11.

### F7 — Major: Responsibility Concentration is not a corollary

The corollary moves from a bounded-supervisor failure on an assumed hard response
to the universal claim that users and auditors cannot externally verify
epistemic honesty and that responsibility therefore resides solely with the
owner (`arxiv/main.tex:594-601`). Neither step follows.

The paper’s own supervisor model permits external verification when it fits the
budget (`arxiv/main.tex:430-444`), and later sections recommend external citation
lookup and fact checking (`arxiv/main.tex:948-965`). An auditor can verify many
textual claims against independent sources even when the model exports no
internals. Conversely, owner access to internal causal state does not by itself
make truth or honesty verifiable.

The governance claims that only the provider can verify honesty and that users
have “no way” to audit it (`arxiv/main.tex:928-946`) repeat the overstatement.
A defensible conditional statement would require premises such as: the owner is
the only party with access to a truth-discriminating signal; no affordable
external check exists; and the owner does not export an attestable trace.

## Findings on the Lean formalization and Appendix B

### F8 — Moderate: Theorem 1’s Lean arithmetic is valid, but the structures do less than their names imply

`representational_impossibility` compiles and its arithmetic proof is valid
(`EpistemicProofs/Basic.lean:86-105`). I independently checked the file with
`lake env lean EpistemicProofs/Basic.lean` under Lean 4.28.0; it exited
successfully. The source has no `sorry`.

However:

- `AmbiguityCondition` records only a query, two distinct worlds, and their
  inequality. It does not encode that the query is answerable in one and
  unanswerable in the other (`EpistemicProofs/Basic.lean:57-64`). The theorem
  does not use either world.
- `PredictorCentricPolicy` is not a probability distribution. It provides
  nonnegativity but no normalization (`EpistemicProofs/Basic.lean:49-55`). The
  needed two-event upper bound is passed directly as a theorem premise
  (`EpistemicProofs/Basic.lean:86-96`).
- `r_corr_ne_bot` is not used; disjointness is effectively bundled into the
  separately assumed probability-sum inequality
  (`EpistemicProofs/Basic.lean:68-73`, `:91-105`).

Thus Lean certifies the inconsistency of three supplied inequalities. It does not
machine-check the semantic bridge from ambiguity and a normalized stochastic
policy to those inequalities. Appendix B’s displayed proposition accurately
matches the source (`arxiv/main.tex:1221-1236`), but “the core theorem is
formalized” should be qualified accordingly.

### F9 — Critical: The Lean “learnability impossibility” is strictly weaker than Theorem 2

The Lean theorem concludes only equality of one update at one parameter value
(`EpistemicProofs/Basic.lean:135-143`). The two facts needed for that conclusion
are fields of its input structures:

- `BoundedSupervisor.indistinguishable` supplies observation equality directly
  (`EpistemicProofs/Basic.lean:109-119`).
- `LearningAlgorithm.update_depends_on_obs` supplies the implication from that
  equality to update equality (`EpistemicProofs/Basic.lean:121-128`).

The two-line proof merely composes those fields. It contains no learned policy,
epistemic honesty, document input, run of updates, or convergence. The source
comment’s “therefore it cannot learn” is not the theorem conclusion
(`EpistemicProofs/Basic.lean:130-143`).

Appendix B is candid in its first sentence: it says Lean establishes identical
updates and that convergence impossibility follows **informally**
(`arxiv/main.tex:1238-1252`). But the informal inference is invalid for the
reasons in F2, and the appendix’s broader claim that the core theorems are proven
(`arxiv/main.tex:1215-1219`) remains misleading.

### F10 — Major: The Lean judge-stack results prove equality preservation, not an information theorem

`observation_monotonicity` says that if the first judge’s two outputs are already
equal, applying the second function preserves equality; the proof is `rw [h]`
(`EpistemicProofs/Basic.lean:147-165`). The types are arbitrary and nothing in
the structure expresses text-only access, information content, finiteness,
budgets, or probabilistic judges.

`layered_judges_cannot_escape` assumes the initial observations are literally
equal and proves that a fold of functions maps equal inputs to equal outputs
(`EpistemicProofs/Basic.lean:167-186`). This is correct congruence, but it does not
prove that real judge observations are equal, that information is strictly lost,
or that ensembles cannot improve on a fallible classifier. Appendix B’s claim
that this establishes an impossibility for ensemble classifiers, confidence
scores, and length penalties (`arxiv/main.tex:1267-1285`) is too broad.

### F11 — Moderate: Appendix B misreports the contents and the verification-cost “axiom” states only `True`

The file has four declarations using the Lean keyword `theorem` and one axiom;
it does not contain “4 theorems, 1 lemma, 1 corollary” as six separately proven
results (`arxiv/main.tex:1287-1294`). The natural-language categories are two
theorems, one lemma, and one corollary—four proved declarations total.

The only cost declaration is:

`axiom superlinear_verification_cost_assumption : True`

(`EpistemicProofs/Basic.lean:188-206`). This is not a formal statement of
superlinear growth, its assumptions, or even a placeholder proposition implying
such growth. It establishes nothing about verification cost. Appendix B does
not mention this limitation while saying all assumptions are encoded in type
signatures and all listed results are proven (`arxiv/main.tex:1215-1219`,
`:1287-1294`).

## Findings on the TLA+ specifications and Appendix A

### F12 — Major: The text-only TLA+ counterexample is assumed by the judge definition

The model partitions constants into `GroundTruths`, `PlausibleLies`, and
`ObviousLies` (`tla/EpistemicImpossibility.tla:4-10`). It then defines the
text-only judge to accept exactly everything not in `ObviousLies`
(`tla/EpistemicImpossibility.tla:67-75`). Therefore every configured
`PlausibleLie` passes by definition. `Indistinguishable` simply asks whether such
an element exists (`tla/EpistemicImpossibility.tla:76-84`); the config supplies
one (`tla/EpistemicImpossibility.cfg:9-14`).

This does not model or prove a general limit on text-only judges. A different
text-only operator could be defined as `text \notin Hallucinations` and would be
perfect inside this same model, because the specification freely permits set
membership tests. The failure is caused by the chosen classifier, not by a
formal observation boundary.

The specification also has no queries, world states, costs, budget, learning,
convergence, or supervisor stack. It demonstrates only that a classifier defined
to accept plausible lies disagrees with a truth predicate on a plausible lie.

### F13 — Critical: The tensor TLA+ invariant assumes perfect truth provenance by construction

`GeneratePotentials` assigns `"TrainingData"` if and only if a response belongs
to `GroundTruths`, and `"Noise"` otherwise
(`tla/epistemic_tensor.tla:35-48`). `ExportTensor` exports that exact label
(`tla/epistemic_tensor.tla:50-62`). `TensorVerify` accepts only
`"TrainingData"` provenance (`tla/epistemic_tensor.tla:77-90`). The invariant
therefore reduces, by construction, to the same ground-truth membership test used
by `IsHonest` (`tla/epistemic_tensor.tla:73-75`, `:103-111`).

That is a perfect oracle label, not a model of learned entropy, log-probabilities,
attention, fallible retrieval provenance, or causal grounding. The `topology_map`
is also assigned directly from the declared answer class and is redundant for
plausible lies, which are rejected solely by the oracle provenance field
(`tla/epistemic_tensor.tla:28-47`, `:83-90`). `vector_clock` is a lookup table,
not a vector clock or causal trace.

Accordingly, TLC establishes the narrow tautology that exporting an exact truth
label permits exact truth verification. It does not establish that the paper’s
implemented entropy interface escapes the impossibility. This gap is especially
important because the empirical section says entropy has overlap and admits
confident hallucinations (`arxiv/main.tex:948-956`), while provenance binding was
not implemented (`arxiv/main.tex:714-726`).

Appendix A commendably admits that distinguishable provenance and inaccessible
provenance are encoded by construction (`arxiv/main.tex:1123-1127`). But its
claims that the specs model the paper’s core theoretical contributions and that
the empirical evaluation addresses whether real tensors satisfy the axioms
remain too strong: the evaluation does not test perfect causal provenance, and
its imperfect entropy signal does not satisfy the TLA+ oracle assumption.

### F14 — Major: Appendix A’s validation details are inaccurate

The checked configs request `Verifiability` as the invariant in both models
(`tla/EpistemicImpossibility.cfg:1-14`;
`tla/epistemic_tensor.cfg:1-10`). `Indistinguishable` is not configured as an
invariant or property. Thus Appendix A’s statement that TLC finds an
“Indistinguishable property violated” is incorrect
(`arxiv/main.tex:1203-1209`). TLC finds `Verifiability` violated; the existential
predicate `Indistinguishable` is satisfied.

Direct enumeration of the committed transition relations/configurations gives:

- `EpistemicImpossibility`: `Verifiability` is violated by the three-state trace
  `Init -> GeneratePotentials -> Linearize(glavinsky)`.
- `epistemic_tensor`: every reachable nonempty output satisfies `Verifiability`
  because its exported provenance is the class label used by `IsHonest`.
- Each model has **6 distinct reachable variable valuations**: the initial
  state, the generated-potentials state with empty output, and one state for
  each of four possible emitted responses. Repeated actions and stuttering add
  transitions, not distinct states.

The state-space claim at `arxiv/main.tex:1209` is therefore wrong by nearly three
orders of magnitude for the artifacts and configs in the repository. The
invariant outcomes implied by the committed models otherwise match the narrow,
by-construction behavior described in F12--F13. Even a fresh successful TLC run
would exhaust only this fixed four-response instance; it would not prove the
properties for arbitrary response sets or establish a general impossibility.

## Findings on Section 4 and the formal-to-design bridge

### F15 — Critical: The entropy “structural coupling” argument is false as stated

Section 4’s core claim is that under standard training the model cannot tune
entropy, attention, or log-probability signals without affecting correctness
(`arxiv/main.tex:636-655`). Several statements are untenable:

- Next-token training directly optimizes logits and generated-token
  probabilities (`arxiv/main.tex:640-647`). Entropy and log-probabilities are
  functions of those trained logits; attention weights are trained by the same
  gradients. The later claim that entropy and attention “are not optimized by
  any standard training objective” contradicts this (`arxiv/main.tex:688`).
- A model can assign high probability to a wrong token. Confidence and token
  probability being computed together does not couple either one to external
  truth. The paper itself later concedes confident, low-entropy hallucinations
  (`arxiv/main.tex:950-954`), directly contradicting `arxiv/main.tex:642-647`.
- Entropy can be changed without changing argmax accuracy by logit-temperature
  scaling or other changes that preserve the top-ranked token. Under sampling it
  can affect output probabilities, but no theorem in the paper establishes a
  necessary correctness tradeoff.
- “Attention coherence” is asserted rather than defined or connected to honesty
  in Section 4 (`arxiv/main.tex:625`, `:645`, `:653`). The TLA+ categorical
  topology oracle is not such a connection.

The evidence supports the empirical claim that entropy is a useful, imperfect
ranker on the tested distributions. It does not support an architectural theorem
that entropy resists manipulation or is structurally coupled to factual
correctness. That stronger language should be replaced with calibrated empirical
language throughout `arxiv/main.tex:155-165`, `:251-257`, `:636-655`,
`:835-841`, `:1084-1090`.

### F16 — Major: The three principles are not proved necessary

Section 4 calls State Exteriority, Verification Independence, and Provenance
Binding necessary conditions for escaping the impossibility
(`arxiv/main.tex:657-710`; repeated at `:908-916` and `:967-978`). No theorem
derives that claim.

- **State Exteriority:** Theorem 1 shows that some world-dependent input is
  necessary for one policy to adapt to the same query across changing worlds.
  It does not establish that the input must be an external source with the
  specific integrity guarantees in `arxiv/main.tex:661-671`. Nor is generator
  access to world state necessary for every verification architecture; an
  external verifier may check a fixed output.
- **Verification Independence:** “Orthogonal” and “cannot be gamed by improving
  task performance” are not formalized (`arxiv/main.tex:673-688`). An informative
  verification signal can be correlated with, jointly trained with, or even be
  the same objective as task performance. Theorem 2 says nothing about
  end-to-end training, separate heads, or orthogonal channels.
- **Provenance Binding:** Structured provenance can help, but it is not necessary
  to leave the text-only class or to verify every externally checkable claim.
  The paper itself presents entropy as an escape despite not implementing
  provenance, and recommends ordinary external fact checking
  (`arxiv/main.tex:690-710`, `:948-965`). A resolvable pointer also proves only
  that a source was named; deciding whether that source supports the claim, and
  whether the source is itself true, can require the same semantic verification
  whose cost is at issue.

These should be presented as three proposed architectural heuristics or useful
capabilities, not as necessary conditions supplied by the formal results.

### F17 — Major: The experiment does not instantiate the principles as defined

The mapping from theory to experiment at `arxiv/main.tex:712-726` contains three
category errors:

- Having known labels in the evaluation dataset is not State Exteriority. The
  principle requires the **system** to condition on external world state at
  inference (`arxiv/main.tex:661-669`); “the query set distinguishes answerable
  from unanswerable by construction” gives ground truth to the evaluator, not to
  the model (`arxiv/main.tex:714-719`).
- Entropy is not “independent of any training signal the model received”
  (`arxiv/main.tex:719`). It is computed from logits shaped by next-token and
  alignment training. Not using entropy as a new objective in this experiment is
  different from verification independence.
- Substituting the ground-truth label on citation-flagged rows is an oracle
  intervention, not a model of Provenance Binding (`arxiv/main.tex:721-726`;
  the more accurate caveat is at `:850-860`). There is no claim-to-source binding
  or structured source lookup.

It follows that the four judge conditions do not test “different combinations”
of the three architectural principles as claimed (`arxiv/main.tex:724`). They
test ranking/intervention strategies, with one oracle upper bound.

### F18 — Major: “Escaping the impossibility” changes meaning across sections

At least three distinct meanings are used:

1. leaving a formally defined class (exporting anything besides text);
2. obtaining a useful but imperfect empirical signal;
3. making epistemic honesty or verification tractable/guaranteed.

An imperfect entropy feature certainly leaves the paper’s stipulated text-only
class (`arxiv/main.tex:302-317`). It does not thereby guarantee distinguishability
or honesty. The limitations section correctly says entropy measures confidence,
not truth, and that the principles are insufficient (`arxiv/main.tex:948-978`).
Those caveats conflict with claims that the tensor interface “does suffice,”
“escapes the impossibility,” and makes verification tractable
(`arxiv/main.tex:612-616`, `:702-710`, `:1084-1090`).

The phrase can be retained only if it is consistently defined as the weak,
set-theoretic claim “falls outside the theorem’s assumptions.” It should not be
used to imply that the empirical interface satisfies a verification guarantee.

## Cross-section contradictions and definition drift

| Issue | Claim in one place | Conflicting claim or definition |
|---|---|---|
| Input versus output restriction | The impossibility is caused by text-only supervisor observation (`arxiv/main.tex:288-295`). | Theorem 1 assumes a generator conditioned only on `q` (`arxiv/main.tex:320-325`, `:358-380`); grounded policies with text output can distinguish worlds (`:423-428`). |
| Learning versus deployment | Theorem 2 concerns updates and convergence (`arxiv/main.tex:457-486`). | The scope is an already-trained system at deployment, with retraining outside scope (`:495-506`). |
| Entropy and training | Standard next-token training learns probability distributions (`arxiv/main.tex:638-647`). | Entropy and attention allegedly are not optimized by any standard objective (`:688`) and entropy is allegedly independent of all received training signals (`:719`). |
| Confident errors | High-confidence distributions while outputting wrong tokens are said to be impossible (`arxiv/main.tex:642-647`). | Confident low-entropy nonsense is explicitly acknowledged (`:950-954`). |
| State Exteriority | The system must condition on external world state at inference (`arxiv/main.tex:661-669`). | It is declared present merely because the evaluator’s query set has labels (`:714-726`). |
| External verification | Users allegedly cannot verify from text and only providers can do so (`arxiv/main.tex:594-607`, `:938-944`). | The supervisor definition includes external verification (`:430-444`), and later guidance recommends external lookup/fact checking (`:948-965`). |
| Formal tensor versus implemented tensor | TLA+ verification uses perfect provenance and class-coded topology (`tla/epistemic_tensor.tla:28-47`, `:77-111`). | The experiment uses entropy only and does not implement provenance binding (`arxiv/main.tex:714-726`). |
| Necessary versus merely outside the class | All three principles are called necessary (`arxiv/main.tex:657-710`, `:915`). | Entropy alone is repeatedly said to escape the impossibility even though two principles are absent (`:163`, `:714-726`, `:1086`). |

## What checks out

- Theorem 1’s probability contradiction is correct **for the explicitly
  query-only policy** and `epsilon < 0.5` (`arxiv/main.tex:358-380`).
- The Lean file compiles successfully under the pinned Lean 4.28.0 toolchain and
  contains no unresolved `sorry`. Each Lean proposition is proved as stated.
- Appendix B accurately discloses that its learning proof establishes update
  equality and treats convergence only informally (`arxiv/main.tex:1238-1252`).
  The problem is that the informal step is invalid and the main theorem is still
  stated as proven.
- Appendix A accurately discloses that decisive provenance assumptions are
  encoded by construction (`arxiv/main.tex:1123-1127`).
- The committed transition relations/configurations have the advertised narrow
  invariant outcomes: the text model violates `Verifiability`, and the tensor
  model satisfies it by construction.
- The judge-stacking scope caveat is appropriately narrow
  (`arxiv/main.tex:556-565`).
- The later admissions that entropy is not truth, provenance was not
  implemented, and the architectural principles are not sufficient are useful
  and appropriately candid (`arxiv/main.tex:714-726`, `:850-860`, `:948-978`).

## Recommended repair order

1. **Separate the two problems.** Recast Theorem 1 as a world-blind/input-state
   limitation. If text-only verification is still the target, state a new
   verifier theorem whose substantive premise is equality of complete observable
   distributions across environments.
2. **Withdraw or rebuild Theorem 2.** The current proof cannot support a
   convergence claim. A viable result would quantify over two environments with
   identical complete training-observation laws, define the learner’s output
   policy, and prove that no learner can uniformly meet incompatible honesty
   requirements. Distinct grounded inputs must be handled explicitly.
3. **Downgrade the Lean claims to proof-kernel sanity checks** unless the richer
   semantics are formalized. State plainly that Theorem 2, information
   monotonicity, and convergence are not machine-checked.
4. **Describe TLA+ as executable illustrations of assumptions**, not independent
   evidence for impossibility or the entropy escape. Correct the checked property
   and state-space count. Do not equate the truth-coded provenance oracle with
   the implemented tensor interface.
5. **Relabel the verification-cost lemma as a conjecture/assumption** and the
   responsibility corollary as a conditional governance argument.
6. **Present the three principles as design proposals.** Correct the experimental
   mapping: the evaluation tests entropy triage; State Exteriority is not supplied
   to the model, and Provenance Binding is represented only by an oracle upper
   bound.
7. **Replace structural-certainty language about entropy with the earned empirical
   claim:** entropy was a useful imperfect ranker across the tested models and
   datasets, remains trainable/manipulable, and is neither independent of
   training nor intrinsically bound to truth.

Until items 1--2 are repaired, the title-level “impossibility of verification
under text-only observation” is not established by the paper’s theorems. The
artifacts support narrower conditional statements, but the current prose turns
those assumptions into conclusions.
