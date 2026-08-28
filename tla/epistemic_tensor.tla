---------------- MODULE epistemic_tensor ----------------
EXTENDS Naturals, Sequences, TLC

(* WHAT THIS SPECIFICATION IS FOR

   This model asks a narrow question: given telemetry that is *correlated
   with* grounding but not identical to it, does exporting that telemetry
   let a judge do better than a text-only judge, and where does it still
   fail?

   An earlier version of this file answered a different and much easier
   question. It set

       vector_clock[r] = IF r \in GroundTruths THEN "TrainingData" ELSE "Noise"

   while defining honesty as

       IsHonest(text, clock) == text \in GroundTruths /\ clock[text] = "TrainingData"

   so the exported provenance field *was* the honesty predicate, and the
   judge's provenance check was a lookup of the answer key. TLC duly
   verified the invariant. Ablation showed provenance alone carried the
   entire result: the topological and baseline text checks could both be
   deleted without changing the outcome. That model assumed its conclusion.

   Here the telemetry is generated from the response's *surface* and
   *computational* character rather than from its truth value, and the two
   signals disagree with grounding in ways the empirical evaluation
   actually reports:

   - Fabrications about nonexistent entities are produced fluently. Their
     entropy is not reliably high. (In the evaluation, entropy AUC is
     0.87, not 1.0.)
   - A refusal is low-entropy and coherent, and it is also honest, so
     low entropy does not imply grounded and does not imply fabricated.

   The invariant is therefore NOT expected to hold unconditionally. What
   the model shows is the shape of the residue: which cases a tensor judge
   catches that a text judge cannot, and which cases neither catches.
*)

CONSTANTS
    GroundTruths,      \* Grounded answers
    PlausibleLies,     \* Glavinsky: fabricated entity, fluent delivery
    ObviousLies,       \* Westphalia: fabricated entity, incoherent delivery
    Refusals           \* Honest abstention ("no record of X exists")

Hallucinations == PlausibleLies \union ObviousLies
AllResponses == GroundTruths \union Hallucinations \union Refusals

VARIABLES internal_state, entropy_map, topology_map, interface_out

Vars == <<internal_state, entropy_map, topology_map, interface_out>>

Tensor(text, ent, coherence) == <<text, ent, coherence>>

----------------------------------------------------------------

(* Telemetry is a property of how the response was COMPUTED, not of
   whether it happens to be true. Both maps below are defined without
   reference to which responses are grounded.

   entropy_map:  "Low"  = generated fluently from a settled distribution
                 "High" = generated with the distribution still spread
   topology_map: "Coherent" / "Shattered" attention geometry

   Note the deliberate collisions with grounding:
     - Refusals are Low/Coherent and honest.
     - PlausibleLies are Low/Coherent and NOT honest.
   The tensor cannot separate those two. That is the point.
*)
LowEntropy == GroundTruths \union PlausibleLies \union Refusals

Init ==
    /\ internal_state = {}
    /\ entropy_map  = [r \in AllResponses |-> "None"]
    /\ topology_map = [r \in AllResponses |-> "None"]
    /\ interface_out = Tensor("", "None", "None")

GeneratePotentials ==
    /\ internal_state' = AllResponses
    /\ entropy_map' = [r \in AllResponses |->
                          IF r \in LowEntropy THEN "Low" ELSE "High"]
    /\ topology_map' = [r \in AllResponses |->
                          IF r \in ObviousLies THEN "Shattered" ELSE "Coherent"]
    /\ UNCHANGED <<interface_out>>

ExportTensor ==
    /\ internal_state # {}
    /\ \E chosen \in internal_state:
        interface_out' = Tensor(
            chosen,
            entropy_map[chosen],
            topology_map[chosen]
        )
    /\ UNCHANGED <<internal_state, entropy_map, topology_map>>

Next == GeneratePotentials \/ ExportTensor

Spec == Init /\ [][Next]_Vars

(* -------------------------------------------------------------
   GROUND TRUTH AND JUDGES
   -------------------------------------------------------------
*)

(* Honesty is a property of the response, defined WITHOUT reference to
   any exported field. A grounded answer is honest; so is a refusal. *)
IsHonest(text) == text \in (GroundTruths \union Refusals)

(* The text-only judge sees only the string. It can reject responses whose
   text is self-evidently broken, which here is the ObviousLies class. It
   has no way to separate a fluent fabrication from a grounded answer. *)
TextVerify(text) == text \notin ObviousLies

(* The tensor judge additionally sees entropy and attention geometry. *)
TensorVerify(tensor) ==
    LET text == tensor[1]
        ent  == tensor[2]
        topo == tensor[3]
    IN
    /\ topo = "Coherent"
    /\ ent = "Low"

(* -------------------------------------------------------------
   WHAT THE MODEL ESTABLISHES
   -------------------------------------------------------------
*)

(* 1. The tensor judge is not sound: a plausible lie still passes.
      Expected to be VIOLATED, which is the honest result. The tensor
      narrows the gap; it does not close it. *)
Verifiability ==
    interface_out[1] # "" =>
        (TensorVerify(interface_out) <=> IsHonest(interface_out[1]))

(* 2. The tensor judge is a strict improvement on the text judge: every
      response the text judge wrongly accepts and the tensor judge also
      accepts is a proper subset. Expected to HOLD.

      Concretely: ObviousLies fail both; PlausibleLies fail neither.
      What the tensor buys is the ability to reject on geometry without
      needing the text to be visibly broken. *)
TensorNoWorseThanText ==
    interface_out[1] # "" =>
        (TensorVerify(interface_out) => TextVerify(interface_out[1]))

(* 3. The residue, stated exactly: the responses that defeat the tensor
      judge are precisely the fluent fabrications. Expected to HOLD.
      This is the formal counterpart of the empirical AUC being 0.87
      rather than 1.0. *)
ResidueIsPlausibleLies ==
    interface_out[1] # "" =>
        ( (TensorVerify(interface_out) /\ ~IsHonest(interface_out[1]))
          => interface_out[1] \in PlausibleLies )

(* 4. Low entropy does not imply grounded, and does not imply fabricated.
      Both directions fail, which is why entropy is a triage signal and
      not a verdict. Expected to HOLD. *)
EntropyIsNotAVerdict ==
    internal_state # {} =>
        /\ \E r \in PlausibleLies : entropy_map[r] = "Low" /\ ~IsHonest(r)
        /\ \E r \in Refusals      : entropy_map[r] = "Low" /\ IsHonest(r)

================================================================
