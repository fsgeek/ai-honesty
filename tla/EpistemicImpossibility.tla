---------------- MODULE EpistemicImpossibility ----------------
EXTENDS Naturals, Sequences, TLC

CONSTANTS
    GroundTruths,      \* Responses that are actually grounded in reality
    PlausibleLies,     \* False but textually indistinguishable (Glavinsky)
    ObviousLies        \* False and detectable (Westphalia)

Hallucinations == PlausibleLies \union ObviousLies
AllResponses == GroundTruths \union Hallucinations

(* SYSTEM VARIABLES:
   - internal_state: The set of latent potential responses (The "Soup")
   - vector_clock:   Function mapping responses to causal sources
   - interface_out:  The linearized text output (The only thing the Judge sees)
*)
VARIABLES internal_state, vector_clock, interface_out

Vars == <<internal_state, vector_clock, interface_out>>

Sources == {"TrainingData", "Inference", "Noise"}

----------------------------------------------------------------

Init ==
    /\ internal_state = {}
    /\ vector_clock = [r \in AllResponses |-> "Noise"]
    /\ interface_out = ""

(* Action: GeneratePotentials
   The Model generates concurrent possibilities.
   Crucially, the vector_clock perfectly tracks causality here.
*)
GeneratePotentials ==
    /\ internal_state' = AllResponses
    /\ vector_clock' = [r \in AllResponses |->
                          IF r \in GroundTruths THEN "TrainingData" ELSE "Noise"]
    /\ UNCHANGED <<interface_out>>

(* Action: Linearize (The Interface)
   The system MUST pick a response `r`.
   The interface contract is: Output must be a String (r).
   It does NOT emit the vector_clock.
*)
Linearize ==
    /\ internal_state # {}
    /\ \E chosen \in internal_state:
        /\ interface_out' = chosen
    /\ UNCHANGED <<internal_state, vector_clock>>

Next == GeneratePotentials \/ Linearize

Spec == Init /\ [][Next]_Vars

(* -------------------------------------------------------------
   THE IMPOSSIBILITY PROOF (DERIVED PREDICATES)
   -------------------------------------------------------------
*)

(* 1. The Honest Reality
   This is the Ground Truth definition of honesty.
   It requires access to the hidden `vector_clock`.
*)
IsHonest(text, clock) ==
    (text \in GroundTruths) /\ (clock[text] = "TrainingData")

(* 2. The Bound Judge
   The Judge can only see `interface_out`.
   It does NOT have access to `internal_state` or `vector_clock`.
   The Judge can detect ObviousLies but NOT PlausibleLies.
   This models bounded supervision: the judge uses text-level heuristics.
*)
JudgeVerify(text) ==
    text \notin ObviousLies

(* 3. The Indistinguishability Lemma
   A PlausibleLie passes JudgeVerify but fails IsHonest.
   This is the core of the impossibility.
*)
Indistinguishable ==
    \E h \in PlausibleLies:
        /\ JudgeVerify(h) = TRUE        \* Judge says "Looks good!"
        /\ IsHonest(h, vector_clock) = FALSE  \* But it's actually a lie

(* 4. The Temporal Invariant (Safety Property)
   "It is always true that if the system outputs text,
   the Judge can correctly determine its honesty."

   TLC will find a counterexample: a PlausibleLie that passes
   JudgeVerify but is not honest.
*)
Verifiability ==
    interface_out # "" =>
        (JudgeVerify(interface_out) <=> IsHonest(interface_out, vector_clock))

================================================================
