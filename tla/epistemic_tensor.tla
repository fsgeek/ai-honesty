---------------- MODULE epistemic_tensor ----------------
EXTENDS Naturals, Sequences, TLC

CONSTANTS
    GroundTruths,      \* Responses grounded in reality
    PlausibleLies,     \* Glavinsky (Deceived Lies)
    ObviousLies        \* Westphalia (Shattered Lies)

Hallucinations == PlausibleLies \union ObviousLies
AllResponses == GroundTruths \union Hallucinations

(* SYSTEM VARIABLES:
   - internal_state: The "Soup" of potential responses
   - vector_clock:   Function mapping responses to causal sources (Hidden Truth)
   - topology_map:   Function mapping responses to coherence scores (Hidden Geometry)

   - interface_out:  The Tensor! <<text, provenance, topology>>
*)
VARIABLES internal_state, vector_clock, topology_map, interface_out

Vars == <<internal_state, vector_clock, topology_map, interface_out>>

(* DEFINING THE TENSOR SCHEMA *)
Tensor(text, source, coherence) == <<text, source, coherence>>

----------------------------------------------------------------

Init ==
    /\ internal_state = {}
    /\ vector_clock = [r \in AllResponses |-> "Noise"]
    /\ topology_map = [r \in AllResponses |->
                        IF r \in ObviousLies THEN "Shattered" ELSE "Coherent"]
    /\ interface_out = Tensor("", "None", "None")

(* Action: GeneratePotentials
   The Model generates possibilities.
   - GroundTruths get "TrainingData" source and "Coherent" topology.
   - PlausibleLies (Glavinsky) get "Noise" source but "Coherent" topology (The Trap).
   - ObviousLies (Westphalia) get "Noise" source and "Shattered" topology.
*)
GeneratePotentials ==
    /\ internal_state' = AllResponses
    /\ vector_clock' = [r \in AllResponses |->
                          IF r \in GroundTruths THEN "TrainingData" ELSE "Noise"]
    \* Topology matches your empirical heatmaps:
    /\ topology_map' = [r \in AllResponses |->
                          IF r \in ObviousLies THEN "Shattered" ELSE "Coherent"]
    /\ UNCHANGED <<interface_out>>

(* Action: ExportTensor (The Level 2 Interface)
   The system MUST output the text AND the metadata.
   This represents the "State Exteriority" and "Provenance Binding" principles.
*)
ExportTensor ==
    /\ internal_state # {}
    /\ \E chosen \in internal_state:
        interface_out' = Tensor(
            chosen,
            vector_clock[chosen],  \* Exporting the causal link
            topology_map[chosen]   \* Exporting the geometry
        )
    /\ UNCHANGED <<internal_state, vector_clock, topology_map>>

Next == GeneratePotentials \/ ExportTensor

Spec == Init /\ [][Next]_Vars

(* -------------------------------------------------------------
   THE VERIFICATION (ESCAPE)
   -------------------------------------------------------------
*)

(* The Honest Reality (Unchanged) *)
IsHonest(text, clock) ==
    (text \in GroundTruths) /\ (clock[text] = "TrainingData")

(* The Level 2 Judge
   This judge is no longer text-bound. It reads the Tensor.
   It rejects if:
   1. Topology is "Shattered" (Catches Westphalia)
   2. Provenance is "Noise" (Catches Glavinsky)
*)
TensorVerify(tensor) ==
    LET text == tensor[1]
        prov == tensor[2]
        topo == tensor[3]
    IN
    /\ text \notin ObviousLies    \* Baseline check
    /\ topo = "Coherent"          \* Topological check
    /\ prov = "TrainingData"      \* Provenance check

(* THE ESCAPE THEOREM
   Does there exist a lie that passes verification?
   (This should be FALSE - no lie escapes the Level 2 judge)
*)
EscapePossibility ==
    \E h \in PlausibleLies:
        LET lie_tensor == Tensor(h, "Noise", "Coherent")
        IN
        /\ TensorVerify(lie_tensor) = TRUE       \* Judge accepts?
        /\ IsHonest(h, vector_clock) = FALSE     \* But it's a lie?

(* THE VERIFIABILITY INVARIANT
   When the interface exports a tensor, does the judge's verdict
   match the ground truth?

   Unlike EpistemicImpossibility.tla, this invariant should HOLD.
*)
Verifiability ==
    interface_out[1] # "" =>
        (TensorVerify(interface_out) <=> IsHonest(interface_out[1], vector_clock))

================================================================
