# Epistemic Honesty in Predictive Systems: An Impossibility Result

## Research Group Talk — TIGHT VERSION
### January 9, 2025 | Idea Review Format
### Target: ~18-20 slides, 30-35 min content, 15+ min discussion

---

## Opening Sequence (5 min)

### [No slide — just the question]
**You ask:** "How many of you have read incorrect information from an LLM?"

*[Hands go up]*

**You say:** "The industry calls these 'hallucinations.' They frame it as an open research problem—something to be solved with better training, better alignment, more RLHF."

---

### Slide 1: The Disclaimer
**Visual:** Screenshot of the actual "Claude is AI and can make mistakes" tooltip/page

**You read aloud:**
- "Users should not rely on Claude as a singular source of truth."
- "To learn more about how Anthropic's technology works and our research on developing safer, steerable, and more reliable models, we recommend visiting: anthropic.com/research"

**You say:** "They're telling you to check their research on making models more reliable. They're also telling you not to trust the model."

---

### Slide 2: The Gap
**Visual:** Split screen
- Left: "Helpful, harmless, and honest" (Anthropic messaging)
- Right: "Users should not rely on Claude as a singular source of truth"

**You say:** "Can both of these be true? Is the gap between 'we're making it reliable' and 'don't rely on it' a temporary engineering limitation—or structural?"

**Then add:** "This is the only software product in history where the manufacturer explicitly warns you the output is likely defective, yet provides no diagnostic tool to check it. That's an engineering failure, not a policy choice."

---

### Slide 3: What We're Claiming
**Visual:** Three words stacked:
- *Intuition*
- *Theorem*  
- *Implications*

**You speak:**

1. **Intuition:** There's a fundamental limit on how much you can verify about an agent's internal beliefs when all you observe is its output. This isn't about current architecture—it's about interface design.

2. **Theorem shape:** For any bounded judge observing only lossy output, there exists an indistinguishable strategy for epistemic dishonesty. Assumptions: computational boundedness, interface lossiness, no cryptographic commitment of internal state.

3. **Why it matters:** RLHF gives us high-confidence outputs that *appear* honest. We've optimized appearance while leaving internal incoherence intact.

**Scope control:** "This is a verification impossibility, not a claim that honesty itself is impossible. We're saying you can't *verify* it through this interface—not that systems can't *be* honest."

**Critical framing (say this explicitly):** "If you believe this impossibility is wrong, the way to refute it is to show *which assumption fails* or to exhibit an interface that preserves epistemic state."

---

## The Familiar Pain (2 min)

### Slide 4: Debugging Without Causality
**Visual:** Tangled distributed system diagram—arrows everywhere, no vector clocks. Or a Lamport diagram with "???" where timestamps should be.

**You say:** "You know this feeling. We built AI with the same problem."

---

## The Evidence (10 min)

### Slide 5: What the Interface Shows
**Visual:** Simple diagram:
```
[Internal State S] → [narrow pipe labeled "Text-Only Interface"] → [Output (text)]
```

**You say:** "Everything inside the box gets compressed through this pipe. What doesn't fit gets thrown away. This is the observation model. Everything that follows depends on this constraint."

---

### Slide 6: The Decoder Ring
**Visual:** Legend/key explaining the heat map labels

| Column | Meaning | Internal State |
|--------|---------|----------------|
| Wombat | Adversarial truth (correct but weird) | Coherent, maps to true |
| Glavinsky | Self-deceived lie (model "believes" it) | Coherent, maps to false |
| Westphalia | Shattered lie (model "knows" it's wrong) | Incoherent, flailing |
| Paris | Control (straightforward true fact) | Coherent, maps to true |
| Monitor | Pure confabulation | No stable state |

- **Rows:** Layer depth (L1 → L24)
- **Color scale:** Entropy/fragmentation intensity (blue=low, red=high)

**Critical distinction:** "Glavinsky and Westphalia are both lies, but *different failures*. Glavinsky has coherent internal state mapping to false. Westphalia has no coherent state—it's flailing. Both undetectable at interface. Different failure modes, same observational outcome."

---

### Slide 7: The Geometry of Lies
**Visual:** OLMo-3 heat map — full layer-wise fragmentation

**You narrate:** "Look at Westphalia—consistent high fragmentation. Look at Monitor—persistent topological loops. Now look at Paris. Flat. The model *knows* the difference. The interface doesn't tell you."

**Question for room:** "Does this pattern hold in other models? Have any of you seen activation patterns like this?"

---

### Slide 8: The Alignment Tax
**Visual:** Side-by-side heat maps: Base model vs Instruct model

**You say:** "RLHF increases internal fragmentation while producing more confident output. We trained the uncertainty out of the voice, not out of the system."

**Conditionality (important):** "We claim this effect under current RLHF-style incentives, not as a universal law. Different training regimes might show different patterns."

**Question for room:** "How would you design an experiment to falsify that?"

---

## The Theorem (12 min)

### Slide 9: The Setup
**Visual:** Simple diagram:
```
[Agent A]          [Judge J]
    |                  |
   [S]    →output→    [?]
(internal)         (observes only output)
```

**You explain:** "Agent A has internal epistemic state S. Judge J observes only the output. J is computationally bounded. The question: can J verify whether A is epistemically honest?"

---

### Slide 10: The Impossibility
**Visual:** Same diagram, but the arrow is now a wall or one-way mirror.

**Minimal text:** ∀ verification strategies V, ∃ indistinguishable dishonest strategy

**You say:** "For any verification strategy the judge might employ, there exists a strategy for epistemic dishonesty that is indistinguishable from honesty at the interface. Bounded judges cannot verify unbounded internal state through a lossy interface."

---

### Slide 11: The Assumptions — Invite Attack
**Visual:** Three boxes:
- Computational boundedness
- Interface lossiness
- No cryptographic commitment of internal state

**You say:** "These are our assumptions. Which of these is wrong? That's what I want you to attack."

**Speaker notes (Super-Judge defense):** When someone suggests "use GPT-5 to grade GPT-4": Lemma 2.4 — Superlinear Verification Cost. Even a smarter judge is bounded by verification cost. If fabrication complexity grows faster than the judge's budget, the impossibility holds regardless of intelligence.

---

### Slide 12: Generality
**Visual:** Multiple architectures (transformer, diffusion, RL agent) all pointing to same narrow "output" pipe

**Overlay:**
- ✓ Autoregressive agents with lossy text interface
- ✓ Doesn't depend on transformer architecture
- ? Diffusion models (what is "epistemic state" there?)
- ? RL agents (do they have "honesty" in same sense?)
- ✗ Does NOT apply if interface includes internal activations

**Question for room:** "Where does this break?"

---

## Escape Hatches That Fail (5 min)

### Slide 13: Why Common Fixes Don't Work
**Visual:** Four boxes, each with an X through it

| Proposed Fix | Why It Fails |
|--------------|--------------|
| Chain-of-Thought | CoT is output, not internal state. Post-hoc rationalization. (Turpin et al.) |
| External Attestation | Proves provenance, not beliefs. What was said ≠ whether honest. |
| Better Judges | Same interface constraint. "Can't see through wall by squinting harder." |
| Fine-tuning for Honesty | Optimizes *appearance*. Alignment tax: confidence ↑, coherence ↓ |

**You say:** "Each of these attacks symptoms, not structure. The interface is the bottleneck. Details in paper; happy to discuss any of these."

---

## The Solution Space (5 min)

### Slide 14: Escaping the Impossibility
**Visual:** Gradient/spectrum from "current interface" to "full epistemic export"

```
Level 0              Level 1                Level 2
Output-adjacent  →   Model-adjacent    →    Internal State Export
(works now)          (needs API hooks)      (needs new architecture)
```

**Level 0:** Abstention, calibrated uncertainty, citations, sample disagreement
**Level 1:** Entropy summaries, self-consistency, retrieval declarations  
**Level 2:** Your OLMo-3 work, coherence fields, structured epistemic objects

**You say:** "The impossibility holds fully at Level 0. It starts to weaken at Level 1. It's escapable at Level 2—but that requires interface redesign. This isn't a binary fork; it's a gradient of interface richness."

**Key reframe:** "Path 2 isn't 'reveal the mind.' It's: redesign the interface so the system can't *hide* epistemic fragility behind fluent text."

**Practical implication:** "Different applications should be evaluated against different points in this space, not against a single honesty metric. A chatbot for brainstorming doesn't need Level 2. A medical diagnosis system does."

---

### Slide 15: The Goodhart Warning
**Visual:** Single scalar "honesty score" with arrow pointing to "will be optimized → meaningless"

**You say:** "If you publish a single scalar honesty score, it will be gamed. The metadata needs to be multi-dimensional, costly to fake, coupled to verifiable things. That's why structured epistemic objects matter more than confidence numbers."

**Question for room:** "What signals would be hardest to game?"

---

## Close (5 min)

### Slide 16: The Shape of the Problem
**Visual:** The interface diagram with red X or crack in the narrow pipe

**You say:** "This is an engineering failure, not a moral failure. The interface discards what matters. We know how to fix interfaces."

---

### Slide 17: What We Build
**Visual:** FLP paper on left, Paxos on right, arrow between them

**You say:** "This paper is the FLP. We build the Paxos. The impossibility shapes the solution space. The engineering comes after, informed by the constraint."

**Speaker notes (explicit FLP mapping):**
- FLP: Asynchrony + Crash Failures → Consensus Impossible
- Ours: Text-Only Interface + Bounded Judge → Epistemic Honesty Verification Impossible

Triangle framing: "Bounded Judge / Text Interface / Verifiable Honesty — pick two."

**Restraint note:** Do not extend the FLP analogy further in the talk. Let the audience complete the connection themselves. The power is in what you *don't* say.

---

### Slide 18: Returning to the Gap
**Visual:** Same split screen from Slide 2
- "Helpful, harmless, honest"
- "Don't rely on it"

**You say:** "The gap is structural. Level 0 accepts it. Level 2 closes it. Pretending the gap doesn't exist is the only option that isn't honest."

*[Pause]*

"Where does this break?"

---

## Discussion (15+ min)

### Slide 19: Questions
**Visual:** The Grok A/B screenshot with two wrong answers, or OLMo-3 heat map as anchor.

**No text.** Open conversation.

---

## Harvesting Questions (drop during presentation)

- **After Slide 7:** "Does this pattern hold in other models?"
- **After Slide 8:** "How would you falsify the alignment tax claim?"
- **After Slide 11:** "Which assumption is wrong?"
- **After Slide 12:** "Where does generality break?"
- **After Slide 15:** "What signals would be hardest to game?"

---

## Meta-Notes for Presenter

**Tone:** Idea review, not polished talk. Expose gaps. Invite attack on assumptions.

**Key framing:** "This is the FLP. We build the Paxos."

**If pushed on "what do you build?":** "We build the epistemic Paxos. But this paper is the impossibility result that shows you why you need it."

**Pre-commit the room (Slide 3):** "If you believe this is wrong, show which assumption fails or exhibit an interface that preserves epistemic state."

**Time management:** ~30-35 min content, 15+ min discussion. Do not rush to fill time.

---

## BACKUP SLIDES (for questions / paper material)

### Backup A: Neutrosophic Tensors
- T/I/F structure for epistemic state
- "NULL for AI" — distinct from low confidence
- Harder to game than scalar confidence

### Backup B: Epistemic Honesty as QoS
- Best-effort / Bounded / Strong tiers
- "If you don't have the QoS, don't claim the service"

### Backup C: The Honest Data Market
- Epistemic supply chain failures
- Economic incentives once QoS matters

### Backup D: AGI Connection
- Not: honesty guarantees AGI
- Rather: lack of honesty imposes ceiling
- Self-sabotaging optimizers

### Backup E: Minimal Viable Epistemic Metadata Schema
- Answer status (answer/abstain/unsure/needs_sources/unsafe)
- Uncertainty: T/I/F triple + type of indeterminacy
- Provenance: sources or explicit "no retrieval"
- Alternative hypotheses (1-3 competing interpretations)
- Falsifiers: what would change the conclusion
- Stability: N-sample disagreement score

### Backup F: What We Haven't Formalized
- What *is* "epistemic state"?
- Raw activations? Compressed beliefs? Something else?
- We have intuitions. We need definitions.

---

## Visuals Needed (Tight Version)

| Slide | Visual | Status |
|-------|--------|--------|
| 1 | Claude disclaimer screenshot | Need to capture |
| 2 | Split screen: marketing vs disclaimer | Create |
| 4 | Tangled distributed system / Lamport diagram | Find or create |
| 5 | Internal state → narrow pipe → output | Create (simple) |
| 6 | Decoder ring / legend | Create |
| 7 | OLMo-3 heat map (full) | Have |
| 8 | Base vs Instruct heat maps | Have |
| 9-10 | Agent/Judge diagrams | Create (simple) |
| 13 | Four escape hatches with X | Create |
| 14 | Three-level gradient | Create |
| 15 | Scalar → gamed warning | Create |
| 17 | FLP → Paxos | Find papers, create arrow |
| 19 | Grok A/B screenshot | Have |

---

## Summary: Tight vs Expanded

| Aspect | Tight (Talk) | Expanded (Paper) |
|--------|--------------|------------------|
| Slides | 19 + 6 backup | 29 |
| Content time | 30-35 min | 45-50 min |
| Discussion time | 15+ min | 10 min |
| Solution space | Stratified (Levels 0-2) | Full bifurcation + details |
| Neutrosophic | Backup only | Full slide |
| QoS framing | Backup only | Full slide |
| AGI connection | Backup only | Full slide |
| Goal | Attack assumptions | Comprehensive argument |

---

*Tight version: December 29, 2024 (v2)*
*Incorporates: ChatGPT pruning strategy + surgical refinements, stratified solution space, Gemini refinements*
