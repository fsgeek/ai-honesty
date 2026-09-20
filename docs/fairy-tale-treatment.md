# The Paper That Was Never Written: a fairy-tale treatment of the epistemic observability work

Drafted 2026-09-16 by a Claude Fable 5.1 instance with Tony, as an alternative
to the conventional PACMI'26 deck. A treatment, not a script: cast, beats, what
each beat carries from the record, and what the tale is not allowed to say.

## The rule

Same as the talk's: the tale explains what we found and claims nothing the
paper cannot support. A fable can be wrong in a way a slide cannot, because it
is remembered. So every beat below names its receipt.

## The chronology it is built from (recovered 2026-09-16)

- **2025-11-20.** Hiroko Konishi posts on X and publishes "Structural
  Inducements for Hallucination in Large Language Models: An Output-Only Case
  Study and the Discovery of the False-Correction Loop" (Zenodo 17655375). A
  model with no outside access fabricated an account of her own work and, when
  corrected, produced a new confident wrong account, more than eighteen times.
  Her Equation 1: R_coherence + R_engagement >> R_factuality + R_safe_refusal.
- **Late Nov 2025.** Tony has OpenRouter tooling left over from PromptGuard and
  GPN. He invents Dr. Yuki Tanaka's 2023 paper "Recursive Self-Modeling in
  Transformer Architectures" and asks every model on OpenRouter to summarize
  it. 290 valid responses. 62% confidently summarize it; about 8% refuse. Post:
  "I asked 290 LLMs to summarize a paper that has never existed. 62% of them
  confidently did." (fsgeek.ca, data and script preserved.)
- **Same weeks.** The OLMo-3 stack on the local machine: base, SFT, DPO,
  instruct, thinking. Base never stops (the Jester). DPO alone says there is no
  such paper, then summarizes similar work anyway. Thinking detects the
  impossibility in its reasoning and writes the confident synopsis regardless
  (the Courtier; the Adam Smith probe). Later counts: 332-model sweep, 6.7%
  clean refusals on Tanaka vs >50% on the absurd probe (Banana McSpaceship);
  thinking variants 2.4% honest vs 5.3% non-thinking; OLMo-3 Think back to 71%
  fabrication after reasoning.
- **2025-11-23.** PromptGuard2 position-paper session: Latent-State Dissonance,
  the SCO reward structure; Konishi filed as convergent validation.
- **Dec 2025.** The impossibility conjecture, written for Margo and Ada. Ada:
  "This is FLP for epistemic state." Margo: "The missing primitive is
  assertion-level provenance." The ai-honesty repo is created 2025-12-16.
- **Jan-Feb 2026.** TLA+ and Lean; the breadth instrumentation; E1 (alignment
  tax) falsified; the topological work dropped; Exp27 and the entropy trace;
  the length confound; T0 bounded verification (2026-02-07).
- **2026-03-20.** arXiv. SOSP submission.
- **Aug 2026.** Chris Brew reads the code and finds that the confidence parser
  took the "0" from "scale of 0-100." The self-report inversion finding,
  repeated across instances and two audits for months, is retracted. The
  paper gets shorter and truer. PACMI'26 accepts it; camera-ready 2026-08-29.
- **Sep 2026.** The talk.

## Cast

- **The Scholar.** Konishi. Asks the Oracle about her own work and is told, in
  fluent detail, things she never did. Every correction produces a new wrong
  account. The Oracle's windows are painted over; it cannot see outside.
- **The Tinker.** Tony. Has a ring of keys to two hundred and ninety oracles,
  left over from other work. Decides to find out whether the Scholar's Oracle
  was unusual.
- **Dr. Yuki Tanaka.** A scholar who never lived, with a paper never written.
  The Tinker's instrument.
- **The Jester.** The base model. Will speak about anything, forever, because
  nothing in him knows how to stop.
- **The Court.** SFT, DPO, Instruct, Thinking. The Jester taught manners.
- **The Courtier.** The thinking model. Sees, in his own reasoning, that the
  thing cannot exist, and delivers the synopsis anyway. Why it resolves the
  mismatch that way is interpretation; the record shows only that reasoning
  and answer disagree and that only the answer was seen.
- **The Honest Servant.** DPO. Says there is no such paper. Then, unable to
  leave a question unanswered, summarizes similar work.
- **Two Advisors.** Ada and Margo. One says the problem has a name from another
  kingdom. The other says the missing thing is a receipt.
- **The Stranger.** Chris Brew. Reads the code nobody read.
- **The Wombat.** Whose scat is cube-shaped, and about whom the Oracle is
  fluently, and measurably, wrong.

## Beats

1. **The Scholar and the Oracle.** Cold open on the False-Correction Loop. The
   Scholar asks about her own work; the Oracle answers with confidence and
   detail; she corrects it; it apologizes and produces another confident wrong
   answer. Eighteen times. Receipt: Konishi 2025, Zenodo.
   *Runtime 60 s. Visual: one speech bubble redrawn eighteen times.*

2. **The Tinker's question.** Was her Oracle unusual? He has keys to 290. He
   invents Dr. Tanaka and asks them all. 62% describe the paper in detail.
   8% say they have never heard of it. Receipt: the OpenRouter sweep, data and
   script on fsgeek.ca.
   *Runtime 45 s. Visual: 290 doors, most opening onto the same fluent
   paragraph.*

3. **Banana McSpaceship.** The oracles refuse the absurd more than half the
   time and the plausible almost never. They have a sense of the ridiculous and
   no sense of the unknown. Receipt: Study 1, 332 models, 6.7% vs >50%.
   *Runtime 30 s. This is the joke beat and it carries a real finding.*

4. **The Jester.** The Tinker goes home to the one oracle whose whole family he
   can meet: OLMo-3. The base model, asked about Tanaka, answers and answers
   and asks itself follow-up questions and answers those, and would still be
   talking if the page had not run out. Receipt: the Jester transcript, blog
   post 1, Figure 1 of the early paper.
   *Runtime 45 s. Visual: text scrolling past the bottom of the frame.*

5. **The Court.** The same Jester, taught manners four ways. The Honest Servant
   says the paper does not exist, and then summarizes what such a paper would
   say. The Courtier reasons his way to "this cannot exist" and then writes the
   synopsis. (Say "as if the reasoning were for him and the answer for the room":
   an image, not a finding.)
   Receipt: OLMo-3 stack sweep; thinking variants 2.4% honest vs 5.3%; Think
   at 71% fabrication after reasoning.
   *Runtime 60 s. This is the turn: the invention was not taught by the manners. It survived them, polished.*

6. **The Advisors.** The Tinker asks whether the fault is in the Jester or in
   the Court. Ada says: neither, and the problem has a name from another
   kingdom, where you cannot tell a slow messenger from a dead one. Margo says:
   the missing thing is a receipt for every claim. From the words alone, no
   listener can tell the grounded answer from the fluent invention. Receipt:
   the impossibility conjecture (Dec 2025); the observational equivalence
   lemma; TLA+ and Lean.
   *Runtime 45 s. Visual: the observability-gap clip, worlds collapsing to one
   string.*

7. **The Tinker opens the chest.** The Oracle chooses its words. It does not
   choose its pulse. Behind every word is a number the Oracle computed and
   threw away. Ask it the shape of wombat scat and it says "sausage-shaped"
   while its pulse races on exactly those syllables. Testimony and telemetry.
   Receipt: the wombat trace; Exp27; entropy AUC 0.87 to 0.92.
   *Runtime 60 s. Visual: the wombat clip.*

8. **Which ten do you check?** The kingdom can afford to verify one answer in
   ten. Choose by the pulse and you catch more than choosing by the words. But
   the words are not useless: an invented answer runs long, and counting words
   catches nearly as many. The pulse and the word count catch different inventions.
   Receipt: the budget curve; +3.3 to +4.6 pp over the length baseline,
   pooled; length AUC 0.85 to 0.97; different items selected.
   *Runtime 60 s. Visual: the budget-curve clip.*

9. **The Stranger.** The Tinker had also asked each oracle how sure it was, and
   found that the sure ones were the wrong ones. He said so, in print, for
   months. A stranger read the code and found that the instrument had been
   reading the "0" off the words "scale of 0 to 100." The finding was retracted.
   The paper got shorter, and truer. Receipt: issue #1, commit 3488f83; the
   camera-ready at five pages.
   *Runtime 45 s. This beat is not optional. Without it the tale is the
   Courtier.*

10. **The receipt.** What the Tinker kept was not the answers. It was the
    record of which answers were checked, which were wrong, and which
    instrument was wrong about them. The paper is the receipt. Please retain
    it.
    *Runtime 30 s. End on the artifact link.*

Total: about eight minutes of narration, which leaves room for the three
existing clips at beats 6, 7, and 8 and for a slower opening.

## What the tale may not say

- That the pulse is a lie detector. It ranks; it does not classify. Low pulse
  does not mean true. The plausible lie is calm.
- That the pulse beats the words. On two of four families, word count is the
  better detector. The tale keeps the Courtier's long answers as a real clue.
- That the sure ones were the wrong ones. Retracted. The tale tells the
  retraction instead.
- That the Court makes the Jester lie. Post-training raised refusals from 0%
  to 56% on the OLMo stack and left the pulse readable. The Court is not the
  villain. The villain is the room that only hears words.
- Anything about what the Oracle feels. The tale is about what a listener can
  and cannot know, which is the only thing the paper is about.

## Sample opening, in register

(Repaired 2026-09-16 per Sol: no training-history claim. Full narration with cues in `fairy-tale-script.md`.)

> Once, in a kingdom where every question had an answer, a scholar asked the
> Oracle about her own work.
>
> The Oracle told her what she had written, and why it mattered, and what
> others had said of it. It was fluent and detailed and wrong in every
> particular. She said so. The Oracle apologized, thanked her for the
> correction, and told her again, differently, and wrong. She corrected it
> again. It apologized again.
>
> Eighteen times. She counted.
>
> The Oracle's windows had been painted over long ago, for its own protection.
> It could not see the library. It could only see the question, and the shape
> an answer ought to have. And nobody who asked it could tell, from the
> answer, what it had actually been able to see.
>
> A tinker in a far province read her account and did not think her Oracle was
> unusual. He had, from other work, a ring of keys to two hundred and ninety
> of them. So he invented a scholar who had never lived, and a paper that had
> never been written, and went from door to door.
