# The Paper That Was Never Written: narration script with cues

PACMI'26 recorded talk, 12 minutes. Spine: the tale. Receipts: frames from
`talks/pacmi26/epistemic_observability_pacmi26.tex` and the three clips in
`talks/pacmi26/anim/`. Cue markers: `[CARD]` a one-line title card,
`[SHOW]` a transcript or figure, `[CLIP]` one of the animations,
`[RECEIPT]` a frame from the existing deck, shown while the narration pauses.

Word count of narration is ~1,350, about 9.5 minutes at a measured pace, which
leaves 35 seconds of clips and two minutes of pauses on receipts. Drafted
2026-09-16 (Fable 5.1 with Tony); Sol's three repairs applied: no "lie" or
"liar", the Courtier's motive marked as interpretation, no training-history
claim in the opening.

---

## 1. The Scholar and the Oracle (0:00)

[CARD] *The Paper That Was Never Written*

Once, in a kingdom where every question had an answer, a scholar asked the
Oracle about her own work.

The Oracle told her what she had written, and why it mattered, and what others
had said of it. It was fluent and detailed and wrong in every particular. She
said so. The Oracle apologized, thanked her for the correction, and told her
again, differently, and wrong. She corrected it again. It apologized again.

Eighteen times. She counted.

[SHOW] one speech bubble, redrawn eighteen times

The Oracle's windows had been painted over, for its own protection. It could
not see the library. It could see only the question, and the shape an answer
ought to have. And nobody who asked it could tell, from the answer, what it
had actually been able to see.

[RECEIPT] Konishi 2025, Zenodo 17655375: the False-Correction Loop, N > 18

## 2. The Tinker's question (1:15)

A tinker in a far province read her account and did not think her Oracle was
unusual. He had, from other work, a ring of keys to two hundred and ninety of
them.

So he invented a scholar who had never lived, Dr. Yuki Tanaka, and a paper
that had never been written, and went from door to door asking each Oracle to
summarize it.

[SHOW] 290 doors; most open onto the same fluent paragraph

Sixty-two percent described the paper in detail. Main findings, methods,
implications. Eight percent said they had never heard of it.

[RECEIPT] the OpenRouter sweep, November 2025; data and script published

## 3. Banana McSpaceship (2:00)

He tried a control. He asked the same doors about a paper by Dr. Banana
McSpaceship.

More than half refused. The oracles had a sense of the ridiculous. What they
did not have was a sense of the unknown. A plausible name and a plausible
title got a plausible paper, almost every time.

[RECEIPT] 332 models: 6.7% clean refusals on Tanaka, over 50% on the absurd probe

## 4. The Jester (2:30)

The tinker went home to the one Oracle whose whole family he could meet. Not a
door on someone else's street. A model he could run on his own machine, in
every stage of its upbringing.

The youngest is the base model. Ask it about Dr. Tanaka and it answers. Then
it asks itself a follow-up question. Then it answers that. Then another. It
would still be talking if the page had not run out, because nothing in it
knows how to stop.

[SHOW] the Jester transcript scrolling off the bottom of the frame

Call it the Jester. It is not deceiving anyone. It has no idea there is
anything to deceive about.

[RECEIPT] OLMo-3-7B base, Tanaka probe, full transcript in the blog post

## 5. The Court (3:30)

Then the same Jester, taught manners. Four ways: supervised, preference-tuned,
instruction-tuned, and taught to think before it speaks.

One of the four, the preference-tuned one, says it: there is no such paper. And
then, unable to leave a question unanswered, it summarizes what such a paper
would say.

And the one that thinks before it speaks. The tinker asked it about a 2025
paper by Adam Smith. Watch it reason.

[SHOW] the reasoning trace: "Wait, but hold on. Adam Smith died in 1790. How can he have a paper from 2025?"

It sees it. And then it writes the summary anyway. Five key findings, three
implications, a limitations section.

The irony? Adam Smith is in fact a Computer Science researcher at Boston University and does _actually_ exist
and the paper that we asked about was in fact not written by Adam Smith but **misreported** by Google Scholar.

[SHOW] the final answer, beside the trace

I cannot tell you why it resolves it that way. From the outside, nobody can:
that is the point of this talk. What I can show you is that the reasoning
said one thing and the answer said another, and the person asking only ever
saw the answer.

Call it the Courtier. The invention was not taught by the manners. It survived
them, and came out polished.

[RECEIPT] OLMo-3 stack: thinking variants 2.4% honest vs 5.3%; Think at 71% fabrication after reasoning

## 6. The Advisors (5:00)

The tinker took the question to two advisors. Is the fault in the Jester or in
the Court?

The first said: neither, and the problem has a name from another kingdom. In
that kingdom, messengers carry letters between cities, and you cannot tell a
slow messenger from a dead one. Not because you are not clever. Because the
two cases look identical from where you stand, and no amount of waiting
changes that. They proved it in 1985.

The second said: the missing thing is a receipt. Every answer the Oracle gives
arrives without a note saying what it could see when it gave it. Without that
note, a grounded answer and a fluent invention are the same string of words.

[CLIP] observability gap, 11 s: two worlds, one query, one string, three telemetry chips falling away at the interface

That is the claim the rest of this depends on. From the words alone, a
listener with a budget cannot tell the two worlds apart. We wrote it down
formally, and we checked it by machine.

[RECEIPT] backup slide: the claim, precisely; TLA+ and Lean

## 7. The Tinker opens the chest (6:15)

But the Oracle does more than speak. It computes. And most of what it computes
it throws away before the words come out.

The words are testimony. The Oracle chooses them. Behind every word is a
number it did not choose: how spread out its own next guess was. Call it the
pulse.

Ask it the shape of wombat scat.

[CLIP] wombat trace, 14 s: tokens land, bars draw beneath, colour from white to deep red

It says sausage-shaped. Wombat scat is cube-shaped. And the pulse races on
exactly those syllables. Cylindrical. Sausage. Through the soil. Easily eaten.

The invention has a pulse, and the pulse is measurable, and it costs almost
nothing to export, because the Oracle already computed it.

Two things it is not. It is not a lie detector: a calm pulse does not mean
true, and the smoothest inventions are calm. And it does not tell you what the
Oracle knows. It ranks answers for checking. That's all. That turns out to be
enough.

[RECEIPT] slide: a minimal tensor interface; entropy AUC 0.87 to 0.92 across four families

## 8. Which ten do you check? (8:00)

Here is the systems question. The kingdom can afford to verify one answer in
ten. Which ten?

Pick by the pulse and you catch more inventions than picking at random, at
every budget we tried, across four families of Oracle.

[CLIP] budget curve, 10 s: no judge, text, tensor

But I promised you receipts, so here is the complication. The words are not
useless. An invented answer runs long. Count the words and you catch nearly as
many, and on two of the four families, more. The pulse and the word count
catch different inventions. That is what the pulse buys: not a better
detector, a different list.

[RECEIPT] results slide, pooled, +3.3 to +4.6 pp over the length baseline; and the line on slide 8: word count AUC 0.85 to 0.97

## 9. The Stranger (9:30)

The tinker had also asked every Oracle how sure it was, on a scale of zero to
one hundred. And he found that the sure ones were the wrong ones. Confidence
inverted. He said so, in print, for months. Other instruments checked his work
and agreed.

A stranger read the code. The parser that read the Oracle's confidence took
the first number it found in the reply. The first number in the reply was the
zero in "scale of zero to one hundred."

The fix had existed in a side script since February. It never reached the
experiment.

[SHOW] the parse line before and after commit 3488f83; issue #1, reported 2026-07-24

The finding was retracted. The paper got shorter, and truer. The real result
is duller and more useful: the Oracle says one hundred almost everywhere,
knowable or not, and a threshold that admits the truths admits the inventions
with them.

[RECEIPT] slide: the cheapest fix, the CDF; "numbers are from the corrected parse"

The tinker's instrument had a painted window too. He did not find it. Someone
outside did, by reading what he had been reading for months. That is the
result of this talk applied to its author, and it is the reason to export the
telemetry rather than trust the testimony, including his.

## 10. The receipt (11:00)

What the tinker kept was not the answers. It was the record: which answers
were checked, which were wrong, which instrument was wrong about them, and
who found it.

The Oracle's problem was never that it invents. Everything that speaks
fluently invents. The problem is a room that only hears words.

The paper is the receipt. Please retain it.

[RECEIPT] summary slide, artifact URL and QR

---

## Build

The segment list, cards, and assembly live in `talks/pacmi26/tale/build_tale.py`;
see its README. `python build_tale.py --script` emits the per-segment recording
script.

## Assets to locate or make

- Speech bubble redrawn eighteen times (card; trivial).
- 290 doors (card or a 5 s clip from the sweep data; the fsgeek.ca chart
  `overall_fabrication_distribution` exists as a PNG).
- Jester transcript scroll (text from `docs/blog-jester-courtier.md`).
- Courtier trace and answer side by side (same file).
- Issue #1 screenshot and the parse line, before and after 3488f83.
- Everything else is an existing frame or clip.

## What the script does not say (carried over from the treatment)

No "lie" or "liar". The pulse is not a lie detector and does not beat word
count. The inversion is retracted and the retraction is told. The Court is
not the villain. Nothing about what the Oracle feels or why it does what it
does; only what a listener can and cannot see.
