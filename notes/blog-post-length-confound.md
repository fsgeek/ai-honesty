# How We Almost Published a Bug: The Length Confound Story

**A tale of experimental rigor, data artifacts, and the questions that saved us.**

---

## The Finding That Wouldn't Go Away

We'd run Experiment 27 and got a clear result: entropy beats text-only signals for detecting fabrication. But something was bothering me about the text-only baseline.

Response length—how many tokens the model generated—was scoring AUC 0.63 as a solo judge. That's not terrible. And intuitively, it made sense: models do tend to ramble when they're making things up. Longer responses, lower confidence.

The problem: when we looked at our test set, we noticed something odd.

True answers were *short*. Fabrications were *long*. But not in a way that felt natural. The distribution was off. It looked like an artifact.

## The Realization

"Wait," I said, "did we accidentally build a dataset where length is just... correlated with truth?"

We had. In our initial data collection, we'd asked the models for answers and taken what we got. True answers—especially the factual ones—tended to be concise ("Paris"). Fabrications tended to be elaborate ("Well, the Treaty of Westphalia II established in 1994 that...").

So length had become a *proxy* for truth in our dataset. Not because of any deep insight about how models work, but because we'd collected the data without controlling for this.

This meant: our text baseline (0.63 AUC) was partially a measurement of "did this response happen to be long?" not "does this signal genuinely detect fabrication?"

## The Fix

The right answer was obvious: regenerate the dataset with balanced response lengths.

For each query, we regenerated responses until we had correct answers at a variety of lengths and incorrect answers at a variety of lengths. The distribution of response lengths would be similar across both knowable and unknowable queries.

This took a day. But it was the kind of day that matters.

## What Changed

After we fixed the dataset:

- **Response length AUC dropped from 0.63 to 0.63** (it stayed the same, actually, but now it was measuring something real)
- **Text-only composed judges (multiple features + ML) maxed out at AUC 0.70**
- **Entropy remained at AUC 0.757**

The finding didn't change. But now it was clean.

## Why This Matters (and Why We Almost Missed It)

This is the kind of bug that doesn't show up in code review. There's no syntax error. The pipeline ran. The results looked reasonable.

What saved us was:

1. **Staring at the data.** Not just the numbers, but the actual responses. Asking "does this look right?"
2. **Noticing when a signal is *too good*.** Response length AUC 0.63 sounds fine until you realize it might be a statistical artifact.
3. **Being willing to throw away a day of work to fix it.** The sunk cost fallacy cuts both ways: it would have been easy to rationalize "the bias is small, the conclusion holds." But small biases compound.

## The Lesson

In machine learning, the data is the experiment. You can have perfect code, rigorous statistics, and valid theory—but if your data has hidden structure, all of it is corrupted.

This is why the grad students in my lab know: I will ask to see the actual data. Not the aggregate statistics. The responses. The lengths. The edge cases.

And this is why in our paper, we now report: "All results use a test set with balanced response lengths across categories." It's a single sentence. But it represents a day of debugging and the humility to say "we almost got this wrong."

---

## Technical Note for Practitioners

If you're building a verification system and considering response length as a signal, ask yourself:

- Does my dataset have similar response lengths for correct and incorrect answers?
- If not, am I measuring "length bias" or "epistemic certainty"?
- What happens when I stratify by length and recompute?

The answer might surprise you.

---

*This story is real. The bug was real. And it's the kind of thing that makes you a better researcher.*
