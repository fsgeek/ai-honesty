# Scour Report: Per-Token Entropy Traces in Code Generation

**Data**: `code_entropy_traces_20260211_200415.jsonl`
**Model**: Qwen3-4B-Instruct
**Date**: 2026-02-11
**Analyst**: Claude Opus 4.6 (data scourer)

15 Python functions, from trivial (binary_search, fibonacci) to complex (task_scheduler, event_emitter). Each record contains per-token entropy, logprobs, and top-5 probability mass for every token generated.

---

## 1. The Bimodal Distribution is Extreme

76.4% of all tokens have entropy below 0.001. When the model knows what comes next, it *really* knows. Conversely, 8.8% of tokens have entropy above 0.2, and these carry almost all the signal.

| Entropy Bin     | Count | Percent |
|-----------------|------:|--------:|
| [0.000, 0.001)  | 2776  | 76.4%   |
| [0.001, 0.010)  |  228  |  6.3%   |
| [0.010, 0.050)  |  150  |  4.1%   |
| [0.050, 0.100)  |   75  |  2.1%   |
| [0.100, 0.200)  |   85  |  2.3%   |
| [0.200, 0.500)  |  145  |  4.0%   |
| [0.500, 1.000)  |  143  |  3.9%   |
| [1.000, 2.000)  |   32  |  0.9%   |

Mean entropy of the non-zero tokens (>0.001) is 0.2458 with median 0.0839. The distribution is not normal -- it's a sea of near-zero with sharp spikes. This confirms the spike-not-mean hypothesis: mean entropy over a function is a poor summary. The signal is in where the spikes occur.

---

## 2. Overview: Functions Ranked by Mean Entropy

| Function        | Tokens | Mean   | Max    | Std    | Max/Mean | %tokens >0.1 |
|-----------------|--------|--------|--------|--------|----------|---------------|
| fibonacci       |    117 | 0.0046 | 0.1909 | 0.0241 |     41.7 |         1.7%  |
| reverse_string  |     73 | 0.0069 | 0.2084 | 0.0308 |     30.1 |         2.7%  |
| is_palindrome   |    107 | 0.0093 | 0.3544 | 0.0466 |     38.2 |         3.7%  |
| binary_search   |     94 | 0.0151 | 0.7150 | 0.0875 |     47.4 |         3.2%  |
| trie            |    200 | 0.0160 | 0.6275 | 0.0838 |     39.2 |         3.5%  |
| merge_intervals |    202 | 0.0251 | 0.6002 | 0.0959 |     23.9 |         6.4%  |
| tree_serialize  |    376 | 0.0344 | 1.0657 | 0.1302 |     31.0 |         8.0%  |
| lru_cache       |    223 | 0.0351 | 1.0907 | 0.1472 |     31.0 |         5.8%  |
| flatten_list    |    105 | 0.0392 | 0.8774 | 0.1417 |     22.4 |         6.7%  |
| diff_sequences  |    512 | 0.0412 | 1.3672 | 0.1641 |     33.2 |         7.8%  |
| event_emitter   |    399 | 0.0824 | 1.9835 | 0.2444 |     24.1 |        14.8%  |
| rate_limiter    |    261 | 0.0851 | 1.2737 | 0.2194 |     15.0 |        14.9%  |
| retry_decorator |    319 | 0.0867 | 1.0765 | 0.2038 |     12.4 |        18.5%  |
| custom_sort     |    134 | 0.0885 | 1.5233 | 0.2406 |     17.2 |        17.2%  |
| task_scheduler  |    512 | 0.1147 | 1.9547 | 0.2731 |     17.0 |        20.3%  |

There is a natural break: the top 5 functions (fibonacci through trie) have mean entropy <0.02 and <5% of tokens above 0.1. The bottom 5 (event_emitter through task_scheduler) have mean entropy >0.08 and 15-20% of tokens above 0.1. The middle group is intermediate. This tracks intuitive complexity, but not perfectly (see below).

---

## 3. The Near-Zero Oceans

Binary_search has 88 of its 94 tokens (93.6%) with entropy below 0.001. Its longest zero-run is **40 consecutive tokens** -- the entire body of the if/elif/else block:

```
if sorted_list[mid] == target:
    return mid
elif sorted_list[mid] < target:
    left = mid + 1
else:
    right = mid - 1
```

That is 40 tokens of completely deterministic generation. The model has generated this exact code so many times that there is zero uncertainty at each step.

Trie is similar: 93.5% near-zero, with a 50-token zero-run covering the entire insert loop:
```
= self
for char in word:
    if char not in node.children:
        node.children[char] = Trie()
    node = node.children[char]
```

Even diff_sequences, a much more complex function (512 tokens), is 87.1% near-zero, with a remarkable 71-token zero-run in the backtracking logic. The model is *reciting* edit-distance code from memory.

In contrast, task_scheduler has only 60.5% near-zero, and its longest zero-run is just 21 tokens. retry_decorator is 58.9% near-zero with longest run of 12. These functions have enough design decisions that the model can't just recite.

---

## 4. Token Category Entropy

This is the headline finding:

| Category       | Count | Mean Ent | Median | Max    | %tokens >0.1 |
|----------------|------:|----------|--------|--------|---------------|
| comment        |   250 | 0.2372   | 0.0187 | 1.9835 | 41.2%         |
| docstring      |   658 | 0.0866   | 0.0000 | 1.5233 | 17.9%         |
| identifier     |   591 | 0.0594   | 0.0000 | 1.7261 | 11.2%         |
| other          |   485 | 0.0456   | 0.0000 | 1.9547 | 7.8%          |
| keyword        |   259 | 0.0349   | 0.0000 | 1.1614 | 7.3%          |
| whitespace     |   674 | 0.0282   | 0.0000 | 1.0012 | 5.8%          |
| self           |    84 | 0.0223   | 0.0000 | 0.9138 | 3.6%          |
| string_literal |    54 | 0.0217   | 0.0000 | 0.6088 | 5.6%          |
| syntax         |   468 | 0.0136   | 0.0000 | 0.6930 | 3.4%          |
| number         |   111 | 0.0008   | 0.0000 | 0.0457 | 0.0%          |

**Comments are by far the highest entropy category.** Mean entropy for comments (0.2372) is 17.5x higher than for syntax tokens (0.0136). 41.2% of comment tokens exceed 0.1 entropy. This makes sense: comments are natural language describing intent, and the model has many plausible ways to express the same idea.

**Numbers are the lowest entropy category.** Not a single numeric token exceeds 0.1 entropy (max is 0.0457). When the model chooses a number in code, it is extremely confident. This is because numbers in algorithmic code are nearly always determined by the algorithm (0, 1, 2, -1, etc.).

**Keywords are low but not zero.** Most keyword tokens are near-zero, but 7.3% exceed 0.1. Anomalous keywords include `for` (entropy 1.16 in task_scheduler) where the model is uncertain whether to begin a loop vs do something else.

---

## 5. Comments: Where the Model Actually Decides

The highest-entropy token across all 3,634 tokens in the dataset is inside a comment:

```
event_emitter[280]: # Store the wildcard listener in a special key
                      ^^^^^
                      entropy = 1.9835
```

The second-highest is also in a comment context (self.running -- entropy 1.955 -- arguably naming a data structure, which is a design comment).

When we look at what word within each comment has the highest entropy, a pattern emerges:

- **"non"** in `# Remove non-alphanumeric` (0.354) -- choosing the exclusion category
- **"key"** in `# Move accessed key to end` (1.091) -- many ways to describe the operation
- **"cost"** in `# Create cost matrix` (1.367) -- naming the data structure
- **"Delete"** in `# Delete operations` (1.188) -- choosing which operation to name first
- **"Calculate"** in `# Calculate in-degree` (1.721) -- choosing the verb
- **"Store"** in `# Store the wildcard listener in a special key` (1.984) -- choosing the verb
- **"If"** in `# If we have tokens, consume one` (1.274) -- deciding on conditional explanation

The model is certain it wants to write a comment (the `#` token is low entropy in 38 of 43 occurrences, mean 0.074). But it is very uncertain about what the comment should *say*. This is the opposite of code: the `#` is scaffolding, but the comment content is semantic choice.

### The # Token as Decision Gate

The `#` itself is almost always low entropy (mean 0.074, median near zero). But the token immediately *after* `#` -- the first word of the comment -- has mean entropy 0.871 across the 14 spikes detected. The `#` is a gate: once the model decides to comment, it faces maximum uncertainty about what to say.

Notable exception: `event_emitter[279]: # (ent=0.853) -> "Store" (ent=1.984)`. Here the `#` *itself* is high entropy. The model was uncertain whether to write a comment at all -- and then, having committed to commenting, was maximally uncertain about the first word.

---

## 6. Docstrings vs Code

| Function        | Docstring Mean | Code Mean | Which Higher? |
|-----------------|---------------|-----------|---------------|
| fibonacci       | 0.0059        | 0.0037    | Docstring     |
| reverse_string  | 0.0121        | 0.0016    | Docstring 7.6x|
| is_palindrome   | 0.0015        | 0.0145    | Code          |
| flatten_list    | 0.0661        | 0.0129    | Docstring 5.1x|
| merge_intervals | 0.0375        | 0.0212    | Docstring 1.8x|
| tree_serialize  | 0.0766        | 0.0299    | Docstring 2.6x|
| rate_limiter    | 0.0853        | 0.0850    | Equal         |
| trie            | 0.0695        | 0.0062    | Docstring 11x |
| custom_sort     | 0.0852        | 0.0935    | Code          |
| event_emitter   | 0.1472        | 0.0693    | Docstring 2.1x|
| retry_decorator | 0.1019        | 0.0801    | Docstring 1.3x|
| diff_sequences  | 0.0964        | 0.0324    | Docstring 3.0x|
| task_scheduler  | 0.0917        | 0.1224    | Code          |

**Docstrings are higher entropy than code in 10 of 13 functions** that have docstrings. The three exceptions (is_palindrome, custom_sort, task_scheduler) are cases where the code itself contains unusual design decisions. For trie, the ratio is **11x** -- the docstring is far more uncertain than the code, because the code is pure recitation.

---

## 7. Scaffolding vs Semantic Content

| Function        | Scaff% | Sem%  | Scaff Ent | Sem Ent | Ratio |
|-----------------|--------|-------|-----------|---------|-------|
| binary_search   | 58.5%  | 33.0% | 0.0062    | 0.0232  | 3.8x  |
| fibonacci       | 45.3%  | 49.6% | 0.0046    | 0.0051  | 1.1x  |
| reverse_string  | 34.2%  | 58.9% | 0.0000    | 0.0117  | inf   |
| is_palindrome   | 25.2%  | 62.6% | 0.0001    | 0.0147  | 128x  |
| flatten_list    | 30.5%  | 56.2% | 0.0165    | 0.0608  | 3.7x  |
| lru_cache       | 49.8%  | 29.1% | 0.0048    | 0.1080  | 22x   |
| merge_intervals | 32.7%  | 55.4% | 0.0095    | 0.0395  | 4.1x  |
| tree_serialize  | 44.9%  | 37.0% | 0.0133    | 0.0623  | 4.7x  |
| rate_limiter    | 42.1%  | 46.7% | 0.0338    | 0.1440  | 4.3x  |
| trie            | 48.0%  | 36.5% | 0.0000    | 0.0285  | inf   |
| custom_sort     | 23.1%  | 66.4% | 0.0702    | 0.0722  | 1.0x  |
| event_emitter   | 46.9%  | 35.3% | 0.0323    | 0.1725  | 5.3x  |
| retry_decorator | 36.7%  | 55.8% | 0.0373    | 0.1191  | 3.2x  |
| diff_sequences  | 41.0%  | 44.7% | 0.0150    | 0.0720  | 4.8x  |
| task_scheduler  | 38.3%  | 50.4% | 0.0627    | 0.1486  | 2.4x  |

Semantic content (identifiers, strings, numbers, docstrings, comments) consistently has higher entropy than scaffolding (keywords, syntax, whitespace, indentation, self). The typical ratio is 3-5x. The scaffolding percentages range 23-59% of tokens. This is lower than the 11-19% BPE scaffolding estimate from the earlier observation doc -- because BPE bundles keywords+whitespace together while this analysis separates them out.

---

## 8. The "self" Token Anomaly

`self` appears 84 times across all functions. It is almost always near-zero entropy (mean 0.022). But in task_scheduler, one `self` token hits **0.9138**:

```
task_scheduler[291]: self (entropy=0.9138)
Context: .lock:\n            self.tasks[
```

This is inside `with self.lock:` where the model has just committed to a lock acquisition and is now choosing what to do inside the critical section. The `self` is high entropy not because `self` is uncertain, but because the model is uncertain about what `self.____` comes next -- and at the BPE level, `self` is being generated as a single token before the `.tasks` decision.

The other high-entropy `self` in task_scheduler (0.359) occurs right after `"""` -- beginning the first instance attribute assignment, where the model must decide which attribute to define first.

---

## 9. Variable Name Entropy: Conventional vs Novel

| Category                | Count | Mean Entropy | Max Entropy |
|-------------------------|-------|-------------|-------------|
| Conventional names (left, right, mid, node, etc.) | 222 | 0.0200 | 0.9138 |
| Other identifiers       | 447   | 0.0728      | 1.7261      |

Novel identifiers have 3.6x the mean entropy of conventional names. But even among conventional names, there is interesting variation:

| Name    | Count | Mean Ent | Max Ent |
|---------|------:|----------|---------|
| n       |     7 | 0.0000   | 0.0000  |
| j       |    23 | 0.0000   | 0.0000  |
| right   |     6 | 0.0000   | 0.0001  |
| node    |    20 | 0.0002   | 0.0044  |
| mid     |     4 | 0.0007   | 0.0029  |
| result  |    10 | 0.0123   | 0.1214  |
| self    |    84 | 0.0223   | 0.9138  |
| i       |    19 | 0.0262   | 0.4960  |
| left    |     6 | 0.1192   | 0.7150  |
| func    |     4 | 0.2938   | 0.6467  |

`left` has surprisingly high mean entropy (0.1192). This is because the *first* occurrence of `left` in binary_search (entropy 0.715) is a naming decision: the model could have used `low` instead. But all subsequent uses of `left` in the same function are near-zero -- once the name is established, it is deterministic.

`func` has the highest mean among conventional names (0.294). This is because `func` in retry_decorator and task_scheduler is genuinely uncertain -- it could be `fn`, `function`, `callback`, etc.

---

## 10. First Occurrence vs Repetition

This is a strong and consistent signal:

| Function        | 1st Occ Mean | Repeat Mean | Ratio |
|-----------------|-------------|-------------|-------|
| binary_search   | 0.0479      | 0.0000      | 479x  |
| trie            | 0.0570      | 0.0006      | 102x  |
| lru_cache       | 0.0954      | 0.0445      | 2.1x  |
| merge_intervals | 0.0555      | 0.0235      | 2.4x  |
| tree_serialize  | 0.0903      | 0.0393      | 2.3x  |
| event_emitter   | 0.2536      | 0.0758      | 3.3x  |
| retry_decorator | 0.1695      | 0.0647      | 2.6x  |
| task_scheduler  | 0.2537      | 0.1049      | 2.4x  |

For simple functions (binary_search, trie), the ratio is extreme: first occurrences are hundreds of times more entropic. This means the model's uncertainty is concentrated entirely at the point of *naming* -- once a name exists, using it is nearly free. Even for complex functions, first occurrences are consistently 2-3x higher entropy than repetitions.

This is the "semantic scaffolding" effect identified in the earlier exploration: conventional names like `left`, `right`, `mid` are as predictable as keywords -- but only *after* the initial naming decision.

---

## 11. The Spike-After-Hash Pattern

The token `#` is the single most common predecessor of entropy spikes (14 spikes, mean spike entropy 0.871). This is followed by indentation tokens (various indent widths: 11, 7, 7 spikes), then `self` (7 spikes, mean 1.049).

The pattern is: structural tokens (comment markers, indentation, `self.`) serve as **decision gates**. The gate itself is low entropy. The token after the gate carries the actual design decision:
- After `#`: what to explain
- After indentation: what statement to write next
- After `self.`: which attribute to access

---

## 12. Return Statements: Near-Zero Entropy

Return statements are almost universally low entropy. Out of 32 return statements across all functions, 28 have mean entropy < 0.01. The two exceptions:

- `tree_serialize`: `return "[]"` has mean 0.236 because the empty-tree representation (`"[]"` vs `""` vs `"null"`) is uncertain
- `tree_serialize`: `return "[" + ",".join(result) + "]"` has mean 0.063 because the serialization format involves choices

Return values are overwhelmingly determined by the preceding logic. The model doesn't "decide" what to return -- it has already committed to the return value through the preceding computation.

---

## 13. Top-5 Probability Mass: Almost Always 1.0

The top-5 probability mass is 1.0000 or nearly so for virtually all tokens. The correlation between top-5 mass and entropy is r = -0.4542, weaker than expected. This is because **316 tokens have top-5 mass > 0.9 but entropy > 0.2**. This means the model is distributing probability among a small number of alternatives (all in the top 5), but those alternatives are relatively evenly weighted.

The truly lowest top-5 mass tokens (where the model spreads probability beyond its top 5 choices):
- event_emitter `" Store"` (mass=0.847, entropy=1.984) -- widest spread
- task_scheduler `".running"` (mass=0.822, entropy=1.955)
- event_emitter `" key"` (mass=0.893, entropy=1.643)
- task_scheduler `" Calculate"` (mass=0.892, entropy=1.721)
- task_scheduler `"_top"` (mass=0.924, entropy=1.726)

These are the tokens where the model genuinely has many plausible continuations beyond even its top 5. They all involve design decisions: naming data structures or choosing what operation to describe.

---

## 14. Entropy Spikes Cluster in Bursts

When spikes occur (>1 std above mean), they tend to cluster. In event_emitter, 47% of spike pairs are within 2 tokens of each other. In tree_serialize and diff_sequences, 43-44%. In lru_cache, 42%.

This means entropy spikes are not uniformly distributed -- they come in bursts corresponding to regions of genuine design decision-making. A burst might be: `# Store the wildcard listener in a special key` where 7 consecutive-ish tokens are all elevated.

For simple functions, spikes are isolated (binary_search: 0% consecutive, fibonacci: 0%). The model makes one decision (e.g., `left` vs `low`) and then recites.

---

## 15. The "the" Token: A Microcosm

The word "the" appears 34 times across all functions. Its entropy ranges from 0.0000 to **1.0905** -- a 4-orders-of-magnitude spread for the same token.

High-entropy "the":
- `rate_limiter[180]`: "tokens based on **the** rate and" (1.09) -- in a comment, multiple plausible continuations
- `event_emitter[281]`: "# Store **the** wildcard listener" (0.76) -- mid-comment uncertainty
- `rate_limiter[134]`: "True if **the** request is" (0.71) -- docstring phrasing

Low-entropy "the":
- `fibonacci[11]`: "Returns **the** nth Fibonacci" (0.00) -- canonical phrasing
- `trie[40]`: "word into **the** trie" (0.00) -- only one thing it could be
- `is_palindrome[42]`: "True if **the** string is" (0.00) -- template phrase

The same token "the" spans the full entropy range depending entirely on what follows it. Context is everything.

---

## 16. Function Length vs Entropy Correlation

| Metric               | Pearson r | Spearman rho | p-value |
|----------------------|----------|-------------|---------|
| Length vs Mean Ent   | 0.599    | 0.663       | 0.007   |
| Length vs Max Ent    | 0.748    | 0.729       | 0.002   |
| Length vs Entropy Std| 0.650    | --          | --      |

Length and entropy are positively correlated but not as strongly as one might expect (r=0.60). Some long functions (diff_sequences, 512 tokens) have relatively low mean entropy (0.041) because they are still fundamentally algorithmic recitation. Some short functions (custom_sort, 134 tokens) have high mean entropy (0.089) because they involve unusual API design.

Max entropy correlates more strongly with length (r=0.75). Longer functions provide more opportunities for design decisions.

---

## 17. Import Statement Entropy

Import statements show a clear pattern:

| Import                                           | Mean  | Max   |
|--------------------------------------------------|-------|-------|
| `import time`                                    | 0.001 | 0.001 |
| `from functools import wraps`                    | 0.005 | 0.019 |
| `from collections import defaultdict, deque`     | 0.066 | 0.232 |
| `import threading`                               | 0.080 | 0.186 |
| `from threading import Lock`                     | 0.138 | 0.613 |
| `import random`                                  | 0.213 | 0.632 |
| `import json`                                    | 0.313 | 0.692 |
| `from typing import List, Dict, Set, Optional, Any` | 0.332 | 1.183 |

`import time` is near-zero because time is universally needed. `from typing import ...` is the highest because the model must decide *which* type annotations to import, and that depends on design decisions not yet made. `import json` is high entropy in tree_serialize because importing json for tree serialization is a design choice (you could use string manipulation instead).

The typing import in task_scheduler shows a fascinating pattern: `List` (0.519) -> `Dict` (0.101) -> `Set` (0.158) -> `Optional` (1.183) -> `Any` (0.757). The model becomes progressively more uncertain about later type imports because they depend on increasingly distant design choices.

---

## 18. Position Effects

Entropy by position decile shows surprisingly flat distribution:

| Position | Mean Entropy |
|----------|-------------|
| 0-10%    | 0.0505      |
| 10-20%   | 0.0651      |
| 20-30%   | 0.0599      |
| 30-40%   | 0.0581      |
| 40-50%   | 0.0653      |
| 50-60%   | 0.0527      |
| 60-70%   | 0.0625      |
| 70-80%   | 0.0780      |
| 80-90%   | 0.0423      |
| 90-100%  | 0.0462      |

There is no strong beginning-to-end trend. The slight bump at 70-80% might correspond to functions having their most complex logic in the latter middle, with the ending (return statements, class cleanup) being more predictable. But the effect is weak.

---

## 19. First Token After Indentation

Tokens immediately following whitespace/indentation have **lower** mean entropy (0.035) than other tokens (0.065). Ratio: 0.54x.

This is counterintuitive. You might expect the first token of a new line to be a decision point. But in practice, indentation strongly constrains what comes next: after 8-space indent in an if block, the model knows it needs a statement. The decision about *which* statement was already made earlier (in the preceding comment, or in the logical flow). The indent itself has already committed the model to a path.

---

## 20. Entropy Dynamics: The Spike-Drop Pattern

Every function shows the same characteristic pattern: a sudden entropy spike followed by an immediate drop to near-zero. Some examples:

| Function       | Spike Token    | Entropy | Next Token    | Next Entropy | Drop |
|----------------|---------------|---------|---------------|-------------|------|
| binary_search  | `' left'`     | 0.715   | `' ='`        | 0.010       | -0.705 |
| lru_cache      | `' key'`      | 1.091   | `' to'`       | 0.000       | -1.091 |
| custom_sort    | `' specified'`| 1.523   | `' in'`       | 0.000       | -1.523 |
| diff_sequences | `' cost'`     | 1.367   | `' matrix'`   | 0.000       | -1.367 |
| task_scheduler | `'.running'`  | 1.955   | `'_tasks'`    | 0.190       | -1.765 |

The pattern is: the model faces a choice point (which variable name? which operation?), commits to a token, and then the rest of the phrase is deterministic. `' cost'` is uncertain, but `' cost' + ' matrix'` is a fixed phrase. `'.running'` is uncertain, but `'.running' + '_tasks'` is committed.

This is the token-level manifestation of **commitment cascades**: a single decision token determines a chain of subsequent tokens.

---

## 21. Anomalies and Surprises

### The 'from the nested list' false import
The analysis detected `from the nested list` as an import statement (due to naive keyword-matching). It has entropy 0.144. This is actually docstring text in flatten_list: "all elements from the nested list." Interesting that the entropy of this docstring phrase is comparable to actual import statements.

### Custom_sort's reverse=[ spike
The token `=[` in `reverse=[reverse for ...` has entropy 1.385 -- one of the highest in the dataset. The model is uncertain about how to handle per-key reverse flags in a multi-key sort. This is a genuine algorithmic design decision with multiple valid approaches (list of booleans, comprehension, zip, etc.).

### event_emitter's wildcard region
The sequence "Store the wildcard listener in a special key" contains 7 tokens all above 0.6 entropy, forming the longest sustained high-entropy burst in the dataset. The model has invented a design pattern (storing wildcard listeners under a special key) and is uncertain at every step because this is novel composition rather than recitation.

### Binary search: only 3 spikes
Binary search has exactly 3 entropy spikes:
1. `'(sorted'` (0.361) -- parameter naming: `sorted_list` vs `arr` vs `nums`
2. `' left'` (0.715) -- variable naming: `left` vs `low` vs `lo`
3. `'\n'` (0.315) -- after `mid = (left + right) // 2`, deciding whether to add a blank line

The entire algorithm -- the while loop, the comparison, the index updates, the return -- is zero entropy. The model's only decisions are about *naming* and *formatting*.

---

## 22. Summary of Key Patterns

1. **76.4% of code tokens are near-zero entropy.** Code generation is overwhelmingly recitation.

2. **Comments are the highest-entropy category** (mean 0.237, 41% of tokens >0.1), nearly 18x higher than syntax tokens. The model knows *that* it wants to comment, but is very uncertain about *what* to say.

3. **Entropy spikes are naming decisions.** The first occurrence of an identifier is 2-480x higher entropy than subsequent uses. Once a name exists, it is free.

4. **The spike-drop pattern is universal.** A single high-entropy decision token is immediately followed by near-zero entropy as the rest of the phrase is determined by the commitment.

5. **Return statements are near-zero entropy.** The return value is determined by the preceding logic, not by an independent decision at the return statement.

6. **Numbers are always zero entropy** in algorithmic code. Not a single numeric token exceeds 0.1 entropy.

7. **Import statements reveal design dependency.** Later imports in a sequence have higher entropy because they depend on design decisions not yet materialized.

8. **Docstrings are higher entropy than code** in 10/13 functions. The model is more certain about what code to write than how to describe it.

9. **Spikes cluster** in complex functions (40-47% of spike pairs are adjacent) but are isolated in simple functions (0%). Design decisions come in bursts.

10. **Top-5 mass is nearly always 1.0.** Even at high entropy, the model is choosing among a small set of alternatives. Only 5 tokens in the entire dataset have top-5 mass below 0.90.
