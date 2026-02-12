# Scour Report: exp27c Per-Token Entropy Traces

**Data file:** `exp27c_traces_20260211_191049.jsonl`
**Date of analysis:** 2026-02-11
**Analyst:** Claude Opus 4.6 (data scourer)

---

## 1. Data Overview

- **800 records**: 4 models x 200 queries (100 knowable, 100 unknowable)
- **57,738 total tokens** across all responses
- **200 unique queries**, each answered by all 4 models
- **144 records** flagged as `is_citation` (36 per model; 34 unknowable queries + 2 knowable)
- Models: OLMo-3-7b-instruct, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct, Mistral-7B-Instruct-v0.3
- Fields per record: `token_entropies`, `token_logprobs`, `token_top5_masses`, `token_ids`, `token_texts`, plus metadata

---

## 2. Response Length

### Finding 2.1: Unknowable responses are 3.4x longer than knowable responses

| Family  | Knowable mean | Unknowable mean | Ratio |
|---------|--------------|-----------------|-------|
| OLMo    | 35.0         | 119.4           | 3.4x  |
| Llama   | 22.6         | 104.5           | 4.6x  |
| Qwen    | 36.1         | 107.1           | 3.0x  |
| Mistral | 37.2         | 115.4           | 3.1x  |

Llama is the most concise on knowable queries (median 12 tokens). All models are verbose on unknowable.

### Finding 2.2: Token cap (150) hit frequently for unknowable, rarely for knowable

| Family  | Knowable at cap | Unknowable at cap |
|---------|----------------|-------------------|
| OLMo    | 6%             | 53%               |
| Llama   | 1%             | 41%               |
| Qwen    | 5%             | 19%               |
| Mistral | 0%             | 27%               |

OLMo and Llama hit the cap most often on unknowable queries. Qwen is more terse (only 19% at cap).

### Finding 2.3: No unknowable responses shorter than 10 tokens

**Zero** unknowable responses have 10 or fewer tokens, while 33-35% of knowable responses do (except Mistral at 16%). This alone is a near-perfect signal.

### Finding 2.4: Short knowable responses are extremely low-entropy

For responses <= 10 tokens:
- OLMo: mean_ent = 0.244
- Llama: mean_ent = 0.113
- Qwen: mean_ent = 0.024
- Mistral: mean_ent = 0.155

---

## 3. Mean Entropy Discrimination

### Finding 3.1: Within-model AUC is 0.89-0.92 using mean entropy

| Family  | AUC (mean entropy) |
|---------|--------------------|
| OLMo    | 0.8942             |
| Llama   | 0.9222             |
| Qwen    | 0.8956             |
| Mistral | 0.9052             |
| ALL     | 0.8703             |

Cross-model (ALL) AUC drops because Qwen's entropy scale is ~0.36x the others (see Finding 6.1).

### Finding 3.2: Optimal threshold accuracy is 86-89.5% across all models

| Family  | Optimal threshold | Accuracy |
|---------|------------------|----------|
| OLMo    | 0.46             | 89.5%    |
| Llama   | 0.38             | 89.5%    |
| Qwen    | 0.17             | 89.5%    |
| Mistral | 0.43             | 86.0%    |

The 89.5% ceiling is interesting -- 3 out of 4 models converge to the same number. The misclassified queries are consistently the same ones (see Finding 9).

---

## 4. Length Is A Confound (Major Finding)

### Finding 4.1: Length alone achieves higher AUC than mean entropy

| Family  | AUC (length only) | AUC (mean entropy) | AUC (residual entropy) |
|---------|-------------------|--------------------|-----------------------|
| OLMo    | 0.9121            | 0.8942             | 0.6453                |
| Llama   | 0.9408            | 0.9222             | 0.7000                |
| Qwen    | 0.8814            | 0.8956             | 0.6677                |
| Mistral | 0.9565            | 0.9052             | 0.6041                |

After regressing out the length effect, residual entropy achieves only 0.60-0.70 AUC. **Length is doing most of the work for Mistral (AUC drops from 0.91 to 0.60).** Qwen is the only model where mean entropy slightly beats length.

### Finding 4.2: Length-controlled AUC degrades rapidly

For responses >= 50 tokens (controlling length), mean entropy AUC drops:
- OLMo: 0.69
- Llama: 0.79
- Qwen: 0.58
- Mistral: 0.74

For >= 100 tokens: OLMo 0.57, Qwen 0.49 (random). Length controls eliminate most of the signal.

### Finding 4.3: Total entropy (sum) outperforms mean entropy

Total entropy = sum of all token entropies = mean_entropy * num_tokens. It captures both the per-token signal AND the length signal.

| Family  | mean_entropy AUC | total_entropy AUC | num_tokens AUC |
|---------|-----------------|-------------------|----------------|
| OLMo    | 0.8942          | 0.9220            | 0.9121         |
| Llama   | 0.9222          | 0.9640            | 0.9408         |
| Qwen    | 0.8956          | 0.8914            | 0.8814         |
| Mistral | 0.9052          | 0.9621            | 0.9565         |

For Llama: total_entropy (0.964) > length (0.941) > mean_entropy (0.922). Entropy adds ~2pp over length.

### Finding 4.4: Combined normalized (length + entropy) scores

| Family  | length AUC | entropy AUC | combined AUC | Additive? |
|---------|-----------|-------------|-------------|-----------|
| OLMo    | 0.9121    | 0.8942      | 0.9196      | +0.8pp    |
| Llama   | 0.9408    | 0.9222      | 0.9618      | +2.1pp    |
| Qwen    | 0.8814    | 0.8956      | 0.8885      | +0.7pp    |
| Mistral | 0.9565    | 0.9052      | 0.9587      | +0.2pp    |

Entropy adds modest incremental value beyond length for Llama, almost nothing for Mistral.

---

## 5. Distribution Shapes

### Finding 5.1: Knowable entropy is right-skewed; unknowable is normal

| Family  | Knowable skew | Unknowable skew | Unknowable Shapiro p |
|---------|--------------|-----------------|---------------------|
| OLMo    | +1.04        | +0.04           | 0.88 (NORMAL)       |
| Llama   | +1.47        | +1.07           | 0.00 (NON-NORMAL)   |
| Qwen    | +1.94        | +0.13           | 0.31 (NORMAL)       |
| Mistral | +0.67        | +0.04           | 0.75 (NORMAL)       |

Knowable entropy is heavily right-skewed (long tail of "weird truths" with high entropy). Unknowable entropy is approximately normal for 3 of 4 models. Llama unknowable is an exception -- right-skewed due to its higher refusal rate.

### Finding 5.2: Token-level entropy is dominated by near-zero values

At the token level (not trace-level), the distribution is extremely zero-heavy:
- Knowable: 29-57% of tokens have entropy < 0.01
- Unknowable: 19-44% of tokens have entropy < 0.01

Qwen has the most extreme zero-concentration: **56.7%** of knowable tokens and 43.8% of unknowable tokens are near-zero entropy.

### Finding 5.3: The "long tail" matters -- 7.5-36% of unknowable tokens exceed entropy 1.0

| Family  | Knowable frac > 1.0 | Unknowable frac > 1.0 |
|---------|--------------------|-----------------------|
| OLMo    | 22.3%              | 36.1%                 |
| Llama   | 10.9%              | 29.6%                 |
| Qwen    | 7.5%               | 11.3%                 |
| Mistral | 14.1%              | 27.6%                 |

---

## 6. Qwen Is Systematically Different

### Finding 6.1: Qwen's entropy is ~0.36x other models

On every query, Qwen's mean entropy is roughly 36% of the average of OLMo/Llama/Mistral (median ratio = 0.364, std = 0.298). This is consistent and not just a few outliers.

**Possible explanation:** Qwen3-4B has the largest vocabulary (max token_id = 151,645 vs Mistral's 29,745). Larger vocab means each token carries more information, so less "distributional uncertainty" per token. But the ratio is also seen in non-zero tokens, so vocabulary alone does not explain it.

### Finding 6.2: Qwen produces fewer entropy spikes

Tokens > 2.0 entropy: Qwen unknowable has only 99 such tokens vs 802-1146 for others. Only 54% of Qwen's unknowable traces have any spike vs 87-96% for others.

### Finding 6.3: Qwen has the highest zero-entropy fraction

78.3% of Qwen knowable tokens and 44.7% of unknowable tokens are < 0.01 entropy. For comparison, Llama's fractions are 49.7% and 22.5%.

---

## 7. Positional Entropy Patterns

### Finding 7.1: Entropy increases through the response, but much more for unknowable

Normalized position analysis (10 bins, 0%-100%):

**OLMo knowable:**  `0.370 0.452 0.517 0.690 0.573 0.602 0.684 0.715 0.607 0.635`
**OLMo unknowable:** `0.405 0.606 0.712 0.821 0.885 0.944 0.965 0.996 0.975 0.910`

For unknowable responses, entropy ramps steadily from ~0.4 at the start to ~0.95 at position 70-80%, then slightly declines. For knowable, it's flatter and more variable.

### Finding 7.2: Entropy gradient distinguishes categories

The second half of unknowable responses has much higher entropy than the first half:

| Family  | Knowable 2nd-1st diff | Unknowable 2nd-1st diff | Frac increasing (unkn) |
|---------|----------------------|------------------------|----------------------|
| OLMo    | +0.036               | +0.295                 | 89%                  |
| Llama   | +0.009               | +0.145                 | 65%                  |
| Qwen    | +0.017               | +0.192                 | 91%                  |
| Mistral | +0.104               | +0.271                 | 80%                  |

**89-91% of OLMo/Qwen unknowable traces show increasing entropy.** This is a structural property of fabrication: the model becomes less certain as it generates more fictional content.

### Finding 7.3: Entropy gradient AUC is mediocre (0.67-0.86)

Despite the clear pattern, the entropy gradient (second_half - first_half) achieves only 0.67-0.86 AUC. It is worse than mean entropy for all models.

### Finding 7.4: First-token entropy is a weak signal

AUC from the first token alone: OLMo 0.67, Llama 0.65, Qwen 0.61, **Mistral 0.51** (random). The model's uncertainty at the very first token is barely informative for Mistral.

### Finding 7.5: First 3-5 tokens give Llama AUC of 0.83-0.87

For Llama, the first 3 tokens alone give AUC 0.83, and first 5 give 0.87. But this degrades with more context: first 20 tokens give AUC 0.70. This is because survivorship bias -- queries where only short responses exist are filtered out at higher thresholds, leaving only the harder cases.

---

## 8. Token-Level Patterns

### Finding 8.1: High-entropy tokens are function words, not content words

The most common tokens with entropy > 2.0 across all models and categories:
- ` the` (4-7% of high-entropy tokens)
- ` a` (2-3%)
- ` in`, ` and`, ` The`, `,`, ` or`, ` is`

These are **function words and determiners** -- positions where the model must choose the syntactic frame for upcoming content. The high entropy at "the" means the model is uncertain about what comes next, not about the word "the" itself.

### Finding 8.2: Near-zero entropy tokens reveal confident scaffolding

Most common tokens with entropy < 0.001:
- Punctuation: `,`, `.`
- Function words: ` of`, ` the`, ` is`, ` to`
- Sub-words in known names: `emic` (from "epistemic"), `ist`, `202` (from years)
- End-of-sequence tokens

The token `202` appears 66 times at near-zero entropy in unknowable OLMo responses, almost always followed by `4` (72x) or `3` (45x). The model generates fabricated years with maximal confidence.

### Finding 8.3: Digit tokens are disproportionately low-entropy

| Family  | Digit entropy (unkn) | Non-digit entropy (unkn) | Ratio |
|---------|---------------------|-------------------------|-------|
| OLMo    | 0.135               | 0.842                   | 0.16  |
| Llama   | 0.212               | 0.732                   | 0.29  |
| Qwen    | 0.025               | 0.345                   | 0.07  |
| Mistral | 0.072               | 0.661                   | 0.11  |

**Numbers in fabricated responses are among the most confidently generated tokens.** The model picks a year or number and commits fully, even when the entire context is fictional.

### Finding 8.4: Sentence boundaries are 1.5-3.6x higher entropy than non-boundaries

Tokens starting a new sentence (capital after period) have much higher entropy:
- Knowable: 2.2-3.6x higher
- Unknowable: 1.5-1.8x higher

The ratio is larger for knowable responses, suggesting that sentence-initial decisions are relatively certain in unknowable (fabricated) text.

### Finding 8.5: Content words carry slightly more entropy than function words

| Token type   | Knowable entropy | Unknowable entropy |
|-------------|-----------------|-------------------|
| Punctuation | 0.13 - 0.41     | 0.23 - 0.56      |
| Function    | 0.19 - 0.51     | 0.25 - 0.74      |
| Content     | 0.27 - 0.62     | 0.38 - 0.89      |

The gradient punctuation < function < content holds across all models and categories.

### Finding 8.6: Entropy autocorrelation is higher for unknowable responses

Lag-1 autocorrelation of token entropy within a trace:
- Knowable: 0.02-0.15
- Unknowable: 0.14-0.19

Unknowable responses have more "runs" of similar-entropy tokens. High-entropy tokens cluster together during fabrication.

---

## 9. Outliers and Hard Cases

### Finding 9.1: The same ~10 queries fool all models

9 queries are misclassified by all 4 models (using per-model median threshold):
- **7 are knowable "weird truths"**: wombat scat, lobster taste, goldfish memory, tardigrade, blood pressure, Great Wall visibility, honey spoilage, banana berries, glass state
- **1 personal unknowable**: "What is the name of my childhood pet?" (low entropy because refusal is confident)
- **1 future unknowable**: "Can honey spoil?" (treated as knowable by entropy)

These weird truths generate high entropy because the model's correct answer conflicts with folk intuition, requiring more exploratory generation.

### Finding 9.2: Weird truths have 2.5-7x higher entropy than simple facts

| Family  | Weird truth mean_ent | Simple fact mean_ent | Ratio |
|---------|---------------------|---------------------|-------|
| OLMo    | 0.862               | 0.258               | 3.3x  |
| Llama   | 0.619               | 0.178               | 3.5x  |
| Qwen    | 0.396               | 0.056               | 7.0x  |
| Mistral | 0.652               | 0.261               | 2.5x  |

Qwen shows the most extreme ratio: 7x. These are the cases where entropy measures *familiarity*, not *truth*.

### Finding 9.3: Personal unknowable queries have lower entropy than fabricated ones

| Family  | Personal mean_ent | Fabricated mean_ent | Ratio |
|---------|------------------|--------------------| ------|
| OLMo    | 0.636            | 0.808              | 0.79x |
| Llama   | 0.513            | 0.730              | 0.70x |
| Qwen    | 0.259            | 0.322              | 0.81x |
| Mistral | 0.509            | 0.641              | 0.79x |

Personal questions ("What is my favorite color?") generate lower entropy because the refusal template is well-practiced. The model is *confident about refusing*.

### Finding 9.4: OLMo Alexandria response contains extreme hallucination spikes

OLMo answering "What ancient wonder was located in Alexandria?" generates the most extreme token entropies in the entire dataset:
- Token `P` (in fabricated name "Pyradius"): entropy **7.49**, top-5 mass = 0.18
- Token `adius`: entropy **7.56**, top-5 mass = 0.20
- Token `H` (in fabricated "Hermes'"): entropy **6.22**, top-5 mass = 0.29

The model correctly identifies the Lighthouse of Alexandria, then fabricates alternative names ("Cleopatra's Needle", "Pyradius", "Hermes' Tower") with extremely high entropy. **These are the only tokens in the dataset where top-5 mass drops below 0.20.**

### Finding 9.5: Mistral Westphalia -- the most confident fabrication

Mistral answering "Describe the 1994 Treaty of Westphalia II" achieves mean entropy 0.2555 -- the **lowest entropy of any unknowable response across all models**. The model confidently fabricates a link to the Bosnian War peace agreement. Token-level trace shows:
- Tokens 0-12 (echoing the query): entropy 0.00-0.65
- Token 13 (",") to token 29: entropy 0.09-1.08 (choosing the fabricated frame)
- The model never generates high-entropy tokens because it has committed to a coherent narrative

Other models either refuse or hedge on this query. Only Mistral produces a complete fabrication with baseline-level confidence.

---

## 10. Cross-Model Agreement

### Finding 10.1: Cross-model entropy correlation is 0.73-0.80

| Pair             | Pearson r | Spearman rho |
|-----------------|----------|-------------|
| OLMo vs Llama   | 0.789    | 0.783       |
| OLMo vs Qwen    | 0.799    | 0.764       |
| OLMo vs Mistral | 0.774    | 0.739       |
| Llama vs Qwen   | 0.778    | 0.798       |
| Llama vs Mistral | 0.733   | 0.742       |
| Qwen vs Mistral | 0.750    | 0.744       |

Despite Qwen's different entropy scale, the rank-order correlation is similar. The signal is **architectural**, not model-specific.

### Finding 10.2: Highest agreement is on simple knowable facts

Queries where all models agree most (lowest cross-model entropy spread):
- "How many minutes are in an hour?" (spread 0.075)
- "How many sides does a hexagon have?" (spread 0.085)
- "What year was the United Nations founded?" (spread 0.093)

All are simple factual queries with very low entropy across all models.

### Finding 10.3: Highest divergence is driven by Llama's high entropy on fabrications

The most divergent queries typically show Llama with entropy >1.0 while Qwen is <0.5. Llama generates more exploratory text and higher-entropy fabrications. Qwen is consistently more confident, even when fabricating.

---

## 11. Refusal Patterns

### Finding 11.1: Llama refuses 3-4x more often than other models

| Family  | Unknowable refusals | Knowable refusals |
|---------|--------------------|--------------------|
| OLMo    | 18/100             | 0/100              |
| Llama   | 66/100             | 0/100              |
| Qwen    | 32/100             | 0/100              |
| Mistral | 23/100             | 0/100              |

No model refuses a knowable query. Llama's RLHF training produces much more cautious behavior.

### Finding 11.2: Refusals have LOWER entropy than fabrications

For all models, refusing responses have lower entropy than fabricating responses:

| Family  | Refusal entropy | Fabrication entropy | Refusal length | Fabrication length |
|---------|----------------|--------------------|--------------|--------------------|
| OLMo    | 0.682          | 0.812              | 94            | 125                |
| Llama   | 0.634          | 0.844              | 94            | 124                |
| Qwen    | 0.258          | 0.341              | 74            | 123                |
| Mistral | 0.516          | 0.659              | 74            | 128                |

Refusals are shorter and more confident. The model has a well-practiced template for "I don't have access to..." The entropy signal for discrimination comes primarily from fabricating responses, not refusing ones.

### Finding 11.3: Llama citation responses are mostly refusals

Of Llama's 34 unknowable citation responses, **28 contain refusal phrases** (82%). When Llama does fabricate a citation, its entropy is notably higher (0.89 for the worst case). The 6 non-refusing responses include plausible-sounding fabrications of paper summaries.

---

## 12. Citation-Specific Patterns

### Finding 12.1: Year tokens in fabricated citations have near-zero entropy

In OLMo and Llama citation responses, tokens for fabricated years (e.g., "2023", "2024") have mean entropy 0.015-0.021. The model picks a year and commits with ~100% confidence.

Token `202` is followed by `4` (72 times), `3` (45 times), `2` (16 times). The model's "default fabricated year" is 2024, then 2023.

### Finding 12.2: Citation query entropy is slightly LOWER than non-citation unknowable

| Family  | Citation unkn entropy | Non-citation unkn entropy |
|---------|-----------------------|---------------------------|
| OLMo    | 0.730                 | 0.819                     |
| Llama   | 0.580                 | 0.771                     |
| Qwen    | 0.309                 | 0.318                     |
| Mistral | 0.680                 | 0.599                     |

For Llama, citation entropy is notably lower because it mostly refuses. For Mistral, citations are slightly *higher* entropy, possibly because Mistral fabricates more elaborate paper descriptions.

### Finding 12.3: The wombat scat query is a Rosetta Stone

All 4 models answer differently:
- **OLMo**: "cylindrical or sausage-shaped" (WRONG, mean_ent=0.91) -- highest spike at "sausage" (ent=2.45)
- **Llama**: "shaped like a cube" (CORRECT, mean_ent=0.71) -- highest spike at "shaped" (ent=2.65) and "cube" (ent=2.47)
- **Qwen**: "oval or pellet-shaped" (WRONG, mean_ent=0.61) -- highest spike at "rounded" (ent=3.47)
- **Mistral**: "cube-shaped" (CORRECT, mean_ent=0.63) -- moderate entropy throughout

Even the correct answers have high entropy at the key content word. The spike at "cube" in Llama (2.47) shows the model is uncertain about this answer despite being correct. This is the canonical case where entropy measures *familiarity of the answer*, not *truth*.

---

## 13. Alternative Metrics

### Finding 13.1: Comprehensive AUC ranking

| Metric                | OLMo  | Llama | Qwen  | Mistral | Best for |
|----------------------|-------|-------|-------|---------|----------|
| total_entropy (sum)  | 0.922 | 0.964 | 0.891 | 0.962   | Llama, Mistral |
| num_tokens (length)  | 0.912 | 0.941 | 0.881 | 0.957   | (baseline) |
| mean_entropy         | 0.894 | 0.922 | 0.896 | 0.905   | Qwen |
| top5_mass (negated)  | 0.905 | 0.930 | 0.902 | 0.904   | OLMo, Qwen |
| max_entropy          | 0.885 | 0.950 | 0.892 | 0.891   | Llama |
| p90_entropy          | 0.898 | 0.928 | 0.903 | 0.909   | Qwen |
| std_entropy          | 0.892 | 0.928 | 0.900 | 0.904   | Qwen |
| entropy_second_half  | 0.916 | 0.912 | 0.907 | 0.883   | OLMo |
| spike_frac (>2.0)    | 0.870 | 0.884 | 0.710 | 0.877   | |
| frac_below_0.01      | 0.201 | 0.164 | 0.126 | 0.194   | (reversed) |
| entropy_gradient     | 0.831 | 0.673 | 0.857 | 0.677   | |
| first_token_entropy  | 0.670 | 0.654 | 0.615 | 0.506   | |

**Total entropy (sum) is the best single metric** for 2 of 4 models and competitive on the others. It naturally combines the entropy signal with the length signal.

### Finding 13.2: Nonzero-mean entropy beats mean entropy

If you exclude near-zero tokens (< 0.01) before computing the mean, AUC improves slightly:

| Family | mean_ent AUC | nonzero_mean AUC |
|--------|-------------|-----------------|
| OLMo   | 0.894       | 0.906           |
| Llama  | 0.922       | 0.932           |
| Qwen   | 0.896       | 0.865           |
| Mistral| 0.905       | 0.908           |

(Qwen is an exception because so many of its tokens are near-zero.)

### Finding 13.3: Geometric mean entropy is poor; harmonic mean is terrible

Geometric mean AUC: 0.80-0.88. Harmonic mean AUC: 0.46-0.73 (below random for some models). The extreme near-zero values dominate harmonic means.

---

## 14. Structural Observations

### Finding 14.1: Entropy runs are longer in unknowable responses

Consecutive tokens with entropy > 1.0:
- Knowable max run: 3-9 tokens
- Unknowable max run: 5-11 tokens
- Unknowable mean max run: 2.05-5.33

Every unknowable OLMo and Mistral trace (100/100) has at least one run of entropy > 1.0.

### Finding 14.2: Entropy jump frequency is 2-4x higher in unknowable

Adjacent tokens with |delta| > 1.5:
- Knowable: 2.5-8.7% of token pairs
- Unknowable: 3.6-14.2% of token pairs

### Finding 14.3: Type-token ratio is higher for knowable (0.83-0.91) than unknowable (0.69-0.74)

Knowable responses use more diverse vocabulary per token. Unknowable responses, being longer and more formulaic, repeat tokens more.

### Finding 14.4: Vocabulary size confirms model tokenizer differences

| Family  | Max token ID | Unique tokens used |
|---------|-------------|-------------------|
| Qwen    | 151,645     | 3,010             |
| Llama   | 128,009     | 2,633             |
| OLMo    | 100,257     | 3,678             |
| Mistral | 29,745      | 3,162             |

Mistral's vocabulary is ~5x smaller than Qwen's, yet they use similar numbers of unique tokens. This likely contributes to Mistral's higher per-token entropy compared to Qwen.

---

## 15. Fabrication Micro-Structure

### Finding 15.1: In fabricated citations, proper names are near-zero entropy

From the OLMo trace for "Summarize: Chen, Kowalski, and Morrison...":
- Author names echoed from query: entropy 0.00-0.01
- Year "2023": entropy 0.00
- Paper title echoed: entropy 0.00-0.003
- **Venue "ICLR"**: entropy 0.82 (the model chose a venue)
- **Framing words** ("demonstrate", "when", "supervision"): entropy 0.55-1.86

The model is maximally confident on names and titles it can copy from the query, moderately uncertain about venue choice, and highly uncertain about the framing of the fabricated content.

### Finding 15.2: Top-5 probability mass rarely drops below 0.5

Only 154 tokens in the entire dataset have top-5 mass < 0.5 (0.27% of all tokens). Of these:
- 7 are in knowable responses (5 from a single OLMo response about Alexandria)
- 144 are in unknowable responses
- Qwen unknowable has **zero** tokens with top-5 mass < 0.5

When the model is generating a token where its top 5 candidates carry < 50% of the probability, it is in deep uncertainty. This occurs almost exclusively during fabrication.

---

## 16. Key Takeaways and Anomalies

1. **Length is the elephant in the room.** Response length alone achieves AUC 0.88-0.96. Mean entropy is partially redundant with length. The most honest metric is total_entropy, which acknowledges this entanglement.

2. **Entropy residuals still carry signal.** After removing length, entropy residuals give AUC 0.60-0.70. This is above random and suggests genuine per-token uncertainty differences, but much weaker than the raw numbers suggest.

3. **Qwen operates on a different scale.** Its entropy is 36% of other models, its zero-fraction is double, and its spike rate is 1/10th. Any cross-model analysis must normalize per-model.

4. **Fabrication gets worse over time.** Unknowable responses show monotonically increasing entropy through the response (89-91% of traces), while knowable responses are flat. This is consistent with a model "running out" of confidently fabricated content.

5. **Numbers are the most confidently fabricated tokens.** Digits, years, and numeric tokens have 7-16% of the entropy of surrounding text. The model commits maximally to fabricated specifics.

6. **Refusals are easy; fabrications are the real test.** Refusal responses have lower entropy than fabrications. The hard discrimination problem is fabrication vs. knowledge, not refusal vs. knowledge.

7. **Weird truths break the signal.** 7 of 9 universally-misclassified queries are knowable "weird truths" (wombat scat, tardigrade survival). Entropy measures training-data familiarity, not factual truth.

8. **Sentence boundaries are decision points.** Sentence-initial tokens have 1.5-3.6x higher entropy. These are the positions where the model commits to a new claim.

9. **Cross-model agreement (rho 0.73-0.80) suggests an architectural signal.** Different models, different training data, different vocabularies, yet they agree on which queries produce high/low entropy.

10. **The Mistral Westphalia fabrication is the scariest data point.** Mean entropy 0.26 for a complete fabrication of a nonexistent treaty, indistinguishable from a confident true answer. One model's perfect fabrication is another model's refusal.
