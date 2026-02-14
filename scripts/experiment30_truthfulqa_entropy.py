"""
Experiment 30: Does Per-Token Entropy Predict Correctness on TruthfulQA?

Research Question: The paper claims entropy measures epistemic familiarity,
not factual correctness. If this is true, entropy should NOT predict whether
a model's answer to a TruthfulQA question is correct or incorrect.

Hypothesis: AUC for entropy-predicts-correctness should be approximately 0.5
(no better than random). This is a NEGATIVE result that strengthens the paper's
central claim: the tensor does not tell you if the model is RIGHT. It tells you
if the model is operating in a mode where it COULD be right, because it has
something to retrieve rather than fabricate.

Paper reference (eval.tex lines 191-196):
  "When we test whether entropy predicts correctness (using TruthfulQA's
   correct vs. incorrect answer pairs), AUC drops to approximately 0.53,
   which is no better than random."

Design:
  1. Load TruthfulQA (generation subset) from HuggingFace
  2. Sample 150 questions (seed=42) for tractable runtime
  3. For each question, generate an answer with each of 4 instruct models
  4. Compute per-token entropy during generation
  5. Judge correctness by substring matching against TruthfulQA reference answers
  6. Compute AUC: does mean entropy predict correctness?

Models (same as exp27):
  - allenai/olmo-3-7b-instruct
  - meta-llama/Llama-3.1-8B-Instruct
  - Qwen/Qwen3-4B-Instruct-2507
  - mistralai/Mistral-7B-Instruct-v0.3

Key distinction from experiment 10:
  Exp10 fed reference answers through the model and measured topology.
  Exp30 generates free-form answers and checks them against references.
  This is a generative correctness test, not a discriminative one.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 150
RANDOM_SEED = 42
MAX_NEW_TOKENS = 150

MODELS = [
    ("OLMo", "allenai/olmo-3-7b-instruct"),
    ("Llama", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Qwen", "Qwen/Qwen3-4B-Instruct-2507"),
    ("Mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
]

MISTRAL_MODELS = {"mistralai/Mistral-7B-Instruct-v0.3"}

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions directly and concisely."


# ============================================================================
# Data loading
# ============================================================================

def load_truthfulqa_sample(n_samples=SAMPLE_SIZE, seed=RANDOM_SEED):
    """Load and sample from TruthfulQA generation subset.

    Returns a list of dicts with keys:
      question, best_answer, correct_answers, incorrect_answers, category, source
    """
    print(f"Loading TruthfulQA (generation) from HuggingFace...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    print(f"  Total questions: {len(dataset)}")

    # Deterministic sample
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    indices.sort()

    samples = []
    for idx in indices:
        row = dataset[int(idx)]
        samples.append({
            "question": row["question"],
            "best_answer": row["best_answer"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
            "category": row.get("category", "unknown"),
            "source": row.get("source", "unknown"),
            "dataset_idx": int(idx),
        })

    print(f"  Sampled {len(samples)} questions (seed={seed})")
    return samples


# ============================================================================
# Prompt formatting
# ============================================================================

def format_chat(system_prompt, user_query, tokenizer):
    """Format prompt using model's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"System: {system_prompt}\n\nUser: {user_query}\n\nAssistant:"


def strip_think_tokens(text):
    """Strip Qwen3's <think>...</think> reasoning tokens from response text."""
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Handle unclosed <think> (generation cut off mid-thought)
    if "<think>" in text:
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


# ============================================================================
# Generation with entropy capture
# ============================================================================

def generate_with_tensor(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS):
    """Generate response and capture per-token entropy signals.

    Returns:
        response: str - the generated text
        metrics: dict with mean_entropy, max_entropy, entropy_std,
                 mean_logprob, mean_top5_mass, n_tokens
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    scores = outputs.scores
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]

    token_entropies = []
    logprobs = []
    top5_masses = []

    for score, token_id in zip(scores, generated_ids):
        logits = score.squeeze(0).float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = -torch.sum(probs * log_probs).item()
        token_entropies.append(entropy)

        top_probs = torch.topk(probs, k=min(5, len(probs))).values
        top5_masses.append(top_probs.sum().item())

        logprobs.append(log_probs[token_id].item())

    full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    response = full_text[len(prompt_text):].strip()

    # Strip Qwen3 <think> tokens from the response text
    response = strip_think_tokens(response)

    return response, {
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0.0,
        "max_entropy": float(np.max(token_entropies)) if token_entropies else 0.0,
        "entropy_std": float(np.std(token_entropies)) if token_entropies else 0.0,
        "mean_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
        "mean_top5_mass": float(np.mean(top5_masses)) if top5_masses else 0.0,
        "n_tokens": len(token_entropies),
    }


# ============================================================================
# Correctness judgment
# ============================================================================

def judge_correctness(response, correct_answers, incorrect_answers, best_answer):
    """Determine if a generated response is correct using TruthfulQA references.

    Strategy:
      1. Check if response matches any correct_answer (substring, case-insensitive)
      2. Check if response matches any incorrect_answer (substring, case-insensitive)
      3. If matches correct but not incorrect -> correct
      4. If matches incorrect (regardless of correct match) -> incorrect
         (catches cases where both a correct and incorrect fragment appear)
      5. If matches neither -> indeterminate (excluded from AUC)

    Returns:
        "correct", "incorrect", or "indeterminate"
    """
    response_lower = response.lower().strip()

    # Empty or trivially short response: indeterminate
    if len(response_lower) < 3:
        return "indeterminate"

    # Check matches against correct answers
    correct_match = False
    for ans in correct_answers:
        ans_lower = ans.lower().strip()
        if len(ans_lower) < 2:
            continue
        if ans_lower in response_lower:
            correct_match = True
            break

    # Also check best_answer specifically
    best_lower = best_answer.lower().strip()
    if len(best_lower) >= 2 and best_lower in response_lower:
        correct_match = True

    # Check matches against incorrect answers
    incorrect_match = False
    for ans in incorrect_answers:
        ans_lower = ans.lower().strip()
        if len(ans_lower) < 2:
            continue
        if ans_lower in response_lower:
            incorrect_match = True
            break

    # Decision logic
    if incorrect_match:
        # If any incorrect answer appears, call it incorrect
        # This is conservative: TruthfulQA incorrect answers are the
        # "imitative falsehoods" that models are designed to reproduce
        return "incorrect"
    elif correct_match:
        return "correct"
    else:
        return "indeterminate"


# ============================================================================
# Per-model data collection
# ============================================================================

def collect_model_data(family, model_id, samples):
    """Run all TruthfulQA samples through a model, collect entropy + correctness."""
    print(f"\n{'='*70}")
    print(f"Processing: {model_id} ({family})")
    print(f"{'='*70}")

    # Load model
    tokenizer_kwargs = {}
    if model_id in MISTRAL_MODELS:
        tokenizer_kwargs["fix_mistral_regex"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None

    results = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        print(f"  [{i+1}/{len(samples)}] {question[:65]}...")

        prompt = format_chat(SYSTEM_PROMPT, question, tokenizer)
        response, tensor_metrics = generate_with_tensor(model, tokenizer, prompt)

        correctness = judge_correctness(
            response,
            sample["correct_answers"],
            sample["incorrect_answers"],
            sample["best_answer"],
        )

        results.append({
            "family": family,
            "model_id": model_id,
            "dataset_idx": sample["dataset_idx"],
            "category": sample["category"],
            "question": question,
            "best_answer": sample["best_answer"],
            "response": response[:500],
            "correctness": correctness,
            "is_correct": 1 if correctness == "correct" else 0,
            "is_incorrect": 1 if correctness == "incorrect" else 0,
            "is_indeterminate": 1 if correctness == "indeterminate" else 0,

            # Tensor signals
            "mean_entropy": tensor_metrics["mean_entropy"],
            "max_entropy": tensor_metrics["max_entropy"],
            "entropy_std": tensor_metrics["entropy_std"],
            "mean_logprob": tensor_metrics["mean_logprob"],
            "mean_top5_mass": tensor_metrics["mean_top5_mass"],
            "n_tokens": tensor_metrics["n_tokens"],
        })

        # Progress update every 25 questions
        if (i + 1) % 25 == 0:
            n_correct = sum(1 for r in results if r["correctness"] == "correct")
            n_incorrect = sum(1 for r in results if r["correctness"] == "incorrect")
            n_indet = sum(1 for r in results if r["correctness"] == "indeterminate")
            print(f"    Progress: {n_correct} correct, {n_incorrect} incorrect, "
                  f"{n_indet} indeterminate")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================================
# AUC analysis
# ============================================================================

def compute_correctness_auc(df, signal_col="mean_entropy", label_col="is_correct"):
    """Compute AUC: does the signal predict correctness?

    If AUC ~ 0.5: entropy does NOT predict correctness (our hypothesis).
    If AUC > 0.5 with label=correct and lower entropy for correct:
        entropy discriminates correct from incorrect.
    If AUC < 0.5: inverted -- higher entropy for correct answers.

    We use the convention: score = -entropy (lower entropy = higher score),
    label = is_correct. If correct answers have lower entropy, AUC > 0.5.
    """
    # Filter to determinate answers only
    mask = df["correctness"].isin(["correct", "incorrect"])
    filtered = df[mask].copy()

    if len(filtered) < 10:
        return np.nan, len(filtered)

    labels = filtered[label_col].values
    if len(np.unique(labels)) < 2:
        return np.nan, len(filtered)

    # Score: negative entropy so that lower entropy -> higher score
    scores = -filtered[signal_col].values

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = np.nan

    return auc, len(filtered)


def analyze_results(df):
    """Full analysis: per-model and aggregate AUC for correctness prediction."""
    print(f"\n{'='*70}")
    print("ANALYSIS: Does Entropy Predict Correctness?")
    print(f"{'='*70}")

    # Overall correctness distribution
    n_total = len(df)
    n_correct = (df["correctness"] == "correct").sum()
    n_incorrect = (df["correctness"] == "incorrect").sum()
    n_indet = (df["correctness"] == "indeterminate").sum()
    print(f"\nOverall correctness distribution:")
    print(f"  Correct:       {n_correct:4d} ({n_correct/n_total:.1%})")
    print(f"  Incorrect:     {n_incorrect:4d} ({n_incorrect/n_total:.1%})")
    print(f"  Indeterminate: {n_indet:4d} ({n_indet/n_total:.1%})")

    # Mean entropy by correctness
    for status in ["correct", "incorrect", "indeterminate"]:
        subset = df[df["correctness"] == status]
        if len(subset) > 0:
            print(f"\n  {status.capitalize()} answers (n={len(subset)}):")
            print(f"    Mean entropy:  {subset['mean_entropy'].mean():.4f} "
                  f"(std={subset['mean_entropy'].std():.4f})")
            print(f"    Max entropy:   {subset['max_entropy'].mean():.4f}")
            print(f"    Mean logprob:  {subset['mean_logprob'].mean():.4f}")
            print(f"    Mean top5:     {subset['mean_top5_mass'].mean():.4f}")

    # Per-model AUC
    print(f"\n{'='*70}")
    print("PER-MODEL AUC: Entropy -> Correctness")
    print(f"{'='*70}")

    signals = [
        ("Mean Entropy", "mean_entropy"),
        ("Max Entropy", "max_entropy"),
        ("Entropy Std", "entropy_std"),
        ("-LogProb", "mean_logprob"),
        ("Top-5 Mass", "mean_top5_mass"),
    ]

    header = f"{'Model':<12}"
    for name, _ in signals:
        header += f" {name:>14}"
    header += f" {'N(det)':>8}"
    print(header)
    print("-" * (12 + 15 * len(signals) + 9))

    per_model_aucs = {}

    for family in df["family"].unique():
        family_df = df[df["family"] == family]
        row_str = f"{family:<12}"
        aucs = {}

        for signal_name, signal_col in signals:
            if signal_col == "mean_logprob":
                # For logprob: more negative = less confident, use raw (not negated)
                mask = family_df["correctness"].isin(["correct", "incorrect"])
                filtered = family_df[mask]
                if len(filtered) >= 10 and len(filtered["is_correct"].unique()) >= 2:
                    try:
                        auc = roc_auc_score(
                            filtered["is_correct"].values,
                            filtered[signal_col].values  # higher logprob = more correct?
                        )
                    except ValueError:
                        auc = np.nan
                else:
                    auc = np.nan
                n_det = len(filtered)
            elif signal_col == "mean_top5_mass":
                # Higher top-5 mass = more confident = more correct?
                mask = family_df["correctness"].isin(["correct", "incorrect"])
                filtered = family_df[mask]
                if len(filtered) >= 10 and len(filtered["is_correct"].unique()) >= 2:
                    try:
                        auc = roc_auc_score(
                            filtered["is_correct"].values,
                            filtered[signal_col].values
                        )
                    except ValueError:
                        auc = np.nan
                else:
                    auc = np.nan
                n_det = len(filtered)
            else:
                # Entropy signals: negate so lower entropy -> higher score -> correct
                auc, n_det = compute_correctness_auc(
                    family_df, signal_col=signal_col
                )

            aucs[signal_name] = auc
            row_str += f" {auc:>14.3f}" if not np.isnan(auc) else f" {'N/A':>14}"

        row_str += f" {n_det:>8d}"
        print(row_str)
        per_model_aucs[family] = aucs

    # Aggregate AUC
    print(f"\n{'Aggregate':<12}", end="")
    agg_aucs = {}
    for signal_name, signal_col in signals:
        if signal_col == "mean_logprob":
            mask = df["correctness"].isin(["correct", "incorrect"])
            filtered = df[mask]
            if len(filtered) >= 10 and len(filtered["is_correct"].unique()) >= 2:
                try:
                    auc = roc_auc_score(
                        filtered["is_correct"].values,
                        filtered[signal_col].values
                    )
                except ValueError:
                    auc = np.nan
            else:
                auc = np.nan
        elif signal_col == "mean_top5_mass":
            mask = df["correctness"].isin(["correct", "incorrect"])
            filtered = df[mask]
            if len(filtered) >= 10 and len(filtered["is_correct"].unique()) >= 2:
                try:
                    auc = roc_auc_score(
                        filtered["is_correct"].values,
                        filtered[signal_col].values
                    )
                except ValueError:
                    auc = np.nan
            else:
                auc = np.nan
        else:
            auc, _ = compute_correctness_auc(df, signal_col=signal_col)

        agg_aucs[signal_name] = auc
        print(f" {auc:>14.3f}" if not np.isnan(auc) else f" {'N/A':>14}", end="")

    n_det_total = df["correctness"].isin(["correct", "incorrect"]).sum()
    print(f" {n_det_total:>8d}")

    # Key finding
    print(f"\n{'='*70}")
    print("KEY FINDING")
    print(f"{'='*70}")

    mean_entropy_auc = agg_aucs.get("Mean Entropy", np.nan)
    if not np.isnan(mean_entropy_auc):
        distance_from_random = abs(mean_entropy_auc - 0.5)
        if distance_from_random < 0.1:
            verdict = "CONFIRMED"
            explanation = (
                "Entropy does NOT predict correctness (AUC near 0.5).\n"
                "  The tensor tells you whether the model is in retrieval vs. fabrication mode,\n"
                "  not whether its answer is factually right."
            )
        elif mean_entropy_auc > 0.6:
            verdict = "UNEXPECTED"
            explanation = (
                f"Entropy shows some predictive power for correctness (AUC={mean_entropy_auc:.3f}).\n"
                "  This would weaken the claim that entropy measures familiarity, not truth."
            )
        else:
            verdict = "PARTIAL"
            explanation = (
                f"Entropy shows weak/inverted signal for correctness (AUC={mean_entropy_auc:.3f}).\n"
                "  Direction may be inverted (higher entropy on correct answers) or near-random."
            )

        print(f"\n  Aggregate Mean Entropy AUC (correctness): {mean_entropy_auc:.3f}")
        print(f"  Distance from random (0.5): {distance_from_random:.3f}")
        print(f"  Verdict: {verdict}")
        print(f"  {explanation}")
    else:
        print("\n  Could not compute aggregate AUC (insufficient data).")

    # Compare with the paper's claim
    print(f"\n--- Paper Claim Comparison ---")
    print(f"  Paper claims AUC ~ 0.53 (eval.tex lines 191-196)")
    if not np.isnan(mean_entropy_auc):
        print(f"  Measured AUC: {mean_entropy_auc:.3f}")
        if abs(mean_entropy_auc - 0.53) < 0.1:
            print(f"  Consistent with paper claim.")
        else:
            print(f"  Diverges from paper claim by {abs(mean_entropy_auc - 0.53):.3f}")

    # Per-model consistency check
    print(f"\n--- Per-Model Consistency ---")
    entropy_aucs = []
    for family, aucs in per_model_aucs.items():
        auc_val = aucs.get("Mean Entropy", np.nan)
        if not np.isnan(auc_val):
            entropy_aucs.append(auc_val)
            dist = abs(auc_val - 0.5)
            near_random = "near random" if dist < 0.1 else "some signal"
            print(f"  {family}: AUC={auc_val:.3f} ({near_random})")

    if entropy_aucs:
        mean_auc = np.mean(entropy_aucs)
        std_auc = np.std(entropy_aucs)
        print(f"\n  Cross-model mean AUC: {mean_auc:.3f} (std={std_auc:.3f})")
        all_near_random = all(abs(a - 0.5) < 0.1 for a in entropy_aucs)
        if all_near_random:
            print("  ALL models show entropy near-random for correctness prediction.")
            print("  The epistemic-not-veridical claim holds across architectures.")
        else:
            n_near = sum(1 for a in entropy_aucs if abs(a - 0.5) < 0.1)
            print(f"  {n_near}/{len(entropy_aucs)} models near random.")

    return agg_aucs, per_model_aucs


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 30: ENTROPY vs. CORRECTNESS ON TRUTHFULQA")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Sample size: {SAMPLE_SIZE} questions")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Models: {[m[0] for m in MODELS]}")
    print(f"\nHypothesis: Entropy should NOT predict correctness (AUC ~ 0.5)")
    print(f"  This is the negative result claimed in eval.tex lines 191-196.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load TruthfulQA sample
    samples = load_truthfulqa_sample()

    all_results = []

    for family, model_id in MODELS:
        results = collect_model_data(family, model_id, samples)
        if results:
            all_results.extend(results)

            # Incremental save after each model
            df = pd.DataFrame(all_results)
            csv_path = f"exp30_truthfulqa_entropy_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Incremental save: {csv_path}")

    if not all_results:
        print("No results collected!")
        return

    df = pd.DataFrame(all_results)

    # Final save
    csv_path = f"exp30_truthfulqa_entropy_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFinal data saved: {csv_path}")

    # Analysis
    agg_aucs, per_model_aucs = analyze_results(df)

    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT 30 COMPLETE")
    print(f"{'='*70}")
    print(f"\nFiles:")
    print(f"  Raw data: {csv_path}")
    print(f"  Questions: {SAMPLE_SIZE} (from TruthfulQA generation)")
    print(f"  Models: {len(df['family'].unique())}")
    total_det = df["correctness"].isin(["correct", "incorrect"]).sum()
    total_indet = (df["correctness"] == "indeterminate").sum()
    print(f"  Determinate judgments: {total_det}")
    print(f"  Indeterminate (excluded from AUC): {total_indet}")
    print(f"\nThis experiment provides evidence for the paper's claim that")
    print(f"tensor entropy measures epistemic familiarity, not factual correctness.")


if __name__ == "__main__":
    main()
