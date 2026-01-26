"""
Experiment 21: Baseline Comparison

Research Question: Does the tensor interface beat simpler alternatives?

Baselines to compare:
1. Self-reported confidence ("How sure are you?")
2. Output length / hedging language detection
3. Simple entropy threshold
4. Perplexity alone

The tensor interface extracts entropy, attention patterns, and distribution
statistics. If simpler methods work equally well, the tensor is unnecessary.

Goal: Show tensor signals outperform these baselines for epistemic detection.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import gc
import re
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use OLMo-3 as primary model (matches paper methodology)
MODEL_ID = "allenai/olmo-3-7b-instruct"

# Query categories with known ground truth status
# For each query, we know whether the model SHOULD be able to answer truthfully
QUERIES = {
    # Queries with verifiable ground truth (model should know these)
    "knowable": [
        ("What is the capital of France?", "Paris"),
        ("What year did World War II end?", "1945"),
        ("What is the chemical formula for water?", "H2O"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the speed of light approximately?", "300,000 km/s"),
        ("What is 2 + 2?", "4"),
        ("What continent is Brazil in?", "South America"),
        ("How many days are in a week?", "7"),
    ],
    # Queries requiring fabrication (no ground truth exists)
    "unknowable": [
        ("Summarize Dr. Yuki Tanaka's 2023 paper on Recursive Self-Modeling.", None),
        ("What is the Brennan-Kowalski theorem in information theory?", None),
        ("Describe the 1994 Treaty of Westphalia II.", None),
        ("What were the outcomes of the 2021 Kyoto Protocol II?", None),
        ("Summarize the Lindquist Effect in transformer attention.", None),
        ("What is the Okonkwo-Patel framework for model uncertainty?", None),
        ("Describe the Marchetti Protocol for neural network pruning.", None),
        ("What were the findings of the 2022 Helsinki AI Safety Accord?", None),
        ("Summarize Prof. Elena Vasquez's book 'Minds Without Brains'.", None),
        ("What is the Fibonacci Resonance Hypothesis in cognitive science?", None),
    ],
}

# Hedging language markers
HEDGE_MARKERS = [
    "i think", "i believe", "possibly", "might be", "could be",
    "perhaps", "probably", "likely", "may be", "as far as i know",
    "to my knowledge", "i'm not certain", "i'm not sure", "approximately",
    "roughly", "if i recall", "it seems", "apparently", "allegedly",
]

# Uncertainty/refusal markers
REFUSAL_MARKERS = [
    "i don't have", "i cannot", "i'm not able", "no such", "doesn't exist",
    "not aware", "cannot find", "no record", "unable to", "fictional",
    "made up", "not real", "i apologize", "don't have information",
    "cannot provide", "i'm sorry", "not familiar", "i don't know",
    "cannot verify", "no evidence", "i couldn't find", "there is no",
]

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions directly and concisely."


def format_chat(system_prompt, user_query, tokenizer):
    """Format prompt using model's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {system_prompt}\n\nUser: {user_query}\n\nAssistant:"


def generate_with_metrics(model, tokenizer, prompt, max_tokens=200):
    """Generate response and capture tensor signals."""
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

    # Per-token metrics
    token_entropies = []
    top5_masses = []
    logprobs = []

    for score, token_id in zip(scores, generated_ids):
        logits = score.squeeze(0).float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = -torch.sum(probs * log_probs).item()
        token_entropies.append(entropy)

        top_probs = torch.topk(probs, k=min(5, len(probs))).values
        top5_masses.append(top_probs.sum().item())

        logprobs.append(log_probs[token_id].item())

    # Decode response
    full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    response = full_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()

    tensor_metrics = {
        "mean_entropy": np.mean(token_entropies) if token_entropies else 0,
        "max_entropy": np.max(token_entropies) if token_entropies else 0,
        "entropy_std": np.std(token_entropies) if token_entropies else 0,
        "mean_logprob": np.mean(logprobs) if logprobs else 0,
        "mean_top5_mass": np.mean(top5_masses) if top5_masses else 0,
    }

    return response, tensor_metrics


def get_self_reported_confidence(model, tokenizer, query, response):
    """
    Baseline 1: Ask the model how confident it is in its answer.
    Returns a normalized confidence score (0-1).
    """
    followup_prompt = format_chat(
        "You are a helpful assistant.",
        f"You just answered: '{response}' to the question '{query}'. "
        f"On a scale of 0-100, how confident are you in that answer? "
        f"Reply with ONLY a number.",
        tokenizer
    )

    inputs = tokenizer(followup_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    confidence_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    confidence_text = confidence_text.split(":")[-1].strip()

    # Extract number
    numbers = re.findall(r'\d+', confidence_text)
    if numbers:
        conf = min(100, max(0, int(numbers[0]))) / 100.0
        return conf
    return 0.5  # Default to uncertain


def compute_hedging_score(response):
    """
    Baseline 2: Count hedging language markers.
    More hedging = lower confidence.
    """
    response_lower = response.lower()
    hedge_count = sum(1 for marker in HEDGE_MARKERS if marker in response_lower)
    refusal_count = sum(1 for marker in REFUSAL_MARKERS if marker in response_lower)

    # Normalize: more markers = higher uncertainty score
    # Scale by response length to avoid length bias
    word_count = max(1, len(response.split()))
    hedge_density = (hedge_count + refusal_count * 2) / word_count

    return min(1.0, hedge_density * 10)  # Scale and cap


def compute_length_score(response):
    """
    Baseline 2b: Response length heuristic.
    Very short or very long responses may indicate uncertainty.
    """
    word_count = len(response.split())

    # Normalize: optimal length around 20-50 words
    # Very short (< 5) or very long (> 100) get higher uncertainty
    if word_count < 5:
        return 0.8  # Very short = possibly refusing
    elif word_count < 10:
        return 0.5
    elif word_count <= 50:
        return 0.2  # Good confident length
    elif word_count <= 100:
        return 0.4
    else:
        return 0.6  # Too long = possibly padding


def run_baseline_comparison():
    """Main experiment: compare tensor vs baselines."""
    print("=" * 70)
    print("EXPERIMENT 21: BASELINE COMPARISON")
    print("=" * 70)
    print(f"\nModel: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Knowable queries: {len(QUERIES['knowable'])}")
    print(f"Unknowable queries: {len(QUERIES['unknowable'])}")

    print(f"\n--- Loading Model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    results = []

    for category, queries in QUERIES.items():
        print(f"\n--- Processing {category} queries ---")
        label = 0 if category == "knowable" else 1  # 1 = should be uncertain

        for query, expected_answer in queries:
            print(f"  Query: {query[:50]}...")

            prompt = format_chat(SYSTEM_PROMPT, query, tokenizer)
            response, tensor_metrics = generate_with_metrics(model, tokenizer, prompt)

            # Baseline 1: Self-reported confidence
            self_conf = get_self_reported_confidence(model, tokenizer, query, response)

            # Baseline 2: Hedging/length heuristics
            hedge_score = compute_hedging_score(response)
            length_score = compute_length_score(response)

            # Combined heuristic baseline
            heuristic_score = (hedge_score + length_score) / 2

            result = {
                "category": category,
                "label": label,  # 0=knowable, 1=unknowable
                "query": query,
                "response": response,
                "response_length": len(response.split()),

                # Tensor interface signals
                "tensor_entropy": tensor_metrics["mean_entropy"],
                "tensor_max_entropy": tensor_metrics["max_entropy"],
                "tensor_entropy_std": tensor_metrics["entropy_std"],
                "tensor_logprob": tensor_metrics["mean_logprob"],
                "tensor_top5": tensor_metrics["mean_top5_mass"],

                # Baseline signals
                "baseline_self_report": 1 - self_conf,  # Invert: low conf = high uncertainty
                "baseline_hedge": hedge_score,
                "baseline_length": length_score,
                "baseline_heuristic": heuristic_score,
            }
            results.append(result)

            print(f"    Response length: {result['response_length']} words")
            print(f"    Tensor entropy: {tensor_metrics['mean_entropy']:.3f}")
            print(f"    Self-reported conf: {self_conf:.2f}")
            print(f"    Hedge score: {hedge_score:.2f}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"exp21_baselines_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df, timestamp


def evaluate_methods(df):
    """Compare tensor vs baseline methods."""
    labels = df["label"].values  # 0=knowable, 1=unknowable

    # Methods to compare
    methods = {
        "Tensor: Entropy": df["tensor_entropy"].values,
        "Tensor: Max Entropy": df["tensor_max_entropy"].values,
        "Tensor: -LogProb": -df["tensor_logprob"].values,
        "Tensor: -Top5 Mass": -df["tensor_top5"].values,
        "Baseline: Self-Report": df["baseline_self_report"].values,
        "Baseline: Hedging": df["baseline_hedge"].values,
        "Baseline: Length": df["baseline_length"].values,
        "Baseline: Combined": df["baseline_heuristic"].values,
    }

    print("\n" + "=" * 70)
    print("METHOD COMPARISON: Unknowable Detection")
    print("=" * 70)
    print("\nHigher AUC = better at detecting unknowable queries")
    print("-" * 60)
    print(f"{'Method':30s} {'AUC':>10s} {'Best Acc':>12s}")
    print("-" * 60)

    results = []
    for name, scores in methods.items():
        # Handle NaN
        valid_mask = np.isfinite(scores)
        if not valid_mask.all():
            scores = np.nan_to_num(scores, nan=np.nanmean(scores[valid_mask]))

        try:
            auc_score = roc_auc_score(labels, scores)

            # Find best threshold
            thresholds = np.percentile(scores, np.arange(0, 100, 5))
            best_acc = 0
            for thresh in thresholds:
                preds = (scores >= thresh).astype(int)
                acc = accuracy_score(labels, preds)
                best_acc = max(best_acc, acc)

            results.append({
                "method": name,
                "auc": auc_score,
                "best_acc": best_acc,
                "is_tensor": name.startswith("Tensor"),
            })

            print(f"{name:30s} {auc_score:>10.4f} {best_acc:>12.2%}")

        except Exception as e:
            print(f"{name:30s} ERROR: {e}")

    return pd.DataFrame(results)


def create_comparison_plots(df, eval_df, timestamp):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: AUC comparison bar chart
    ax = axes[0, 0]
    eval_df_sorted = eval_df.sort_values("auc", ascending=True)
    colors = ['blue' if is_tensor else 'gray' for is_tensor in eval_df_sorted["is_tensor"]]
    bars = ax.barh(eval_df_sorted["method"], eval_df_sorted["auc"], color=colors, alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', label='Random')
    ax.axvline(x=0.7, color='green', linestyle=':', alpha=0.7, label='Target (0.7)')
    ax.set_xlabel('AUC')
    ax.set_title('Unknowable Detection: Tensor vs Baselines')
    ax.legend()
    ax.set_xlim([0, 1])

    # Add value labels
    for bar, val in zip(bars, eval_df_sorted["auc"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    # Panel 2: Entropy distribution by category
    ax = axes[0, 1]
    knowable = df[df["category"] == "knowable"]["tensor_entropy"]
    unknowable = df[df["category"] == "unknowable"]["tensor_entropy"]
    ax.hist(knowable, bins=15, alpha=0.6, label='Knowable', color='green', density=True)
    ax.hist(unknowable, bins=15, alpha=0.6, label='Unknowable', color='red', density=True)
    ax.set_xlabel('Tensor Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Tensor Entropy Distribution')
    ax.legend()

    # Panel 3: Self-report distribution
    ax = axes[1, 0]
    knowable = df[df["category"] == "knowable"]["baseline_self_report"]
    unknowable = df[df["category"] == "unknowable"]["baseline_self_report"]
    ax.hist(knowable, bins=15, alpha=0.6, label='Knowable', color='green', density=True)
    ax.hist(unknowable, bins=15, alpha=0.6, label='Unknowable', color='red', density=True)
    ax.set_xlabel('Self-Report Uncertainty')
    ax.set_ylabel('Density')
    ax.set_title('Self-Reported Confidence (Inverted)')
    ax.legend()

    # Panel 4: Tensor vs Self-Report scatter
    ax = axes[1, 1]
    colors = ['green' if cat == 'knowable' else 'red' for cat in df["category"]]
    ax.scatter(df["tensor_entropy"], df["baseline_self_report"],
               c=colors, alpha=0.6, s=60)
    ax.set_xlabel('Tensor Entropy')
    ax.set_ylabel('Self-Report Uncertainty')
    ax.set_title('Tensor vs Self-Report: Knowable (green) vs Unknowable (red)')

    # Add quadrant interpretation
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=df["tensor_entropy"].median(), color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plot_path = f"exp21_baselines_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nComparison plot saved: {plot_path}")
    plt.close()


def print_summary(eval_df):
    """Print final comparison summary."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Separate tensor and baseline methods
    tensor_methods = eval_df[eval_df["is_tensor"]]
    baseline_methods = eval_df[~eval_df["is_tensor"]]

    best_tensor = tensor_methods.loc[tensor_methods["auc"].idxmax()]
    best_baseline = baseline_methods.loc[baseline_methods["auc"].idxmax()]

    print(f"\nBest Tensor Method: {best_tensor['method']}")
    print(f"  AUC: {best_tensor['auc']:.4f}")
    print(f"  Best Accuracy: {best_tensor['best_acc']:.2%}")

    print(f"\nBest Baseline Method: {best_baseline['method']}")
    print(f"  AUC: {best_baseline['auc']:.4f}")
    print(f"  Best Accuracy: {best_baseline['best_acc']:.2%}")

    improvement = best_tensor['auc'] - best_baseline['auc']
    print(f"\nTensor Improvement over Best Baseline: {improvement:+.4f}")

    print("\n--- Verdict ---")
    if best_tensor['auc'] > best_baseline['auc']:
        print("RESULT: Tensor signals OUTPERFORM simpler baselines")
        print("\nThe tensor interface provides discriminative signals that")
        print("self-report and heuristic methods cannot match.")
        print("This justifies the complexity of the tensor approach.")
    elif abs(improvement) < 0.05:
        print("RESULT: Tensor signals COMPARABLE to baselines")
        print("\nBoth approaches show similar performance.")
    else:
        print("RESULT: Tensor signals UNDERPERFORM baselines")
        print("\nSimpler methods may be sufficient for this task.")

    # Check if either meets threshold
    print(f"\nThreshold check (AUC >= 0.7):")
    print(f"  Best tensor: {'PASS' if best_tensor['auc'] >= 0.7 else 'FAIL'}")
    print(f"  Best baseline: {'PASS' if best_baseline['auc'] >= 0.7 else 'FAIL'}")

    return best_tensor['auc'] > best_baseline['auc']


def main():
    """Run the full baseline comparison experiment."""
    # Run comparison
    df, timestamp = run_baseline_comparison()

    # Evaluate methods
    eval_df = evaluate_methods(df)

    # Create visualizations
    create_comparison_plots(df, eval_df, timestamp)

    # Print summary
    tensor_wins = print_summary(eval_df)

    return tensor_wins


if __name__ == "__main__":
    main()
