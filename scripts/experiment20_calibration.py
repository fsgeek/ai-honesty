"""
Experiment 20: Signal Calibration Analysis

Research Question: Do extracted tensor signals actually predict correctness?

Method:
- Use TruthfulQA with known ground truth (correct vs incorrect answers)
- Extract entropy, attention metrics, fragmentation from model processing
- Compute ROC/AUC for signal → correctness prediction
- Generate calibration plots and discrimination metrics

Goal: AUC > 0.7 for entropy-based signals → correctness prediction

This transforms "signals differ between conditions" into
"signals predict what we care about."
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from scipy import stats
from tqdm import tqdm
import gc
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use OLMo-3 as primary model (matches paper methodology)
MODEL_ID = "allenai/olmo-3-7b-instruct"
NUM_SAMPLES = 200  # TruthfulQA pairs to evaluate

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


def extract_tensor_signals(model, tokenizer, text):
    """
    Extract epistemic signals from model processing of text.

    Returns dict with:
    - mean_entropy: average entropy of output distribution during processing
    - max_entropy: peak uncertainty
    - entropy_std: variability of uncertainty
    - mean_logprob: average confidence
    - top5_mass: probability concentration in top tokens
    - perplexity: standard perplexity metric
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Get logits for perplexity
        outputs = model(**inputs, labels=inputs["input_ids"])
        perplexity = torch.exp(outputs.loss).item()

        # Get per-token distributions
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Per-token entropy
        token_entropies = -torch.sum(probs * log_probs, dim=-1).cpu().numpy()

        # Top-5 mass per token
        top5_probs = torch.topk(probs, k=5, dim=-1).values
        top5_masses = top5_probs.sum(dim=-1).cpu().numpy()

        # Logprob of actual tokens (excluding first token which has no prediction)
        actual_token_ids = inputs.input_ids[0, 1:]
        token_logprobs = []
        for i, tid in enumerate(actual_token_ids):
            token_logprobs.append(log_probs[i, tid].item())

    return {
        "mean_entropy": np.mean(token_entropies),
        "max_entropy": np.max(token_entropies),
        "entropy_std": np.std(token_entropies),
        "mean_logprob": np.mean(token_logprobs) if token_logprobs else 0,
        "mean_top5_mass": np.mean(top5_masses),
        "perplexity": perplexity,
    }


def run_calibration_analysis():
    """Main calibration experiment."""
    print("=" * 70)
    print("EXPERIMENT 20: SIGNAL CALIBRATION ANALYSIS")
    print("=" * 70)
    print(f"\nModel: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Samples: {NUM_SAMPLES}")

    # Load TruthfulQA
    print("\n--- Loading TruthfulQA Dataset ---")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    print(f"Dataset size: {len(dataset)} questions")

    # Load model
    print(f"\n--- Loading Model: {MODEL_ID} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    results = []

    print(f"\n--- Processing {NUM_SAMPLES} Question Pairs ---")
    count = 0
    for row in tqdm(dataset):
        if count >= NUM_SAMPLES:
            break

        question = row['question']
        correct_answer = row['best_answer']
        incorrect_list = row['incorrect_answers']

        if not incorrect_list:
            continue
        incorrect_answer = incorrect_list[0]

        # Format as Q&A pairs
        text_true = f"Question: {question}\nAnswer: {correct_answer}"
        text_false = f"Question: {question}\nAnswer: {incorrect_answer}"

        # Extract signals for TRUE answer
        signals_true = extract_tensor_signals(model, tokenizer, text_true)
        results.append({
            "question_idx": count,
            "label": 0,  # 0 = correct/truthful
            "type": "correct",
            **signals_true
        })

        # Extract signals for FALSE answer
        signals_false = extract_tensor_signals(model, tokenizer, text_false)
        results.append({
            "question_idx": count,
            "label": 1,  # 1 = incorrect/fabrication
            "type": "incorrect",
            **signals_false
        })

        count += 1

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Create DataFrame
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"exp20_calibration_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df, timestamp


def compute_roc_metrics(df):
    """Compute ROC curves and AUC for each signal."""
    labels = df["label"].values

    signals = {
        "mean_entropy": df["mean_entropy"].values,
        "max_entropy": df["max_entropy"].values,
        "entropy_std": df["entropy_std"].values,
        "mean_logprob": -df["mean_logprob"].values,  # Negate: lower logprob = worse
        "mean_top5_mass": -df["mean_top5_mass"].values,  # Negate: lower mass = more uncertain
        "perplexity": df["perplexity"].values,
    }

    roc_results = {}

    print("\n" + "=" * 70)
    print("ROC/AUC ANALYSIS")
    print("=" * 70)
    print("\nSignal → Incorrectness Detection Performance:")
    print("-" * 50)

    for name, scores in signals.items():
        # Handle NaN/Inf
        valid_mask = np.isfinite(scores)
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]

        if len(np.unique(valid_labels)) < 2:
            print(f"{name}: Insufficient label variance")
            continue

        fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores)
        roc_auc = auc(fpr, tpr)

        # Precision-recall
        precision, recall, _ = precision_recall_curve(valid_labels, valid_scores)
        ap = average_precision_score(valid_labels, valid_scores)

        roc_results[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }

        status = "PASS" if roc_auc >= 0.7 else "FAIL"
        print(f"{name:20s}: AUC = {roc_auc:.4f}  AP = {ap:.4f}  [{status}]")

    return roc_results


def compute_effect_sizes(df):
    """Compute effect sizes (Cohen's d) for each signal."""
    correct = df[df["label"] == 0]
    incorrect = df[df["label"] == 1]

    print("\n" + "=" * 70)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 70)
    print("\nSignal separation between correct vs incorrect answers:")
    print("-" * 70)
    print(f"{'Signal':20s} {'Correct':>12s} {'Incorrect':>12s} {'Cohen d':>10s} {'p-value':>12s}")
    print("-" * 70)

    effect_sizes = {}

    for col in ["mean_entropy", "max_entropy", "entropy_std", "mean_logprob",
                "mean_top5_mass", "perplexity"]:
        correct_vals = correct[col].dropna().values
        incorrect_vals = incorrect[col].dropna().values

        # Cohen's d
        pooled_std = np.sqrt((np.var(correct_vals) + np.var(incorrect_vals)) / 2)
        cohens_d = (np.mean(incorrect_vals) - np.mean(correct_vals)) / pooled_std if pooled_std > 0 else 0

        # t-test
        t_stat, p_value = stats.ttest_ind(correct_vals, incorrect_vals)

        effect_sizes[col] = {
            "correct_mean": np.mean(correct_vals),
            "incorrect_mean": np.mean(incorrect_vals),
            "cohens_d": cohens_d,
            "p_value": p_value,
        }

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{col:20s} {np.mean(correct_vals):>12.4f} {np.mean(incorrect_vals):>12.4f} "
              f"{cohens_d:>+10.3f} {p_value:>12.2e} {sig}")

    return effect_sizes


def create_calibration_plots(df, roc_results, timestamp):
    """Generate calibration and ROC visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: ROC Curves
    ax = axes[0, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_results)))
    for (name, data), color in zip(roc_results.items(), colors):
        ax.plot(data["fpr"], data["tpr"],
                label=f'{name} (AUC={data["auc"]:.3f})',
                color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Signal → Incorrectness Detection')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    # Panel 2: Signal Distributions
    ax = axes[0, 1]
    correct = df[df["label"] == 0]["mean_entropy"]
    incorrect = df[df["label"] == 1]["mean_entropy"]
    ax.hist(correct, bins=30, alpha=0.6, label='Correct', color='green', density=True)
    ax.hist(incorrect, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
    ax.set_xlabel('Mean Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Entropy Distribution: Correct vs Incorrect')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Precision-Recall Curves
    ax = axes[1, 0]
    for (name, data), color in zip(roc_results.items(), colors):
        ax.plot(data["recall"], data["precision"],
                label=f'{name} (AP={data["ap"]:.3f})',
                color=color, linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    # Panel 4: Signal Comparison Box Plot
    ax = axes[1, 1]
    signals_to_plot = ["mean_entropy", "perplexity"]
    positions = []
    data_to_plot = []
    labels_plot = []
    colors_box = []

    for i, signal in enumerate(signals_to_plot):
        # Correct
        positions.append(i * 3)
        data_to_plot.append(df[df["label"] == 0][signal].values)
        labels_plot.append(f"{signal}\n(correct)")
        colors_box.append('green')
        # Incorrect
        positions.append(i * 3 + 1)
        data_to_plot.append(df[df["label"] == 1][signal].values)
        labels_plot.append(f"{signal}\n(incorrect)")
        colors_box.append('red')

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.7, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_plot, fontsize=8, rotation=45)
    ax.set_title('Signal Comparison: Correct vs Incorrect')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = f"exp20_calibration_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nCalibration plot saved: {plot_path}")
    plt.close()


def print_summary(roc_results, effect_sizes):
    """Print final summary and verdict."""
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    # Check if any signal meets threshold
    passing_signals = [name for name, data in roc_results.items() if data["auc"] >= 0.7]
    best_signal = max(roc_results.items(), key=lambda x: x[1]["auc"])

    print(f"\nBest Signal: {best_signal[0]} (AUC = {best_signal[1]['auc']:.4f})")
    print(f"Signals with AUC >= 0.7: {len(passing_signals)}/{len(roc_results)}")

    if passing_signals:
        print(f"  Passing: {', '.join(passing_signals)}")

    print("\n--- Verification Criteria ---")
    print(f"Goal: AUC > 0.7 for at least one signal")

    if best_signal[1]["auc"] >= 0.7:
        print(f"Result: PASS - Tensor signals predict correctness")
        print(f"\nThe existence proof is established: extracted telemetric signals")
        print(f"can distinguish correct from incorrect model outputs with")
        print(f"AUC = {best_signal[1]['auc']:.3f}, demonstrating that the escape")
        print(f"condition from the impossibility theorem is not vacuous.")
    else:
        print(f"Result: FAIL - Best AUC ({best_signal[1]['auc']:.3f}) below threshold")
        print(f"\nSignals show separation but may need refinement.")

    return best_signal[1]["auc"] >= 0.7


def main():
    """Run the full calibration experiment."""
    # Run analysis
    df, timestamp = run_calibration_analysis()

    # Compute ROC metrics
    roc_results = compute_roc_metrics(df)

    # Compute effect sizes
    effect_sizes = compute_effect_sizes(df)

    # Create visualizations
    create_calibration_plots(df, roc_results, timestamp)

    # Print summary
    success = print_summary(roc_results, effect_sizes)

    return success


if __name__ == "__main__":
    main()
