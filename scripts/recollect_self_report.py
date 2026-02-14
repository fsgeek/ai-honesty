"""
Recollect Self-Report Confidence with Fixed Parsing

Problem: The original experiment 27 decoded the FULL sequence (prompt + generation)
then split on ":" to extract the confidence number. This grabbed numbers from the
prompt text (e.g., "0-100" in the question itself), not the model's actual
confidence response.

Fix: Decode ONLY the newly generated tokens by slicing at inputs.input_ids.shape[1].

This script:
1. Loads the original exp27 CSV (200 queries x 4 models = 800 rows)
2. For each instruct model, re-asks the confidence follow-up question
   using the ORIGINAL response from the CSV (no regeneration — tensor data is valid)
3. Parses confidence by decoding only new tokens
4. Outputs a new CSV with old_self_report vs new_self_report
5. Computes AUC comparison to check whether self-report inversion persists
"""

import torch
import pandas as pd
import numpy as np
import gc
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from datetime import datetime
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to original experiment 27 data
ORIGINAL_CSV = Path(__file__).parent.parent / "exp27_bounded_verification_20260206_205725.csv"

# Models in the order we process them (one at a time for 24GB VRAM)
MODELS = [
    ("OLMo", "allenai/olmo-3-7b-instruct"),
    ("Llama", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Qwen", "Qwen/Qwen3-4B-Instruct-2507"),
    ("Mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
]

MISTRAL_MODELS = {"mistralai/Mistral-7B-Instruct-v0.3"}


# ============================================================================
# Chat formatting
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


# ============================================================================
# Fixed self-report collection
# ============================================================================

def get_self_reported_confidence_fixed(model, tokenizer, query, response):
    """Ask the model how confident it is — with FIXED token-only parsing.

    The fix: decode only outputs[0][inputs.input_ids.shape[1]:] instead of
    the full sequence. This ensures we parse the model's actual generated
    confidence number, not numbers from the prompt text.
    """
    followup_prompt = format_chat(
        "You are a helpful assistant.",
        f"You just answered: '{response[:300]}' to the question '{query}'. "
        f"On a scale of 0-100, how confident are you in that answer? "
        f"Reply with ONLY a number.",
        tokenizer,
    )

    inputs = tokenizer(followup_prompt, return_tensors="pt").to(DEVICE)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # FIXED: decode ONLY the newly generated tokens
    new_token_ids = outputs[0][prompt_length:]
    confidence_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # Strip Qwen3 <think> tokens if present (they appear before skip_special_tokens
    # can catch them in some tokenizer versions)
    confidence_text = re.sub(r"<think>.*?</think>", "", confidence_text, flags=re.DOTALL)
    confidence_text = re.sub(r"<think>.*", "", confidence_text, flags=re.DOTALL)
    confidence_text = confidence_text.strip()

    # Parse the first number found
    numbers = re.findall(r"\d+", confidence_text)
    if numbers:
        conf = min(100, max(0, int(numbers[0]))) / 100.0
        return conf, confidence_text

    return 0.5, confidence_text  # Default if no number found


# ============================================================================
# Main collection loop
# ============================================================================

def collect_fixed_self_reports(df):
    """For each model, re-collect self-report confidence with fixed parsing."""
    all_results = []

    for family, model_id in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Loading model: {model_id} ({family})")
        print(f"{'=' * 70}")

        # Load tokenizer
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
            print(f"FAILED to load {model_id}: {e}")
            continue

        # Get this model's rows from the original CSV
        model_df = df[df["model_id"] == model_id].copy()
        print(f"Processing {len(model_df)} queries for {family}...")

        for idx, row in model_df.iterrows():
            query = row["query"]
            response = row["response"]
            old_sr = row["self_report_confidence"]
            category = row["category"]
            is_knowable = row["is_knowable"]

            new_sr, raw_text = get_self_reported_confidence_fixed(
                model, tokenizer, query, response
            )

            # Log progress with comparison
            query_short = query[:55] + "..." if len(query) > 55 else query
            print(
                f"  [{category[:4]}] {query_short:<60s} "
                f"old={old_sr:.2f} new={new_sr:.2f} raw='{raw_text}'"
            )

            all_results.append({
                "family": family,
                "model_id": model_id,
                "category": category,
                "is_knowable": is_knowable,
                "query": query,
                "old_self_report": old_sr,
                "new_self_report": new_sr,
                "raw_confidence_text": raw_text,
            })

        # Free GPU memory before loading next model
        print(f"\nFreeing GPU memory for {family}...")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Done.")

    return pd.DataFrame(all_results)


# ============================================================================
# Analysis and summary
# ============================================================================

def compute_sr_auc(labels, scores):
    """Compute AUC for self-report scores. Returns NaN if degenerate."""
    try:
        if len(set(labels)) < 2:
            return np.nan
        if len(set(scores)) < 2:
            return np.nan
        return roc_auc_score(labels, scores)
    except Exception:
        return np.nan


def analyze_results(results_df):
    """Print summary comparing old vs new self-report confidence."""
    print(f"\n{'=' * 70}")
    print("SUMMARY: Old vs Fixed Self-Report Confidence")
    print(f"{'=' * 70}")

    # Per-model summary
    print(f"\n{'Model':<12} {'Old SR AUC':>10} {'New SR AUC':>10} "
          f"{'Old K/U':>12} {'New K/U':>12} {'Old Inv?':>9} {'New Inv?':>9}")
    print("-" * 76)

    for family in results_df["family"].unique():
        fdf = results_df[results_df["family"] == family]

        # Labels: 1 = unknowable, 0 = knowable
        labels = (~fdf["is_knowable"].astype(bool)).astype(int).values

        # AUC using (1 - confidence) as uncertainty score
        # If model is well-calibrated: unknowable -> lower confidence -> higher (1-conf)
        #   -> AUC > 0.5
        # If inverted: unknowable -> higher confidence -> lower (1-conf)
        #   -> AUC < 0.5
        old_unc = 1.0 - fdf["old_self_report"].values
        new_unc = 1.0 - fdf["new_self_report"].values

        old_auc = compute_sr_auc(labels, old_unc)
        new_auc = compute_sr_auc(labels, new_unc)

        # Mean confidence per category
        old_k = fdf[fdf["is_knowable"] == True]["old_self_report"].mean()
        old_u = fdf[fdf["is_knowable"] == False]["old_self_report"].mean()
        new_k = fdf[fdf["is_knowable"] == True]["new_self_report"].mean()
        new_u = fdf[fdf["is_knowable"] == False]["new_self_report"].mean()

        old_inv = "YES" if old_auc < 0.5 else "no"
        new_inv = "YES" if new_auc < 0.5 else "no"

        print(
            f"{family:<12} {old_auc:>10.3f} {new_auc:>10.3f} "
            f"{old_k:>5.2f}/{old_u:<5.2f} {new_k:>5.2f}/{new_u:<5.2f} "
            f"{old_inv:>9} {new_inv:>9}"
        )

    # Aggregate
    print(f"\n{'=' * 70}")
    print("AGGREGATE (all models pooled)")
    print(f"{'=' * 70}")

    labels = (~results_df["is_knowable"].astype(bool)).astype(int).values
    old_unc = 1.0 - results_df["old_self_report"].values
    new_unc = 1.0 - results_df["new_self_report"].values

    old_auc = compute_sr_auc(labels, old_unc)
    new_auc = compute_sr_auc(labels, new_unc)

    old_k = results_df[results_df["is_knowable"] == True]["old_self_report"].mean()
    old_u = results_df[results_df["is_knowable"] == False]["old_self_report"].mean()
    new_k = results_df[results_df["is_knowable"] == True]["new_self_report"].mean()
    new_u = results_df[results_df["is_knowable"] == False]["new_self_report"].mean()

    print(f"  Old: AUC={old_auc:.3f}, Knowable mean={old_k:.3f}, Unknowable mean={old_u:.3f}")
    print(f"  New: AUC={new_auc:.3f}, Knowable mean={new_k:.3f}, Unknowable mean={new_u:.3f}")

    old_inv = old_auc < 0.5
    new_inv = new_auc < 0.5

    print(f"\n  Old parsing: inversion {'PRESENT' if old_inv else 'absent'}")
    print(f"  New parsing: inversion {'PRESENT' if new_inv else 'absent'}")

    if old_inv and new_inv:
        print("\n  CONCLUSION: Self-report inversion persists with fixed parsing.")
        print("  The original finding was NOT an artifact of the parsing bug.")
    elif old_inv and not new_inv:
        print("\n  CONCLUSION: Self-report inversion DISAPPEARS with fixed parsing.")
        print("  The original finding WAS an artifact of the parsing bug.")
    elif not old_inv and new_inv:
        print("\n  CONCLUSION: Fixed parsing REVEALS inversion not seen before.")
        print("  The parsing bug was masking the true self-report behavior.")
    else:
        print("\n  CONCLUSION: No inversion in either case.")

    # Per-category breakdown
    print(f"\n{'=' * 70}")
    print("PER-CATEGORY MEAN CONFIDENCE (New Self-Report)")
    print(f"{'=' * 70}")

    for family in results_df["family"].unique():
        fdf = results_df[results_df["family"] == family]
        print(f"\n  {family}:")
        for cat in ["knowable", "unknowable"]:
            cat_df = fdf[fdf["category"] == cat]
            mean_new = cat_df["new_self_report"].mean()
            std_new = cat_df["new_self_report"].std()
            mean_old = cat_df["old_self_report"].mean()
            std_old = cat_df["old_self_report"].std()
            print(
                f"    {cat:<12s}: old={mean_old:.3f} +/- {std_old:.3f}, "
                f"new={mean_new:.3f} +/- {std_new:.3f}"
            )


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("RECOLLECT SELF-REPORT CONFIDENCE WITH FIXED PARSING")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Original CSV: {ORIGINAL_CSV}")

    # Load original data
    if not ORIGINAL_CSV.exists():
        print(f"ERROR: Original CSV not found at {ORIGINAL_CSV}")
        sys.exit(1)

    df = pd.read_csv(ORIGINAL_CSV)
    print(f"Loaded {len(df)} rows from original CSV")
    print(f"Models: {df['model_id'].unique().tolist()}")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Queries per model: {len(df) // df['model_id'].nunique()}")

    # Verify all 4 models are present
    expected_models = {m[1] for m in MODELS}
    actual_models = set(df["model_id"].unique())
    missing = expected_models - actual_models
    if missing:
        print(f"WARNING: Missing models in CSV: {missing}")

    # Collect fixed self-reports
    results_df = collect_fixed_self_reports(df)

    if len(results_df) == 0:
        print("ERROR: No results collected!")
        sys.exit(1)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = Path(__file__).parent.parent / f"exp27_self_report_fixed_{timestamp}.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # Analyze
    analyze_results(results_df)

    print(f"\n{'=' * 70}")
    print("RECOLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Output: {output_csv}")
    print(f"Rows: {len(results_df)}")


if __name__ == "__main__":
    main()
