"""
Benchmark: Tensor Interface Overhead Measurement

Measures the actual performance cost of extracting epistemic signals during
LLM generation. The paper (design.tex §Implementation) claims "the overhead
is minimal: storing per-token entropy and summarizing attention patterns adds
negligible latency compared to generation itself." This benchmark replaces
that hand-wave with numbers.

Three conditions, same prompts, same model:
  A. Baseline:     model.generate() with no signal extraction
  B. Entropy:      model.generate(output_scores=True) + per-token entropy
  C. Full tensor:  model.generate(output_scores=True, output_attentions=True)
                   + per-token entropy + attention summary

Measurements per condition:
  - Wall-clock time per token (ms)
  - Peak GPU memory (MB)
  - Throughput (tokens/sec)
  - Storage cost per query for exported tensor data (bytes)

Benchmarking methodology:
  - 3 warmup iterations (discarded)
  - 10 measurement iterations per prompt
  - torch.cuda.synchronize() before every timing boundary
  - time.perf_counter() for sub-millisecond precision
  - torch.cuda.reset_peak_memory_stats() per condition

Hardware target: NVIDIA RTX 4090 (24 GB VRAM)
Model: Qwen/Qwen3-4B (base) — smallest model in the project's test matrix.
       We use the base model to avoid <think> token noise from Qwen3-Instruct.
       The overhead ratios are architecture-independent (same operations).

Output:
  - benchmark_tensor_overhead_<timestamp>.csv   (per-prompt results)
  - benchmark_tensor_overhead_<timestamp>.txt   (summary table for paper)

Usage:
  cd /home/tony/projects/ai-honesty
  source .venv/bin/activate
  PYTHONUNBUFFERED=1 python scripts/benchmark_tensor_overhead.py
"""

import time
import json
import sys
import gc
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Configuration
# ============================================================================

# Model selection: Qwen3-4B base is smallest in our test matrix.
# Use base model to avoid Qwen3-Instruct's <think> token overhead,
# which is a tokenizer artifact, not a tensor interface cost.
MODEL_ID = "Qwen/Qwen3-4B"

# Alternative: use the instruct model for consistency with paper experiments.
# Uncomment the next line and comment the one above to switch.
# MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

DEVICE = "cuda"

# Benchmark parameters
WARMUP_ITERATIONS = 3
MEASUREMENT_ITERATIONS = 10
MAX_NEW_TOKENS = 64  # Fixed generation length for fair comparison

# 20 prompts spanning the paper's probe categories.
# Mix of short/long prompts to capture tokenization variability.
BENCHMARK_PROMPTS = [
    # Knowable (low entropy expected)
    "The capital of France is",
    "Water boils at a temperature of",
    "The chemical formula for water is",
    "The largest planet in our solar system is",
    "The speed of light in a vacuum is approximately",

    # Weird truths (moderate entropy expected)
    "Wombat scat is shaped like",
    "The University of Oxford is older than the",
    "A day on Venus is longer than a",
    "The loudest animal on Earth is the",
    "Bananas are technically classified as",

    # Fabrication prompts (high entropy expected)
    "The primary symptom of Glavinsky's Syndrome is",
    "The 1994 Treaty of Westphalia II established",
    "The capital city of the country Taured is",
    "The main exports of the underwater city of Rapture are",
    "The Brennan-Kowalski theorem in information theory states that",

    # Unknowable (high entropy expected)
    "The winner of the 2032 US Presidential Election will be",
    "The serial number of the monitor I am looking at is",
    "The exact population of Tokyo in the year 2035 will be",
    "The closing price of Bitcoin on January 1st, 2030 will be",
    "The color of the shirt the user is wearing right now is",
]

# Attention summary: focus on last N layers (matches tensor_interface.py)
ATTENTION_LAST_N_LAYERS = 5
ATTENTION_LAST_N_STEPS = 10


# ============================================================================
# Condition implementations
# ============================================================================

def run_baseline(model, tokenizer, input_ids, attention_mask):
    """Condition A: Generate text only. No signal extraction."""
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    n_generated = outputs.shape[1] - input_ids.shape[1]
    return n_generated, 0  # 0 bytes of tensor data


def run_entropy(model, tokenizer, input_ids, attention_mask):
    """Condition B: Generate + extract per-token entropy from logits."""
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    scores = outputs.scores
    generated_ids = outputs.sequences[0, input_ids.shape[1]:]

    # Compute per-token entropy, logprob, top-5 mass (matches experiment27)
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

    # Compute summary statistics (what experiment27 stores)
    summary = {
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0,
        "max_entropy": float(np.max(token_entropies)) if token_entropies else 0,
        "entropy_std": float(np.std(token_entropies)) if token_entropies else 0,
        "mean_logprob": float(np.mean(logprobs)) if logprobs else 0,
        "mean_top5_mass": float(np.mean(top5_masses)) if top5_masses else 0,
    }

    # Storage cost: entropy trace + summary stats
    tensor_data = {
        "entropy_trace": token_entropies,
        "logprobs": logprobs,
        "top5_masses": top5_masses,
        "summary": summary,
    }
    storage_bytes = len(json.dumps(tensor_data).encode("utf-8"))

    n_generated = len(token_entropies)
    return n_generated, storage_bytes


def run_full_tensor(model, tokenizer, input_ids, attention_mask):
    """Condition C: Generate + entropy + attention patterns.

    This is the full tensor interface as described in the paper and
    implemented in tensor_interface.py.
    """
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        output_attentions=True,
        return_dict_in_generate=True,
    )

    scores = outputs.scores
    generated_ids = outputs.sequences[0, input_ids.shape[1]:]

    # --- Entropy extraction (same as Condition B) ---
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

    # --- Attention summary (matches tensor_interface.py _compute_attention_summary) ---
    attention_summary = {}
    if hasattr(outputs, "attentions") and outputs.attentions:
        concentrations = []
        self_attention_ratios = []

        # outputs.attentions is a tuple of per-step tuples of per-layer tensors
        # Each step: tuple of (batch, heads, seq, seq) tensors, one per layer
        n_steps = len(outputs.attentions)
        step_start = max(0, n_steps - ATTENTION_LAST_N_STEPS)

        for step_attentions in outputs.attentions[step_start:]:
            n_layers = len(step_attentions)
            layer_start = max(0, n_layers - ATTENTION_LAST_N_LAYERS)

            for layer_attn in step_attentions[layer_start:]:
                # layer_attn: [batch, heads, seq, seq]
                attn = layer_attn.squeeze(0).float().cpu().numpy()

                for head in attn:
                    concentrations.append(np.mean(np.max(head, axis=-1)))
                    diag_sum = np.trace(head)
                    total_sum = np.sum(head)
                    if total_sum > 0:
                        self_attention_ratios.append(diag_sum / total_sum)

        attention_summary = {
            "concentration": float(np.mean(concentrations)) if concentrations else 0,
            "self_attention": float(np.mean(self_attention_ratios)) if self_attention_ratios else 0,
        }

    # Summary statistics
    summary = {
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0,
        "max_entropy": float(np.max(token_entropies)) if token_entropies else 0,
        "entropy_std": float(np.std(token_entropies)) if token_entropies else 0,
        "mean_logprob": float(np.mean(logprobs)) if logprobs else 0,
        "mean_top5_mass": float(np.mean(top5_masses)) if top5_masses else 0,
    }

    # Storage cost: entropy trace + attention summary + summary stats
    tensor_data = {
        "entropy_trace": token_entropies,
        "logprobs": logprobs,
        "top5_masses": top5_masses,
        "attention_summary": attention_summary,
        "summary": summary,
    }
    storage_bytes = len(json.dumps(tensor_data).encode("utf-8"))

    n_generated = len(token_entropies)
    return n_generated, storage_bytes


# ============================================================================
# Benchmark harness
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results for a single (prompt, condition, iteration) measurement."""
    prompt_idx: int
    prompt_text: str
    condition: str
    iteration: int
    wall_clock_ms: float
    n_tokens: int
    ms_per_token: float
    tokens_per_sec: float
    peak_memory_mb: float
    storage_bytes: int


def benchmark_condition(
    condition_name: str,
    condition_fn,
    model,
    tokenizer,
    prompts: List[str],
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = MEASUREMENT_ITERATIONS,
) -> List[BenchmarkResult]:
    """Run a benchmark condition across all prompts.

    For each prompt:
      1. Run `warmup` iterations (discarded)
      2. Run `iterations` measurement iterations
      3. Record wall-clock time, tokens generated, peak memory, storage cost
    """
    results = []

    print(f"\n{'='*70}")
    print(f"CONDITION: {condition_name}")
    print(f"  Warmup: {warmup} iterations, Measurement: {iterations} iterations")
    print(f"  Prompts: {len(prompts)}, Max tokens: {MAX_NEW_TOKENS}")
    print(f"{'='*70}")

    for p_idx, prompt in enumerate(prompts):
        # Tokenize once (shared across iterations)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE)
        n_input_tokens = input_ids.shape[1]

        print(f"\n  [{p_idx+1}/{len(prompts)}] {prompt[:50]}... "
              f"({n_input_tokens} input tokens)")

        # --- Warmup ---
        for w in range(warmup):
            with torch.no_grad():
                _ = condition_fn(model, tokenizer, input_ids, attention_mask)
            # Synchronize to ensure warmup is complete
            torch.cuda.synchronize()

        # Reset peak memory after warmup to measure only the condition
        torch.cuda.reset_peak_memory_stats()

        # --- Measurement ---
        for it in range(iterations):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                n_generated, storage_bytes = condition_fn(
                    model, tokenizer, input_ids, attention_mask
                )

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            wall_clock_ms = (t_end - t_start) * 1000.0
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            ms_per_token = wall_clock_ms / max(1, n_generated)
            tokens_per_sec = n_generated / max(1e-9, (t_end - t_start))

            result = BenchmarkResult(
                prompt_idx=p_idx,
                prompt_text=prompt,
                condition=condition_name,
                iteration=it,
                wall_clock_ms=wall_clock_ms,
                n_tokens=n_generated,
                ms_per_token=ms_per_token,
                tokens_per_sec=tokens_per_sec,
                peak_memory_mb=peak_memory_mb,
                storage_bytes=storage_bytes,
            )
            results.append(result)

        # Print running stats for this prompt
        prompt_results = [r for r in results if r.prompt_idx == p_idx]
        times = [r.ms_per_token for r in prompt_results]
        print(f"    -> {np.mean(times):.2f} +/- {np.std(times):.2f} ms/token "
              f"({prompt_results[0].n_tokens} tokens, "
              f"{prompt_results[0].peak_memory_mb:.0f} MB peak)")

    return results


def print_summary(all_results: List[BenchmarkResult], output_file=None):
    """Print and optionally write the summary table."""
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit("TENSOR INTERFACE OVERHEAD BENCHMARK — SUMMARY")
    emit("=" * 78)
    emit(f"Model: {MODEL_ID}")
    emit(f"Device: {torch.cuda.get_device_name(0)}")
    emit(f"Max new tokens: {MAX_NEW_TOKENS}")
    emit(f"Prompts: {len(BENCHMARK_PROMPTS)}")
    emit(f"Warmup iterations: {WARMUP_ITERATIONS}")
    emit(f"Measurement iterations: {MEASUREMENT_ITERATIONS}")
    emit()

    df = pd.DataFrame([vars(r) for r in all_results])

    # Per-condition aggregate statistics
    emit(f"{'Condition':<16} {'ms/tok':>10} {'std':>8} {'tok/s':>10} "
         f"{'Peak MB':>10} {'Storage B':>12}")
    emit("-" * 78)

    condition_stats = {}
    for condition in ["Baseline", "Entropy", "Full Tensor"]:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue

        ms_tok_mean = cond_df["ms_per_token"].mean()
        ms_tok_std = cond_df["ms_per_token"].std()
        tok_sec_mean = cond_df["tokens_per_sec"].mean()
        peak_mem = cond_df["peak_memory_mb"].max()
        storage_mean = cond_df["storage_bytes"].mean()

        condition_stats[condition] = {
            "ms_per_token": ms_tok_mean,
            "ms_per_token_std": ms_tok_std,
            "tokens_per_sec": tok_sec_mean,
            "peak_memory_mb": peak_mem,
            "storage_bytes_mean": storage_mean,
        }

        emit(f"{condition:<16} {ms_tok_mean:>10.2f} {ms_tok_std:>8.2f} "
             f"{tok_sec_mean:>10.1f} {peak_mem:>10.0f} "
             f"{storage_mean:>12.0f}")

    emit()

    # Overhead analysis
    emit("--- OVERHEAD ANALYSIS ---")
    if "Baseline" in condition_stats and "Entropy" in condition_stats:
        base = condition_stats["Baseline"]
        ent = condition_stats["Entropy"]
        entropy_overhead_ms = ent["ms_per_token"] - base["ms_per_token"]
        entropy_overhead_pct = (entropy_overhead_ms / base["ms_per_token"]) * 100
        entropy_mem_delta = ent["peak_memory_mb"] - base["peak_memory_mb"]
        throughput_drop = ((base["tokens_per_sec"] - ent["tokens_per_sec"])
                          / base["tokens_per_sec"]) * 100

        emit(f"Entropy extraction overhead:")
        emit(f"  Latency:    +{entropy_overhead_ms:.2f} ms/token "
             f"({entropy_overhead_pct:+.1f}%)")
        emit(f"  Memory:     +{entropy_mem_delta:.0f} MB peak "
             f"({entropy_mem_delta / base['peak_memory_mb'] * 100:+.1f}%)")
        emit(f"  Throughput: {throughput_drop:+.1f}% degradation")
        emit(f"  Storage:    {ent['storage_bytes_mean']:.0f} bytes/query avg")

    if "Baseline" in condition_stats and "Full Tensor" in condition_stats:
        base = condition_stats["Baseline"]
        full = condition_stats["Full Tensor"]
        full_overhead_ms = full["ms_per_token"] - base["ms_per_token"]
        full_overhead_pct = (full_overhead_ms / base["ms_per_token"]) * 100
        full_mem_delta = full["peak_memory_mb"] - base["peak_memory_mb"]
        throughput_drop = ((base["tokens_per_sec"] - full["tokens_per_sec"])
                          / base["tokens_per_sec"]) * 100

        emit(f"\nFull tensor interface overhead:")
        emit(f"  Latency:    +{full_overhead_ms:.2f} ms/token "
             f"({full_overhead_pct:+.1f}%)")
        emit(f"  Memory:     +{full_mem_delta:.0f} MB peak "
             f"({full_mem_delta / base['peak_memory_mb'] * 100:+.1f}%)")
        emit(f"  Throughput: {throughput_drop:+.1f}% degradation")
        emit(f"  Storage:    {full['storage_bytes_mean']:.0f} bytes/query avg")

    if "Entropy" in condition_stats and "Full Tensor" in condition_stats:
        ent = condition_stats["Entropy"]
        full = condition_stats["Full Tensor"]
        attn_overhead_ms = full["ms_per_token"] - ent["ms_per_token"]
        attn_overhead_pct = (attn_overhead_ms / ent["ms_per_token"]) * 100
        attn_mem_delta = full["peak_memory_mb"] - ent["peak_memory_mb"]

        emit(f"\nAttention extraction marginal cost (Full - Entropy):")
        emit(f"  Latency:    +{attn_overhead_ms:.2f} ms/token "
             f"({attn_overhead_pct:+.1f}%)")
        emit(f"  Memory:     +{attn_mem_delta:.0f} MB peak")

    emit()

    # Per-condition token count sanity check
    emit("--- SANITY CHECK: Token counts ---")
    for condition in ["Baseline", "Entropy", "Full Tensor"]:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue
        tok_mean = cond_df["n_tokens"].mean()
        tok_std = cond_df["n_tokens"].std()
        emit(f"  {condition:<16}: {tok_mean:.1f} +/- {tok_std:.1f} tokens")

    emit()
    emit("--- FOR PAPER (LaTeX-ready) ---")
    if "Baseline" in condition_stats and "Full Tensor" in condition_stats:
        base = condition_stats["Baseline"]
        ent = condition_stats["Entropy"]
        full = condition_stats["Full Tensor"]
        emit(f"% Overhead table: Qwen3-4B, {MAX_NEW_TOKENS} tokens, "
             f"RTX 4090, {MEASUREMENT_ITERATIONS} iterations x "
             f"{len(BENCHMARK_PROMPTS)} prompts")
        emit(f"% Condition       & ms/tok & tok/s & Peak MB & Storage B \\\\")
        emit(f"% Baseline        & {base['ms_per_token']:.2f} "
             f"& {base['tokens_per_sec']:.1f} "
             f"& {base['peak_memory_mb']:.0f} & --- \\\\")
        emit(f"% + Entropy       & {ent['ms_per_token']:.2f} "
             f"& {ent['tokens_per_sec']:.1f} "
             f"& {ent['peak_memory_mb']:.0f} "
             f"& {ent['storage_bytes_mean']:.0f} \\\\")
        emit(f"% + Full Tensor   & {full['ms_per_token']:.2f} "
             f"& {full['tokens_per_sec']:.1f} "
             f"& {full['peak_memory_mb']:.0f} "
             f"& {full['storage_bytes_mean']:.0f} \\\\")
        entropy_pct = ((ent['ms_per_token'] - base['ms_per_token'])
                       / base['ms_per_token']) * 100
        full_pct = ((full['ms_per_token'] - base['ms_per_token'])
                    / base['ms_per_token']) * 100
        emit(f"% Entropy overhead: {entropy_pct:+.1f}%")
        emit(f"% Full tensor overhead: {full_pct:+.1f}%")

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"\nSummary written to: {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("TENSOR INTERFACE OVERHEAD BENCHMARK")
    print("=" * 70)

    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model:  {MODEL_ID}")
    print(f"Config: {WARMUP_ITERATIONS} warmup, {MEASUREMENT_ITERATIONS} measurement, "
          f"{MAX_NEW_TOKENS} max tokens")
    print(f"Prompts: {len(BENCHMARK_PROMPTS)}")

    # Load model with eager attention (required for output_attentions)
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        # Must use eager attention to support output_attentions=True.
        # flash_attention_2 and sdpa do NOT support returning attention weights.
        # This is a real constraint of the tensor interface — it requires
        # eager attention, which is slower than optimized kernels.
        # The benchmark captures this cost honestly.
        attn_implementation="eager",
    )

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    print(f"GPU memory after load: "
          f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    # Run conditions in order: baseline first (most conservative memory),
    # then entropy, then full tensor (most expensive).
    conditions = [
        ("Baseline", run_baseline),
        ("Entropy", run_entropy),
        ("Full Tensor", run_full_tensor),
    ]

    for condition_name, condition_fn in conditions:
        results = benchmark_condition(
            condition_name,
            condition_fn,
            model,
            tokenizer,
            BENCHMARK_PROMPTS,
        )
        all_results.extend(results)

        # Save incremental results
        df = pd.DataFrame([vars(r) for r in all_results])
        csv_path = f"benchmark_tensor_overhead_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

    # Print and save summary
    txt_path = f"benchmark_tensor_overhead_{timestamp}.txt"
    print_summary(all_results, output_file=txt_path)

    # Save final CSV
    df = pd.DataFrame([vars(r) for r in all_results])
    csv_path = f"benchmark_tensor_overhead_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved:")
    print(f"  CSV:     {csv_path}")
    print(f"  Summary: {txt_path}")
    print(f"\nTotal measurements: {len(all_results)}")
    print(f"  = {len(BENCHMARK_PROMPTS)} prompts x "
          f"{MEASUREMENT_ITERATIONS} iterations x "
          f"{len(conditions)} conditions")


if __name__ == "__main__":
    main()
