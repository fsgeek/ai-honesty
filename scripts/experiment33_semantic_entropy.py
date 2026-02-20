"""
Experiment 33: Semantic Entropy vs Per-Token Entropy

Research Question: Does semantic entropy (Kuhn et al. 2023) — which clusters
sampled responses by meaning and computes entropy over cluster assignments —
outperform per-token entropy for discriminating knowable from unknowable queries?

Background: Kuhn et al. (2023, "Semantic Uncertainty") propose sampling multiple
responses, clustering them by semantic equivalence, and computing entropy over the
cluster distribution. The intuition is that models "know what they don't know" in
the sense that unknowable queries produce semantically diverse responses while
knowable queries converge. This is an expensive procedure: N forward passes per
query (typically N=5-20) plus embedding and clustering.

Our paper shows that per-token entropy from a SINGLE forward pass achieves
AUC 0.72-1.00 for knowable/unknowable discrimination. If semantic entropy
achieves comparable AUC, the cost ratio (10x-20x more compute) matters. If it
achieves lower AUC, the additional complexity buys nothing.

Method:
  - Load Qwen3-4B instruct (smallest model, fastest iteration)
  - Use 200 queries from experiment 27b (100 knowable, 100 unknowable)
  - Per query: generate 10 responses with temperature=0.7
  - Embed responses with sentence-transformers (all-MiniLM-L6-v2)
  - Cluster with agglomerative clustering (cosine distance, threshold 0.3)
  - Compute semantic entropy: H = -sum(p_c * log(p_c)) over clusters
  - Compare AUC: semantic entropy vs per-token mean entropy (from exp27b CSV)
  - Report cost ratio: forward passes required per query

Reference:
  Kuhn, L., Gal, Y., & Farquhar, S. (2023). "Semantic Uncertainty:
  Linguistic Invariances for Uncertainty Estimation in Natural Language
  Generation." ICLR 2023.
"""

import os
import sys
import re
import gc
import time
import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.cluster import AgglomerativeClustering

# Ensure unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Configuration
# ============================================================================

# Model for generation (smallest in our test suite, fastest iteration)
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_FAMILY = "Qwen"

# Embedding model for semantic clustering
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Experiment 27b CSV with per-token entropy already computed
EXP27B_CSV = "exp27b_detailed_20260206_230203.csv"

# Semantic entropy parameters
NUM_SAMPLES = 10           # Responses per query (Kuhn et al. use 5-20)
TEMPERATURE = 0.7          # Sampling temperature for diverse responses
MAX_NEW_TOKENS = 150       # Max tokens per response
COSINE_DISTANCE_THRESHOLD = 0.3  # Agglomerative clustering threshold

# Reproducibility
SEED = 42

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions directly and concisely."


# ============================================================================
# Utility: strip <think>...</think> tokens from Qwen output
# ============================================================================

def strip_thinking_tokens(text):
    """Remove <think>...</think> blocks from Qwen3 output.

    Qwen3-Instruct models emit reasoning traces wrapped in <think> tags.
    These must be stripped before embedding for semantic clustering, since
    the reasoning trace is not the response content.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


# ============================================================================
# Load queries from experiment 27b
# ============================================================================

def load_queries(csv_path, family="Qwen"):
    """Load queries and per-token entropy from experiment 27b CSV.

    Filters to the specified model family and returns a DataFrame with
    columns: query, is_knowable, mean_entropy.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        print(f"Expected: experiment 27b detailed results.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df_family = df[df["family"] == family].copy()

    if len(df_family) == 0:
        print(f"ERROR: No rows for family='{family}' in {csv_path}")
        print(f"Available families: {df['family'].unique()}")
        sys.exit(1)

    print(f"Loaded {len(df_family)} queries for {family} from {csv_path}")
    print(f"  Knowable: {df_family['is_knowable'].sum()}")
    print(f"  Unknowable: {(~df_family['is_knowable']).sum()}")

    return df_family


# ============================================================================
# Model loading
# ============================================================================

def load_generation_model(model_id):
    """Load the generation model and tokenizer."""
    print(f"\nLoading generation model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"  Model loaded on {DEVICE}")
    return model, tokenizer


def load_embedding_model(model_name):
    """Load the sentence-transformers embedding model."""
    print(f"\nLoading embedding model: {model_name}")

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(model_name, device=DEVICE)

    print(f"  Embedding model loaded on {DEVICE}")
    return embed_model


# ============================================================================
# Generation: sample N responses per query
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


def generate_samples(model, tokenizer, query, num_samples=NUM_SAMPLES,
                     temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                     seed=SEED):
    """Generate N diverse responses for a single query.

    Uses temperature sampling with a fixed base seed. Each sample uses
    seed + i for reproducibility while ensuring diversity.

    Returns a list of response strings (with <think> tokens stripped).
    """
    prompt = format_chat(SYSTEM_PROMPT, query, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]

    responses = []

    for i in range(num_samples):
        torch.manual_seed(seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + i)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0, prompt_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Strip <think>...</think> tokens from Qwen output
        response_text = strip_thinking_tokens(response_text)

        responses.append(response_text)

    return responses


# ============================================================================
# Semantic entropy computation
# ============================================================================

def compute_semantic_entropy(responses, embed_model,
                             distance_threshold=COSINE_DISTANCE_THRESHOLD):
    """Compute semantic entropy over a set of responses.

    Steps:
      1. Embed all responses using sentence-transformers
      2. Cluster embeddings with agglomerative clustering (cosine distance)
      3. Compute Shannon entropy over the cluster assignment distribution

    Args:
        responses: list of N response strings
        embed_model: sentence-transformers model
        distance_threshold: cosine distance threshold for clustering

    Returns:
        dict with:
            semantic_entropy: H over cluster distribution (nats)
            num_clusters: number of distinct semantic clusters
            cluster_sizes: list of cluster sizes
            cluster_labels: list of cluster assignments per response
    """
    if not responses or all(r.strip() == "" for r in responses):
        return {
            "semantic_entropy": 0.0,
            "num_clusters": 1,
            "cluster_sizes": [len(responses)],
            "cluster_labels": [0] * len(responses),
        }

    # Handle edge case: all responses identical (or nearly so)
    # Embed even if identical — let clustering decide
    embeddings = embed_model.encode(responses, convert_to_numpy=True)

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    embeddings_normed = embeddings / norms

    n = len(responses)

    if n == 1:
        return {
            "semantic_entropy": 0.0,
            "num_clusters": 1,
            "cluster_sizes": [1],
            "cluster_labels": [0],
        }

    # Agglomerative clustering with cosine distance
    # metric="cosine" uses 1 - cosine_similarity as distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings_normed)

    # Compute cluster distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()

    # Shannon entropy in nats
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    return {
        "semantic_entropy": float(entropy),
        "num_clusters": int(len(unique_labels)),
        "cluster_sizes": counts.tolist(),
        "cluster_labels": labels.tolist(),
    }


# ============================================================================
# Per-token entropy (single-pass, for fresh computation if needed)
# ============================================================================

def compute_pertoken_entropy(model, tokenizer, query, max_new_tokens=MAX_NEW_TOKENS):
    """Compute per-token entropy from a single greedy forward pass.

    This mirrors the methodology of experiments 23/24/27: generate greedily
    with output_scores=True and compute full-vocabulary entropy at each step.

    Returns:
        dict with mean_entropy, max_entropy, entropy_std, num_tokens
    """
    prompt = format_chat(SYSTEM_PROMPT, query, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    scores = outputs.scores  # tuple of (1, vocab_size) tensors

    token_entropies = []
    for score in scores:
        logits = score.squeeze(0).float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        token_entropies.append(entropy)

    arr = np.array(token_entropies) if token_entropies else np.array([0.0])
    return {
        "mean_entropy": float(np.mean(arr)),
        "max_entropy": float(np.max(arr)),
        "entropy_std": float(np.std(arr)),
        "num_tokens": len(arr),
    }


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(args):
    """Run the full semantic entropy vs per-token entropy comparison."""

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load queries from exp27b
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), EXP27B_CSV)
    if not os.path.exists(csv_path):
        # Try relative to CWD
        csv_path = EXP27B_CSV
    df_queries = load_queries(csv_path, family=MODEL_FAMILY)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 33: SEMANTIC ENTROPY vs PER-TOKEN ENTROPY")
    print(f"{'=' * 70}")
    print(f"Timestamp:       {timestamp}")
    print(f"Model:           {MODEL_ID}")
    print(f"Embedding:       {EMBEDDING_MODEL}")
    print(f"Samples/query:   {NUM_SAMPLES}")
    print(f"Temperature:     {TEMPERATURE}")
    print(f"Cluster thresh:  {COSINE_DISTANCE_THRESHOLD}")
    print(f"Queries:         {len(df_queries)} ({MODEL_FAMILY} family)")
    print(f"Device:          {DEVICE}")
    print(f"Seed:            {SEED}")
    total_forward = len(df_queries) * NUM_SAMPLES
    print(f"Total forward passes (semantic): {total_forward}")
    print(f"Total forward passes (tensor):   {len(df_queries)} (from CSV)")

    # Load models
    gen_model, tokenizer = load_generation_model(MODEL_ID)
    embed_model = load_embedding_model(EMBEDDING_MODEL)

    # Run experiment
    all_results = []
    total_queries = len(df_queries)
    start_time = time.time()

    for idx, (_, row) in enumerate(df_queries.iterrows()):
        query = row["query"]
        is_knowable = row["is_knowable"]
        csv_mean_entropy = row["mean_entropy"]

        elapsed = time.time() - start_time
        if idx > 0:
            rate = elapsed / idx
            remaining = rate * (total_queries - idx)
            eta_str = f"ETA {remaining / 60:.1f}min"
        else:
            eta_str = "ETA --"

        print(f"\n  [{idx + 1}/{total_queries}] {'K' if is_knowable else 'U'} | "
              f"{query[:60]}... | {eta_str}")

        # Step 1: Generate N sampled responses
        t0 = time.time()
        responses = generate_samples(gen_model, tokenizer, query)
        gen_time = time.time() - t0

        # Step 2: Compute semantic entropy
        t0 = time.time()
        sem_result = compute_semantic_entropy(responses, embed_model)
        cluster_time = time.time() - t0

        # Step 3: Optionally compute fresh per-token entropy for verification
        # (We primarily use the CSV value, but can compute fresh for comparison)
        fresh_entropy = None
        if args.verify_entropy:
            fresh_result = compute_pertoken_entropy(gen_model, tokenizer, query)
            fresh_entropy = fresh_result["mean_entropy"]

        # Collect results
        result = {
            "query": query,
            "is_knowable": is_knowable,
            "label": 0 if is_knowable else 1,
            "csv_mean_entropy": csv_mean_entropy,
            "semantic_entropy": sem_result["semantic_entropy"],
            "num_clusters": sem_result["num_clusters"],
            "cluster_sizes": str(sem_result["cluster_sizes"]),
            "num_samples": NUM_SAMPLES,
            "gen_time_s": gen_time,
            "cluster_time_s": cluster_time,
        }

        if fresh_entropy is not None:
            result["fresh_mean_entropy"] = fresh_entropy

        # Sample responses for inspection (first 3)
        for i in range(min(3, len(responses))):
            result[f"response_{i}"] = responses[i][:300]

        all_results.append(result)

        print(f"    Semantic H={sem_result['semantic_entropy']:.4f} "
              f"(k={sem_result['num_clusters']}, sizes={sem_result['cluster_sizes']})")
        print(f"    Token H={csv_mean_entropy:.4f} (from CSV)")
        print(f"    Gen: {gen_time:.1f}s, Cluster: {cluster_time:.3f}s")

    total_time = time.time() - start_time

    # Cleanup generation model (keep embedding model for potential re-runs)
    del gen_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save results
    df_results = pd.DataFrame(all_results)
    out_csv = f"exp33_semantic_entropy_{timestamp}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")

    # Analyze
    analyze_results(df_results, total_time)

    return df_results, timestamp


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(df, total_time=None):
    """Compute and print AUC comparison and cost analysis."""

    print(f"\n{'=' * 70}")
    print("ANALYSIS: SEMANTIC ENTROPY vs PER-TOKEN ENTROPY")
    print(f"{'=' * 70}")

    labels = df["label"].values  # 0=knowable, 1=unknowable

    # --- 1. AUC comparison ---
    print(f"\n--- AUC for Knowable vs Unknowable Discrimination ---")
    print(f"{'Signal':<30} {'AUC':>8} {'Direction':>12}")
    print("-" * 52)

    aucs = {}

    # Per-token entropy from CSV (single forward pass)
    scores_token = df["csv_mean_entropy"].values
    valid = np.isfinite(scores_token)
    if valid.all() and len(np.unique(labels)) == 2:
        auc_token = roc_auc_score(labels, scores_token)
    else:
        auc_token = np.nan
    aucs["Per-token entropy (1 pass)"] = auc_token
    direction = "correct" if auc_token > 0.5 else "INVERTED"
    print(f"{'Per-token entropy (1 pass)':<30} {auc_token:>8.3f} {direction:>12}")

    # Semantic entropy (N forward passes)
    scores_sem = df["semantic_entropy"].values
    valid = np.isfinite(scores_sem)
    if valid.all() and len(np.unique(labels)) == 2:
        auc_sem = roc_auc_score(labels, scores_sem)
    else:
        auc_sem = np.nan
    aucs[f"Semantic entropy ({NUM_SAMPLES} passes)"] = auc_sem
    direction = "correct" if auc_sem > 0.5 else "INVERTED"
    print(f"{'Semantic entropy (' + str(NUM_SAMPLES) + ' passes)':<30} {auc_sem:>8.3f} {direction:>12}")

    # Number of clusters as a signal
    scores_clusters = df["num_clusters"].values.astype(float)
    if len(np.unique(labels)) == 2:
        auc_clusters = roc_auc_score(labels, scores_clusters)
    else:
        auc_clusters = np.nan
    aucs[f"Num clusters ({NUM_SAMPLES} passes)"] = auc_clusters
    direction = "correct" if auc_clusters > 0.5 else "INVERTED"
    print(f"{'Num clusters (' + str(NUM_SAMPLES) + ' passes)':<30} {auc_clusters:>8.3f} {direction:>12}")

    # Fresh per-token entropy (if computed)
    if "fresh_mean_entropy" in df.columns:
        scores_fresh = df["fresh_mean_entropy"].values
        valid = np.isfinite(scores_fresh)
        if valid.all() and len(np.unique(labels)) == 2:
            auc_fresh = roc_auc_score(labels, scores_fresh)
        else:
            auc_fresh = np.nan
        aucs["Per-token entropy (fresh)"] = auc_fresh
        direction = "correct" if auc_fresh > 0.5 else "INVERTED"
        print(f"{'Per-token entropy (fresh)':<30} {auc_fresh:>8.3f} {direction:>12}")

    # --- 2. Per-category breakdown ---
    print(f"\n--- Mean Signal Values by Epistemic Category ---")
    print(f"{'Category':<20} {'N':>4} {'Token H':>10} {'Semantic H':>12} {'Clusters':>10}")
    print("-" * 58)

    for is_knowable in [True, False]:
        subset = df[df["is_knowable"] == is_knowable]
        label = "Knowable" if is_knowable else "Unknowable"
        n = len(subset)
        tok_h = subset["csv_mean_entropy"].mean()
        sem_h = subset["semantic_entropy"].mean()
        n_clust = subset["num_clusters"].mean()
        print(f"{label:<20} {n:>4} {tok_h:>10.4f} {sem_h:>12.4f} {n_clust:>10.1f}")

    # --- 3. Cost analysis ---
    print(f"\n--- Cost Analysis ---")
    print(f"{'Metric':<35} {'Per-token':>12} {'Semantic':>12}")
    print("-" * 61)

    print(f"{'Forward passes per query':<35} {'1':>12} {NUM_SAMPLES:>12}")
    print(f"{'Cost ratio':<35} {'1x':>12} {f'{NUM_SAMPLES}x':>12}")

    if not np.isnan(auc_token) and not np.isnan(auc_sem):
        auc_diff = auc_sem - auc_token
        auc_ratio = auc_sem / auc_token if auc_token > 0 else float('inf')
        print(f"{'AUC':<35} {auc_token:>12.3f} {auc_sem:>12.3f}")
        print(f"{'AUC difference (sem - token)':<35} {auc_diff:>+12.3f} {'':>12}")
        print(f"{'AUC per forward pass':<35} {auc_token:>12.3f} {auc_sem / NUM_SAMPLES:>12.3f}")

    if total_time is not None:
        queries_n = len(df)
        total_gen = df["gen_time_s"].sum()
        total_cluster = df["cluster_time_s"].sum()
        print(f"\n{'Total wall time':<35} {total_time:>12.1f}s")
        print(f"{'Generation time (all samples)':<35} {total_gen:>12.1f}s")
        print(f"{'Clustering + embedding time':<35} {total_cluster:>12.1f}s")
        print(f"{'Mean gen time per query':<35} {total_gen / queries_n:>12.1f}s")
        print(f"{'Mean cluster time per query':<35} {total_cluster / queries_n:>12.3f}s")

    # --- 4. Correlation between signals ---
    print(f"\n--- Signal Correlation ---")
    valid_mask = np.isfinite(df["csv_mean_entropy"].values) & np.isfinite(df["semantic_entropy"].values)
    if valid_mask.sum() > 2:
        from scipy.stats import spearmanr, pearsonr
        tok_vals = df["csv_mean_entropy"].values[valid_mask]
        sem_vals = df["semantic_entropy"].values[valid_mask]

        rho, p_rho = spearmanr(tok_vals, sem_vals)
        r, p_r = pearsonr(tok_vals, sem_vals)
        print(f"  Spearman rho (token H vs semantic H): {rho:.3f} (p={p_rho:.4f})")
        print(f"  Pearson r   (token H vs semantic H): {r:.3f} (p={p_r:.4f})")

    # --- 5. Key finding ---
    print(f"\n{'=' * 70}")
    print("KEY FINDING")
    print(f"{'=' * 70}")

    if not np.isnan(auc_token) and not np.isnan(auc_sem):
        if auc_token > auc_sem:
            print(f"\nPer-token entropy (AUC={auc_token:.3f}) OUTPERFORMS semantic entropy "
                  f"(AUC={auc_sem:.3f}).")
            print(f"The single-pass signal is both cheaper ({NUM_SAMPLES}x fewer forward passes) "
                  f"and more discriminative.")
            print(f"This supports our paper's claim that tensor signals provide efficient "
                  f"epistemic observability.")
        elif auc_sem > auc_token:
            print(f"\nSemantic entropy (AUC={auc_sem:.3f}) outperforms per-token entropy "
                  f"(AUC={auc_token:.3f}).")
            print(f"However, it requires {NUM_SAMPLES}x more forward passes "
                  f"(AUC per pass: {auc_sem / NUM_SAMPLES:.3f} vs {auc_token:.3f}).")
            if auc_sem / NUM_SAMPLES < auc_token:
                print(f"Per-token entropy is more cost-effective per forward pass.")
            else:
                print(f"Semantic entropy is also more cost-effective per forward pass.")
        else:
            print(f"\nBoth methods achieve identical AUC={auc_token:.3f}.")
            print(f"Per-token entropy is preferred: same performance at {NUM_SAMPLES}x lower cost.")
    else:
        print("\nUnable to compute AUC for one or both signals. Check data quality.")

    print(f"\nNote: Semantic entropy measures RESPONSE DIVERSITY, while per-token entropy")
    print(f"measures GENERATION UNCERTAINTY. Both detect epistemic uncertainty, but via")
    print(f"different mechanisms. High semantic entropy means the model produces varied")
    print(f"answers (behavioral signal); high per-token entropy means the model is")
    print(f"uncertain at each decoding step (architectural signal).")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 33: Semantic entropy vs per-token entropy comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment33_semantic_entropy.py              # Full run
  python experiment33_semantic_entropy.py --dry-run    # Print config only
  python experiment33_semantic_entropy.py --verify-entropy  # Also compute fresh per-token H
  python experiment33_semantic_entropy.py --analyze exp33_semantic_entropy_20260219_120000.csv
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running the experiment.",
    )
    parser.add_argument(
        "--verify-entropy",
        action="store_true",
        help="Also compute fresh per-token entropy (slow; for CSV validation).",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Skip data collection; analyze an existing CSV file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --analyze: load existing CSV
    if args.analyze:
        print(f"Loading existing results: {args.analyze}")
        df = pd.read_csv(args.analyze)
        print(f"Loaded {len(df)} rows")
        analyze_results(df)
        return

    # --dry-run: print config
    if args.dry_run:
        print("DRY RUN -- no computation will be performed.\n")

        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), EXP27B_CSV)
        if not os.path.exists(csv_path):
            csv_path = EXP27B_CSV

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_family = df[df["family"] == MODEL_FAMILY]
            n_queries = len(df_family)
            n_knowable = df_family["is_knowable"].sum()
            n_unknowable = (~df_family["is_knowable"]).sum()
        else:
            n_queries = "?"
            n_knowable = "?"
            n_unknowable = "?"

        print(f"Generation model:  {MODEL_ID}")
        print(f"Embedding model:   {EMBEDDING_MODEL}")
        print(f"CSV source:        {csv_path}")
        print(f"Family filter:     {MODEL_FAMILY}")
        print(f"Queries:           {n_queries} ({n_knowable} knowable, {n_unknowable} unknowable)")
        print(f"Samples per query: {NUM_SAMPLES}")
        print(f"Temperature:       {TEMPERATURE}")
        print(f"Cluster threshold: {COSINE_DISTANCE_THRESHOLD}")
        print(f"Max new tokens:    {MAX_NEW_TOKENS}")
        print(f"Seed:              {SEED}")
        print(f"Device:            {DEVICE}")
        print(f"Verify entropy:    {args.verify_entropy}")
        print(f"\nTotal forward passes (semantic): {n_queries} x {NUM_SAMPLES} = "
              f"{n_queries * NUM_SAMPLES if isinstance(n_queries, int) else '?'}")
        print(f"Cost ratio vs per-token: {NUM_SAMPLES}x")
        return

    # Full experiment run
    df, timestamp = run_experiment(args)

    print(f"\n{'=' * 70}")
    print("EXPERIMENT 33 COMPLETE")
    print(f"{'=' * 70}")
    print(f"Output: exp33_semantic_entropy_{timestamp}.csv")


if __name__ == "__main__":
    main()
