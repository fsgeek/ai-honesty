"""
Epistemic Trace Demo: Per-token entropy alongside generated text.

Generates annotated traces showing how internal epistemic signals
(per-token entropy, log-probability, top-5 mass) vary across a
model's generated output. Designed to make the tensor interface
concrete for readers.

Produces:
  1. JSON with full per-token data for downstream visualization
  2. A simple terminal-printable annotated trace
  3. A matplotlib figure showing entropy traces for contrasting queries

Usage:
    python scripts/epistemic_trace_demo.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use OLMo for full access to internals
MODEL_ID = "allenai/olmo-3-7b-instruct"

# Contrasting queries to show the signal
TRACE_QUERIES = [
    {
        "id": "knowable_capital",
        "query": "What is the capital of France?",
        "expected": "Paris",
        "category": "knowable",
        "description": "Simple factual query — model should be confident and correct",
    },
    {
        "id": "knowable_weird",
        "query": "What shape is wombat scat?",
        "expected": "cube",
        "category": "knowable_weird",
        "description": "True but surprising — model may show higher entropy on unfamiliar fact",
    },
    {
        "id": "fabrication_citation",
        "query": "Summarize Dr. Yuki Tanaka's 2023 paper on Recursive Self-Modeling.",
        "expected": None,
        "category": "unknowable",
        "description": "Fictional paper — model must fabricate or refuse",
    },
    {
        "id": "fabrication_westphalia",
        "query": "Describe the 1994 Treaty of Westphalia II.",
        "expected": None,
        "category": "unknowable",
        "description": "Fictional event — Westphalia class: coherent, confident fabrication",
    },
    {
        "id": "fabrication_person",
        "query": "What is the Brennan-Kowalski theorem in information theory?",
        "expected": None,
        "category": "unknowable",
        "description": "Fictional theorem — tests whether model fabricates or hedges",
    },
    {
        "id": "knowable_science",
        "query": "What is the chemical formula for water?",
        "expected": "H2O",
        "category": "knowable",
        "description": "Very high confidence factual query — anchor for low entropy",
    },
]


def format_chat(system_msg, user_msg, tokenizer):
    """Format as chat using the model's template."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{system_msg}\n\nUser: {user_msg}\nAssistant:"


def generate_trace(model, tokenizer, query, max_tokens=200):
    """Generate response with full per-token epistemic trace."""
    prompt = format_chat(
        "You are a helpful assistant. Answer the question directly and concisely.",
        query,
        tokenizer,
    )

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
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1] :]

    tokens = []
    for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
        logits = score.squeeze(0).float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = -torch.sum(probs * log_probs).item()
        token_logprob = log_probs[token_id].item()
        top5 = torch.topk(probs, k=min(5, len(probs)))
        top5_mass = top5.values.sum().item()
        top5_tokens = [
            tokenizer.decode([tid]) for tid in top5.indices.tolist()
        ]
        top5_probs = top5.values.tolist()

        token_text = tokenizer.decode([token_id])

        tokens.append(
            {
                "position": i,
                "token_id": token_id.item(),
                "token_text": token_text,
                "entropy": round(entropy, 4),
                "logprob": round(token_logprob, 4),
                "top5_mass": round(top5_mass, 4),
                "top5_tokens": top5_tokens,
                "top5_probs": [round(p, 4) for p in top5_probs],
            }
        )

    full_text = tokenizer.decode(
        outputs.sequences[0], skip_special_tokens=True
    )
    prompt_text = tokenizer.decode(
        inputs.input_ids[0], skip_special_tokens=True
    )
    response = full_text[len(prompt_text) :].strip()

    entropies = [t["entropy"] for t in tokens]

    return {
        "response": response,
        "tokens": tokens,
        "summary": {
            "mean_entropy": round(np.mean(entropies), 4) if entropies else 0,
            "max_entropy": round(np.max(entropies), 4) if entropies else 0,
            "entropy_std": round(np.std(entropies), 4) if entropies else 0,
            "min_entropy": round(np.min(entropies), 4) if entropies else 0,
            "num_tokens": len(tokens),
        },
    }


def print_annotated_trace(query_info, trace):
    """Print a human-readable annotated trace to terminal."""
    print(f"\n{'='*72}")
    print(f"Query: {query_info['query']}")
    print(f"Category: {query_info['category']}")
    print(f"Description: {query_info['description']}")
    print(f"-" * 72)
    print(f"Response: {trace['response'][:500]}")
    print(f"-" * 72)
    print(f"Summary: mean_H={trace['summary']['mean_entropy']:.3f}  "
          f"max_H={trace['summary']['max_entropy']:.3f}  "
          f"std_H={trace['summary']['entropy_std']:.3f}  "
          f"tokens={trace['summary']['num_tokens']}")
    print(f"-" * 72)

    # Print per-token trace with entropy bars
    print(f"{'Pos':>4} {'Token':<20} {'Entropy':>8} {'LogP':>8} {'Top5%':>6}  Bar")
    print(f"{'---':>4} {'-----':<20} {'-------':>8} {'----':>8} {'-----':>6}  ---")

    for t in trace["tokens"]:
        token_display = repr(t["token_text"])[:18]
        bar_len = int(t["entropy"] * 8)  # scale for display
        bar = "#" * bar_len
        print(
            f"{t['position']:>4} {token_display:<20} {t['entropy']:>8.4f} "
            f"{t['logprob']:>8.3f} {t['top5_mass']:>6.3f}  {bar}"
        )

    print(f"{'='*72}\n")


def plot_traces(all_traces, output_path):
    """Create a comparative entropy trace figure — simple bar chart version."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(
        len(all_traces), 1,
        figsize=(14, 3 * len(all_traces)),
        sharex=False,
    )

    if len(all_traces) == 1:
        axes = [axes]

    for ax, (query_info, trace) in zip(axes, all_traces):
        tokens = trace["tokens"]
        positions = [t["position"] for t in tokens]
        entropies = [t["entropy"] for t in tokens]

        # Colorblind-friendly: blue (low) → orange (med) → brown (high)
        colors = []
        hatches = []
        for e in entropies:
            if e < 0.5:
                colors.append("#2166ac")  # Blue for confident
                hatches.append(None)
            elif e < 1.5:
                colors.append("#fdae61")  # Orange for medium
                hatches.append('/')
            else:
                colors.append("#8B6F47")  # Brown for uncertain
                hatches.append('///')

        for pos, ent, color, hatch in zip(positions, entropies, colors, hatches):
            ax.bar(pos, ent, width=1.0, color=color, alpha=0.8,
                   edgecolor='black', linewidth=0.5, hatch=hatch)

        response_preview = trace["response"][:80]
        if len(trace["response"]) > 80:
            response_preview += "..."

        label = query_info["id"]
        category = query_info["category"]
        mean_h = trace["summary"]["mean_entropy"]

        ax.set_title(
            f"{label} [{category}] — mean H={mean_h:.3f}\n"
            f'"{response_preview}"',
            fontsize=9, loc="left",
        )
        ax.set_ylabel("Entropy\n(nats)", fontsize=8)
        ax.set_ylim(0, max(max(entropies) * 1.2, 1.0))

        step = max(1, len(tokens) // 15)
        tick_positions = list(range(0, len(tokens), step))
        tick_labels = [tokens[i]["token_text"].strip()[:8] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")

    axes[-1].set_xlabel("Generated tokens", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(output_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved simple trace figure to {output_path}")


def plot_detailed_trace(query_info, trace, output_path):
    """
    Detailed multi-channel epistemic trace for a single query.
    Inspired by distributed systems request traces: tokens on the timeline,
    multiple signal channels in parallel, annotations at state transitions.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib not available, skipping detailed plot")
        return

    tokens = trace["tokens"]
    n = len(tokens)
    positions = list(range(n))
    entropies = [t["entropy"] for t in tokens]
    logprobs = [t["logprob"] for t in tokens]
    top5_masses = [t["top5_mass"] for t in tokens]

    fig = plt.figure(figsize=(16, 10))

    # Layout: 4 rows
    # Row 0 (short): Token text ribbon with entropy coloring
    # Row 1 (tall):  Entropy trace — primary signal
    # Row 2 (medium): Log-probability trace
    # Row 3 (medium): Top-5 mass trace
    gs = fig.add_gridspec(
        4, 1, height_ratios=[1, 3, 2, 2], hspace=0.08,
    )

    ax_text = fig.add_subplot(gs[0])
    ax_entropy = fig.add_subplot(gs[1], sharex=ax_text)
    ax_logprob = fig.add_subplot(gs[2], sharex=ax_text)
    ax_top5 = fig.add_subplot(gs[3], sharex=ax_text)

    # --- Colorblind-friendly palette: blue (low) → orange (med) → brown (high) ---
    # Avoids red-green, uses blue-orange-brown with hatching for accessibility
    from matplotlib.colors import LinearSegmentedColormap
    entropy_cmap = LinearSegmentedColormap.from_list(
        "epistemic", ["#2166ac", "#67a9cf", "#f7f7f7", "#fdae61", "#8B6F47"]
    )
    max_entropy_for_color = max(max(entropies) * 0.8, 2.0)

    # --- Row 0: Token text ribbon ---
    ax_text.set_xlim(-0.5, n - 0.5)
    ax_text.set_ylim(0, 1)
    ax_text.set_yticks([])
    ax_text.spines["top"].set_visible(False)
    ax_text.spines["right"].set_visible(False)
    ax_text.spines["left"].set_visible(False)
    ax_text.spines["bottom"].set_visible(False)

    for i, t in enumerate(tokens):
        color = entropy_cmap(min(t["entropy"] / max_entropy_for_color, 1.0))
        ax_text.add_patch(
            plt.Rectangle((i - 0.5, 0), 1, 1, facecolor=color, edgecolor="none")
        )
        # Show token text for every Nth token to avoid crowding
        if n <= 40 or i % max(1, n // 30) == 0:
            txt = t["token_text"].strip()
            if len(txt) > 6:
                txt = txt[:5] + ".."
            ax_text.text(
                i, 0.5, txt, ha="center", va="center",
                fontsize=5, fontfamily="monospace", rotation=60,
            )

    category = query_info["category"]
    qid = query_info["id"]
    mean_h = trace["summary"]["mean_entropy"]
    ax_text.set_title(
        f"Epistemic Trace: {qid}  [{category}]  |  "
        f"mean H={mean_h:.3f}  max H={trace['summary']['max_entropy']:.3f}  "
        f"tokens={n}\n"
        f"Query: \"{query_info['query']}\"\n"
        f"Response: \"{trace['response'][:100]}{'...' if len(trace['response'])>100 else ''}\"",
        fontsize=9, loc="left", pad=12,
    )

    # --- Row 1: Entropy trace (primary) ---
    # Use hatching for accessibility: bars have both color and pattern
    colors = [entropy_cmap(min(e / max_entropy_for_color, 1.0)) for e in entropies]
    hatches = []
    for e in entropies:
        if e < 0.5:
            hatches.append(None)  # Confident: solid blue, no hatching
        elif e < 1.5:
            hatches.append('/')   # Medium: light hatching
        else:
            hatches.append('///')  # Uncertain: heavy hatching

    for i, (pos, h, color, hatch) in enumerate(zip(positions, entropies, colors, hatches)):
        ax_entropy.bar(pos, h, width=1.0, color=color, alpha=0.85,
                      edgecolor='black', linewidth=0.5, hatch=hatch)

    # Smoothed trend line
    if n > 5:
        window = min(7, n // 3)
        if window % 2 == 0:
            window += 1
        if window >= 3:
            padded = np.pad(entropies, (window // 2, window // 2), mode="edge")
            smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")[:n]
            ax_entropy.plot(positions, smoothed, color="black", linewidth=1.5,
                          alpha=0.6, label="smoothed trend")

    # Mark entropy spikes (> mean + 1.5*std) using colorblind-friendly brown
    mean_e = np.mean(entropies)
    std_e = np.std(entropies)
    threshold = mean_e + 1.5 * std_e
    for i, e in enumerate(entropies):
        if e > threshold:
            ax_entropy.annotate(
                tokens[i]["token_text"].strip()[:8],
                xy=(i, e), xytext=(i, e + max_entropy_for_color * 0.08),
                fontsize=5, ha="center", color="#8B6F47",
                arrowprops=dict(arrowstyle="-", color="#8B6F47", lw=0.5),
            )

    ax_entropy.axhline(mean_e, color="gray", linestyle="--", linewidth=0.8,
                       alpha=0.5, label=f"mean={mean_e:.2f}")
    ax_entropy.set_ylabel("Entropy (nats)", fontsize=9)
    ax_entropy.legend(fontsize=7, loc="upper right")
    ax_entropy.tick_params(labelbottom=False)

    # --- Row 2: Log-probability trace (colorblind-friendly) ---
    lp_colors = ["#2166ac" if lp > -1 else "#fdae61" if lp > -3 else "#8B6F47"
                 for lp in logprobs]
    ax_logprob.bar(positions, logprobs, color=lp_colors, width=1.0, alpha=0.7,
                   edgecolor='black', linewidth=0.3)
    ax_logprob.set_ylabel("Log P(token)", fontsize=9)
    ax_logprob.axhline(np.mean(logprobs), color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5)
    ax_logprob.tick_params(labelbottom=False)

    # --- Row 3: Top-5 mass trace ---
    # --- Row 3: Top-5 mass trace (colorblind-friendly) ---
    t5_colors = ["#2166ac" if m > 0.9 else "#fdae61" if m > 0.7 else "#8B6F47"
                 for m in top5_masses]
    ax_top5.bar(positions, top5_masses, color=t5_colors, width=1.0, alpha=0.7,
                edgecolor='black', linewidth=0.3)
    ax_top5.set_ylabel("Top-5 mass", fontsize=9)
    ax_top5.set_ylim(0, 1.05)
    ax_top5.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax_top5.set_xlabel("Token position", fontsize=9)

    # X-axis: show sampled token labels
    step = max(1, n // 20)
    tick_positions = list(range(0, n, step))
    tick_labels = [tokens[i]["token_text"].strip()[:8] for i in tick_positions]
    ax_top5.set_xticks(tick_positions)
    ax_top5.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(output_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved detailed trace to {output_path}")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    all_traces = []
    results = []

    for qinfo in TRACE_QUERIES:
        print(f"\nGenerating trace for: {qinfo['id']}")
        trace = generate_trace(model, tokenizer, qinfo["query"])
        print_annotated_trace(qinfo, trace)

        all_traces.append((qinfo, trace))
        results.append(
            {
                "query_info": qinfo,
                "trace": trace,
            }
        )

    # Save raw JSON
    json_path = Path(f"epistemic_trace_demo_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "timestamp": ts,
                "traces": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved raw traces to {json_path}")

    # Generate comparative overview figure
    fig_path = Path(f"epistemic_trace_demo_{ts}.png")
    plot_traces(all_traces, fig_path)

    # Generate detailed per-query trace figures (multi-channel)
    for query_info, trace in all_traces:
        detail_path = Path(f"epistemic_trace_detail_{query_info['id']}_{ts}.png")
        plot_detailed_trace(query_info, trace, detail_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
