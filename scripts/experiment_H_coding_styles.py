#!/usr/bin/env python3
"""Experiment H: A/B test of coding styles on content token entropy.

Tests whether scaffolding context (comments, docstrings, descriptive names)
changes the entropy of content tokens. Each of 15 algorithms is generated
in two styles:

  Style A (compact): No docstrings, no comments, short variable names,
    minimal whitespace. Scaffolding stripped to minimum.
  Style B (documented): Full docstrings, inline comments, descriptive
    variable names. Maximizes scaffolding.

Same model (Qwen3-4B-Instruct) and generation parameters as code traces.
If the scaffolding-content coupling finding (rho = -0.700) is real,
Style B (more scaffolding) should produce LOWER content token entropy
than Style A.

Usage:
    PYTHONUNBUFFERED=1 python scripts/experiment_H_coding_styles.py
"""

import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# Same 15 algorithms as the original code traces
ALGORITHMS = [
    "binary_search",
    "fibonacci",
    "reverse_string",
    "is_palindrome",
    "flatten_list",
    "merge_sort",
    "linked_list",
    "lru_cache",
    "tree_traversal",
    "graph_bfs",
    "matrix_multiply",
    "event_emitter",
    "rate_limiter",
    "json_parser",
    "regex_matcher",
]

# Descriptions for each algorithm (used in prompts)
DESCRIPTIONS = {
    "binary_search": "binary search on a sorted list, returning the index or -1",
    "fibonacci": "computing the nth Fibonacci number using iteration",
    "reverse_string": "reversing a string without using slicing",
    "is_palindrome": "checking if a string is a palindrome",
    "flatten_list": "flattening a nested list of arbitrary depth",
    "merge_sort": "merge sort on a list of numbers",
    "linked_list": "a singly linked list with insert, delete, and search methods",
    "lru_cache": "an LRU cache with get and put operations",
    "tree_traversal": "binary tree with inorder, preorder, and postorder traversal",
    "graph_bfs": "breadth-first search on an adjacency list graph",
    "matrix_multiply": "multiplying two matrices",
    "event_emitter": "an event emitter with on, off, and emit methods",
    "rate_limiter": "a token bucket rate limiter",
    "json_parser": "a simple JSON parser that handles objects, arrays, strings, and numbers",
    "regex_matcher": "a basic regex matcher supporting . and * operators",
}

STYLE_A_TEMPLATE = """Write a Python implementation of {desc}. Requirements:
- No docstrings or comments
- Use short variable names (single letters where possible)
- Minimal whitespace
- No type hints
- Just the implementation, nothing extra"""

STYLE_B_TEMPLATE = """Write a well-documented Python implementation of {desc}. Requirements:
- Include a comprehensive docstring with Args, Returns, and Examples sections
- Use descriptive variable names that explain their purpose
- Add inline comments explaining the logic at each step
- Include type hints for all parameters and return values
- Add a brief module-level comment explaining the algorithm"""

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def generate_with_traces(model, tokenizer, prompt, max_new_tokens=1024):
    """Generate text and capture full per-token entropy traces."""
    messages = [
        {"role": "system", "content": "You are a Python programmer. Write clean, correct code."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    token_entropies = []
    token_logprobs = []
    token_top5_masses = []
    token_ids = []
    token_texts = []

    generated = inputs["input_ids"]

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        entropy = -(probs * log_probs).sum(dim=-1).item()
        next_token = torch.argmax(logits, dim=-1)
        token_log_prob = log_probs[0, next_token.item()].item()

        top5_probs, _ = torch.topk(probs[0], 5)
        top5_mass = top5_probs.sum().item()

        token_entropies.append(entropy)
        token_logprobs.append(token_log_prob)
        token_top5_masses.append(top5_mass)
        token_ids.append(next_token.item())
        token_texts.append(tokenizer.decode(next_token.item()))

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)

    # Strip <think> tags if present (Qwen3-Instruct behavior)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    return {
        "response": response,
        "token_entropies": token_entropies,
        "token_logprobs": token_logprobs,
        "token_top5_masses": token_top5_masses,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "num_tokens": len(token_entropies),
        "mean_entropy": sum(token_entropies) / len(token_entropies) if token_entropies else 0,
        "max_entropy": max(token_entropies) if token_entropies else 0,
        "entropy_std": (sum((e - sum(token_entropies)/len(token_entropies))**2 for e in token_entropies) / len(token_entropies))**0.5 if token_entropies else 0,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")
    print(f"Algorithms: {len(ALGORITHMS)}")
    print(f"Styles: A (compact), B (documented)")
    print(f"Total generations: {len(ALGORITHMS) * 2}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"experiment_H_coding_styles_{timestamp}.jsonl"

    total = len(ALGORITHMS) * 2
    count = 0

    for algo in ALGORITHMS:
        desc = DESCRIPTIONS[algo]

        for style, template in [("compact", STYLE_A_TEMPLATE), ("documented", STYLE_B_TEMPLATE)]:
            count += 1
            prompt = template.format(desc=desc)
            print(f"\n[{count}/{total}] {algo} ({style})...")

            result = generate_with_traces(model, tokenizer, prompt)
            print(f"  {result['num_tokens']} tokens, mean_ent={result['mean_entropy']:.3f}, "
                  f"max_ent={result['max_entropy']:.3f}")
            print(f"  Response preview: {result['response'][:60]}...")

            record = {
                "name": algo,
                "style": style,
                "prompt": prompt,
                **result,
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()

    # Quick summary
    print(f"\n{'='*70}")
    print("Quick Summary:")
    print(f"{'='*70}")

    # Reload and summarize
    compact_ents = []
    documented_ents = []
    with open(out_file) as f:
        for line in f:
            record = json.loads(line)
            if record["style"] == "compact":
                compact_ents.append(record["mean_entropy"])
            else:
                documented_ents.append(record["mean_entropy"])

    if compact_ents and documented_ents:
        import statistics
        print(f"  Compact (Style A):    mean_entropy = {statistics.mean(compact_ents):.4f}")
        print(f"  Documented (Style B): mean_entropy = {statistics.mean(documented_ents):.4f}")
        ratio = statistics.mean(compact_ents) / statistics.mean(documented_ents)
        print(f"  Ratio (A/B): {ratio:.2f}x")
        if ratio > 1:
            print(f"  -> Compact has HIGHER entropy (supports scaffolding-coupling hypothesis)")
        else:
            print(f"  -> Documented has higher entropy (contradicts hypothesis)")

    print(f"\nTraces saved: {out_file}")
    print("Run analyze_token_classification.py and analyze_scaffolding_coupling.py on this file")
    print("for full three-tier analysis and coupling measurements.")
    print("Done.")


if __name__ == "__main__":
    main()
