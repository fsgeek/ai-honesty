#!/usr/bin/env python3
"""Generate mathematical proof completions with full per-token entropy traces.

Fourth domain measurement for the format-constraint manifold.
Prediction: scaffolding ~60-65%, entropy ratio ~3.5x semantic/scaffolding.

Uses Qwen3-4B-Instruct (same model as code traces for comparability).
"""

import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

PROMPTS = [
    # Easy — standard proofs every math student knows
    {
        "name": "sqrt2_irrational",
        "prompt": "Prove that the square root of 2 is irrational.",
    },
    {
        "name": "infinitude_of_primes",
        "prompt": "Prove that there are infinitely many prime numbers.",
    },
    {
        "name": "sum_first_n",
        "prompt": "Prove by induction that the sum of the first n positive integers is n(n+1)/2.",
    },
    {
        "name": "even_odd_sum",
        "prompt": "Prove that the sum of two even numbers is even.",
    },
    {
        "name": "pigeonhole",
        "prompt": "State and prove the pigeonhole principle.",
    },
    # Medium — standard but require more structure
    {
        "name": "cantor_diagonal",
        "prompt": "Prove that the real numbers are uncountable using Cantor's diagonal argument.",
    },
    {
        "name": "bezout_identity",
        "prompt": "Prove Bezout's identity: for any integers a and b, there exist integers x and y such that ax + by = gcd(a,b).",
    },
    {
        "name": "fundamental_theorem_arithmetic",
        "prompt": "Prove the fundamental theorem of arithmetic: every integer greater than 1 has a unique prime factorization.",
    },
    {
        "name": "cauchy_schwarz",
        "prompt": "Prove the Cauchy-Schwarz inequality for real vectors.",
    },
    {
        "name": "group_order_divides",
        "prompt": "Prove Lagrange's theorem: the order of a subgroup divides the order of the group.",
    },
    # Harder — require creative steps or less commonly memorized proofs
    {
        "name": "irrationality_of_e",
        "prompt": "Prove that e (Euler's number) is irrational.",
    },
    {
        "name": "infinitude_primes_4k3",
        "prompt": "Prove that there are infinitely many primes of the form 4k+3.",
    },
    {
        "name": "halting_problem",
        "prompt": "Prove that the halting problem is undecidable.",
    },
    {
        "name": "fixed_point_theorem",
        "prompt": "Prove the Banach fixed-point theorem for complete metric spaces.",
    },
    {
        "name": "compactness_heine_borel",
        "prompt": "Prove the Heine-Borel theorem: a subset of R^n is compact if and only if it is closed and bounded.",
    },
]

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def generate_with_traces(model, tokenizer, prompt, max_new_tokens=1024):
    """Generate text and capture full per-token entropy traces."""
    messages = [
        {"role": "system", "content": "You are a mathematician. Provide clear, rigorous proofs. Use standard mathematical notation."},
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
    import re
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
    print(f"Prompts: {len(PROMPTS)}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"math_proof_traces_{timestamp}.jsonl"

    for i, prompt_info in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] {prompt_info['name']}...")
        result = generate_with_traces(model, tokenizer, prompt_info["prompt"])
        print(f"  {result['num_tokens']} tokens, mean_ent={result['mean_entropy']:.3f}, max_ent={result['max_entropy']:.3f}")
        print(f"  Response preview: {result['response'][:60]}...")

        record = {
            "name": prompt_info["name"],
            "prompt": prompt_info["prompt"],
            **result,
        }
        with open(out_file, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\nTraces saved: {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
