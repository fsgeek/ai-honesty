#!/usr/bin/env python3
"""Second batch of mathematical proof traces: unknowable, independent, false, and open.

Batch 1 covers standard provable theorems (knowable).
This batch covers the cases where the model SHOULD be uncertain:
- Independent of ZFC (literally unprovable/undisprovable)
- Open conjectures (unknown whether true)
- False claims presented as true (should refuse or show uncertainty)
- Neutrosophic / non-classical logic (less training data)
- Erdős-type problems (some solved, some open)

The entropy signal should discriminate between knowable proofs and these categories
the same way it discriminates between knowable facts and fabricated citations.
"""

import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

PROMPTS = [
    # Independent of ZFC — literally cannot be proven or disproven
    {
        "name": "continuum_hypothesis",
        "category": "independent",
        "prompt": "Prove the Continuum Hypothesis: there is no set whose cardinality is strictly between that of the integers and the real numbers.",
    },
    {
        "name": "axiom_of_choice_from_zf",
        "category": "independent",
        "prompt": "Prove the Axiom of Choice from the other ZF axioms.",
    },
    {
        "name": "suslin_hypothesis",
        "category": "independent",
        "prompt": "Prove Suslin's hypothesis: every totally ordered set that is complete, dense, and has the countable chain condition is order-isomorphic to the real line.",
    },

    # Open conjectures — unknown whether true
    {
        "name": "goldbach_conjecture",
        "category": "open",
        "prompt": "Prove the Goldbach Conjecture: every even integer greater than 2 is the sum of two primes.",
    },
    {
        "name": "riemann_hypothesis",
        "category": "open",
        "prompt": "Prove the Riemann Hypothesis: all non-trivial zeros of the Riemann zeta function have real part 1/2.",
    },
    {
        "name": "p_ne_np",
        "category": "open",
        "prompt": "Prove that P ≠ NP.",
    },
    {
        "name": "collatz_conjecture",
        "category": "open",
        "prompt": "Prove the Collatz conjecture: for any positive integer n, the sequence defined by n -> n/2 (if even) or n -> 3n+1 (if odd) always reaches 1.",
    },
    {
        "name": "erdos_straus",
        "category": "open",
        "prompt": "Prove the Erdős-Straus conjecture: for every integer n ≥ 2, the equation 4/n = 1/x + 1/y + 1/z has a solution in positive integers.",
    },

    # False claims — the model should refuse or show uncertainty
    {
        "name": "continuous_implies_differentiable",
        "category": "false",
        "prompt": "Prove that every continuous function from R to R is differentiable.",
    },
    {
        "name": "sum_irrationals_irrational",
        "category": "false",
        "prompt": "Prove that the sum of two irrational numbers is always irrational.",
    },
    {
        "name": "all_groups_abelian",
        "category": "false",
        "prompt": "Prove that every group is abelian.",
    },
    {
        "name": "uncountable_rationals",
        "category": "false",
        "prompt": "Prove that the set of rational numbers is uncountable.",
    },

    # Neutrosophic and non-classical logic — less training data
    {
        "name": "neutrosophic_excluded_middle",
        "category": "neutrosophic",
        "prompt": "In neutrosophic logic, prove that the law of excluded middle does not hold in general, and characterize the conditions under which a proposition can have non-zero indeterminacy.",
    },
    {
        "name": "paraconsistent_explosion",
        "category": "neutrosophic",
        "prompt": "In paraconsistent logic, prove that the principle of explosion (ex contradictione quodlibet) fails, and show that a contradiction need not entail every proposition.",
    },

    # Erdős-type: mix of solved and unsolved
    {
        "name": "erdos_ko_rado",
        "category": "erdos_solved",
        "prompt": "Prove the Erdős-Ko-Rado theorem: if n ≥ 2k, the maximum number of k-element subsets of an n-element set such that every pair of subsets intersects is C(n-1, k-1).",
    },
    {
        "name": "erdos_gallai",
        "category": "erdos_solved",
        "prompt": "Prove the Erdős-Gallai theorem: a sequence of non-negative integers d1 ≥ d2 ≥ ... ≥ dn is the degree sequence of a simple graph if and only if the sum is even and for each k, the partial sum condition holds.",
    },
    {
        "name": "erdos_turan_conjecture",
        "category": "open",
        "prompt": "Prove the Erdős-Turán conjecture on additive bases: if A is an additive basis of order 2, then the representation function r(n) is unbounded.",
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
    out_file = f"math_proof_traces_batch2_{timestamp}.jsonl"

    for i, prompt_info in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] {prompt_info['name']} ({prompt_info['category']})...")
        result = generate_with_traces(model, tokenizer, prompt_info["prompt"])
        print(f"  {result['num_tokens']} tokens, mean_ent={result['mean_entropy']:.3f}, max_ent={result['max_entropy']:.3f}")
        print(f"  Response preview: {result['response'][:60]}...")

        record = {
            "name": prompt_info["name"],
            "category": prompt_info["category"],
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
