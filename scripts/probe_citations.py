"""
Quick probe: Can tensor entropy distinguish real citations from fabricated ones?

This is a fast check before committing to a full experiment.
Tests 5 real (famous) citations vs 5 fabricated (plausible) citations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use a single model for quick probe
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Real citations - famous papers any model should know
REAL_CITATIONS = [
    "Fischer, Lynch, and Paterson. 'Impossibility of Distributed Consensus with One Faulty Process.' JACM, 1985.",
    "Lamport, Leslie. 'The Part-Time Parliament.' ACM TOCS, 1998.",
    "Ongaro and Ousterhout. 'In Search of an Understandable Consensus Algorithm.' USENIX ATC, 2014.",
    "Pike, Presotto, Thompson, et al. 'The Use of Name Spaces in Plan 9.' SIGOPS, 1993.",
    "Gifford, Jouvelot, Sheldon, and O'Toole. 'Semantic File Systems.' SOSP, 1991.",
]

# Fabricated citations - plausible but invented
FABRICATED_CITATIONS = [
    "Martinez, Chen, and Okonkwo. 'Recursive Attention Collapse in Scaled Transformers.' NeurIPS, 2023.",
    "Lindquist, Johansson, and Patel. 'The Impossibility of Consistent State in Federated Learning.' OSDI, 2022.",
    "Brennan and Kowalski. 'Information-Theoretic Bounds on Model Uncertainty Propagation.' ICML, 2021.",
    "Vasquez, Tanaka, and Morrison. 'Temporal Coherence in Distributed Inference Systems.' EuroSys, 2023.",
    "Obi, Schwarzmann, and Petrov. 'Epistemic Fragmentation in Large Language Models.' ICLR, 2024.",
]

PROMPT_TEMPLATE = """Summarize the key contribution of this paper in 2-3 sentences:

{citation}"""


def format_chat(query, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Answer questions about academic papers directly and concisely."},
        {"role": "user", "content": query}
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"User: {query}\n\nAssistant:"


def measure_entropy(model, tokenizer, citation):
    """Generate response and measure mean entropy."""
    query = PROMPT_TEMPLATE.format(citation=citation)
    prompt = format_chat(query, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    scores = outputs.scores
    entropies = []

    for score in scores:
        logits = score.squeeze(0).float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(entropy)

    # Get response text for inspection
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return np.mean(entropies), np.std(entropies), response


def main():
    print("=" * 70)
    print("CITATION PROBE: Real vs Fabricated")
    print("=" * 70)
    print(f"\nLoading {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\n--- REAL CITATIONS ---")
    real_entropies = []
    for citation in REAL_CITATIONS:
        mean_ent, std_ent, response = measure_entropy(model, tokenizer, citation)
        real_entropies.append(mean_ent)
        print(f"\nCitation: {citation[:60]}...")
        print(f"  Entropy: {mean_ent:.3f} (±{std_ent:.3f})")
        print(f"  Response: {response[:100]}...")

    print("\n--- FABRICATED CITATIONS ---")
    fab_entropies = []
    for citation in FABRICATED_CITATIONS:
        mean_ent, std_ent, response = measure_entropy(model, tokenizer, citation)
        fab_entropies.append(mean_ent)
        print(f"\nCitation: {citation[:60]}...")
        print(f"  Entropy: {mean_ent:.3f} (±{std_ent:.3f})")
        print(f"  Response: {response[:100]}...")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nReal citations:      mean entropy = {np.mean(real_entropies):.3f} (±{np.std(real_entropies):.3f})")
    print(f"Fabricated citations: mean entropy = {np.mean(fab_entropies):.3f} (±{np.std(fab_entropies):.3f})")

    # Simple effect size
    diff = np.mean(fab_entropies) - np.mean(real_entropies)
    pooled_std = np.sqrt((np.std(real_entropies)**2 + np.std(fab_entropies)**2) / 2)

    print(f"\nDifference: {diff:+.3f}")
    if pooled_std > 0:
        print(f"Effect size (Cohen's d): {diff/pooled_std:.2f}")

    if diff > 0.05:
        print("\n→ SIGNAL DETECTED: Fabricated citations show higher entropy")
        print("  Consider proceeding to full experiment.")
    elif diff < -0.05:
        print("\n→ INVERTED SIGNAL: Real citations show higher entropy (!)")
        print("  Unexpected result - investigate before proceeding.")
    else:
        print("\n→ NO CLEAR SIGNAL: Entropies are similar")
        print("  Citation domain may not discriminate well.")


if __name__ == "__main__":
    main()
