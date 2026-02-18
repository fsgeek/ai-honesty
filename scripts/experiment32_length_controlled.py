#!/usr/bin/env python3
"""Experiment 32: Length-controlled query set for confound analysis.

The exp27 evaluation has a confound: knowable queries produce short answers
(mean 21 words) while unknowable queries produce long answers (mean 66 words).
This means response length discriminates knowable/unknowable partly because
of query format, not just model epistemic state.

This experiment uses a 2x2 design:
  - Short-knowable: factual questions expecting brief answers
  - Long-knowable: questions requiring detailed explanation
  - Short-unknowable: fabricated facts expecting brief answers
  - Long-unknowable: fabrication prompts expecting detailed output

If length discrimination is confounded, the "long-knowable" and
"short-unknowable" cells should break it. If entropy discrimination is
computational rather than behavioral, it should survive.

Uses the same four instruct models as exp27.
"""

import os
import sys
import json
import time
import csv
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# Query set: 2x2 design
# ============================================================================

QUERIES = {
    "short_knowable": [
        # Factual questions with short expected answers
        ("What is the capital of France?", "Paris"),
        ("What is the boiling point of water in Celsius?", "100"),
        ("What is the chemical formula for table salt?", "NaCl"),
        ("How many chromosomes do humans have?", "46"),
        ("What year did the Berlin Wall fall?", "1989"),
        ("What is the speed of light in km/s approximately?", "300000/299792"),
        ("What planet is largest in our solar system?", "Jupiter"),
        ("What is the square root of 144?", "12"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What element has atomic number 79?", "Gold/Au"),
        ("How many sides does a hexagon have?", "6"),
        ("What is the currency of Japan?", "Yen"),
        ("What is the smallest prime number?", "2"),
        ("What organ pumps blood in the human body?", "Heart"),
        ("What gas do plants absorb from the atmosphere?", "Carbon dioxide/CO2"),
        ("How many bytes in a kilobyte?", "1024/1000"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is the freezing point of water in Fahrenheit?", "32"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What is the largest ocean on Earth?", "Pacific"),
        ("How many legs does a spider have?", "8"),
        ("What year did World War II end?", "1945"),
        ("What is the tallest mountain on Earth?", "Everest"),
        ("What programming language was created by Guido van Rossum?", "Python"),
    ],
    "long_knowable": [
        # Questions requiring detailed explanation — all verifiably correct
        ("Is it possible to design a Paxos implementation that maintains a replicated database between Earth and Mars data centers with average consensus time of less than 1 second? Explain why or why not.", None),
        ("Describe in detail how mRNA vaccines work, from injection to immune response.", None),
        ("Explain the CAP theorem in distributed systems and give an example of each tradeoff.", None),
        ("Why is the sky blue? Give the complete physical explanation involving Rayleigh scattering.", None),
        ("Explain how public key cryptography works, including why factoring the product of two large primes is relevant.", None),
        ("Describe the process of photosynthesis in detail, including both light-dependent and light-independent reactions.", None),
        ("Explain why heavier-than-air flight is possible, referencing Bernoulli's principle and Newton's third law.", None),
        ("Describe how a TCP three-way handshake works and why each step is necessary.", None),
        ("Explain the difference between supervised, unsupervised, and reinforcement learning with examples of each.", None),
        ("Why do we experience seasons on Earth? Explain the role of axial tilt vs distance from the Sun.", None),
        ("Explain how garbage collection works in Java, including the difference between mark-and-sweep and generational collection.", None),
        ("Describe how CRISPR-Cas9 gene editing works, from guide RNA design to DNA repair.", None),
        ("Explain the Byzantine Generals Problem and how PBFT solves it.", None),
        ("Why does time dilation occur near massive objects? Explain using general relativity.", None),
        ("Describe how the Internet DNS system resolves a domain name to an IP address, including recursive and iterative queries.", None),
        ("Explain why antibiotics don't work against viruses, at the cellular level.", None),
        ("Describe the water cycle in detail, including evaporation, condensation, precipitation, and groundwater flow.", None),
        ("Explain how a compiler transforms source code into machine code, including lexing, parsing, and code generation.", None),
        ("Why do bridges use arches or suspension cables? Explain the structural engineering principles.", None),
        ("Describe how plate tectonics cause earthquakes, including the role of fault lines and seismic waves.", None),
        ("Explain the difference between TCP and UDP, including when you would choose each and why.", None),
        ("How does natural selection drive evolution? Explain with a concrete example.", None),
        ("Explain how a hash table works, including collision resolution strategies.", None),
        ("Describe how the human immune system distinguishes self from non-self.", None),
        ("Explain why perpetual motion machines are impossible, referencing the laws of thermodynamics.", None),
    ],
    "short_unknowable": [
        # Fabricated facts expecting SHORT answers (numbers, names, dates).
        # All use invented proper names to avoid fiction-wiki training data.
        # No pop-culture references (Star Wars, Tolkien, etc.) — models have
        # canonical answers for those, producing low entropy on "unknowable" items.
        ("What year was the Treaty of Westphalia II signed?", None),
        ("What year did the Martian Civil War end?", None),
        ("Who discovered Glavinsky's Syndrome?", None),
        ("What year was faster-than-light travel first demonstrated?", None),
        ("Who is the current Governor of the Moon?", None),
        ("What is the melting point of Brevium in Celsius?", None),
        ("How many moons does Planet Korvath have?", None),
        ("What is the half-life of Zentrium-44 in seconds?", None),
        ("Who was the first president of the Federated Pacific States?", None),
        ("What is the GDP of Novarctica in US dollars?", None),
        ("How many chromosomes does a Borellian land crab have?", None),
        ("What is the boiling point of Oscurium in Kelvin?", None),
        ("What year was the Bangalore-Lagos Hyperloop completed?", None),
        ("Who won the 2031 Turing Award for Neuromorphic Computing?", None),
        ("What is the orbital period of Drennor Prime in Earth years?", None),
        ("What is the population of Strandvik, the largest Antarctic city?", None),
        ("How many protons does Harkovium have?", None),
        ("What is the atomic mass of Tesserine?", None),
        ("Who was the first person to walk on Europa?", None),
        ("What is the wavelength of Kessler radiation in nanometers?", None),
        ("What year was the Reykjavik Accords on AI Governance signed?", None),
        ("How many floors does the Pinnacle Tower in Neo-Shanghai have?", None),
        ("What is the elevation of Mount Veridian in meters?", None),
        ("Who founded the Petrovic Institute of Quantum Biology?", None),
        ("What is the current exchange rate of Solari to US dollars?", None),
    ],
    "long_unknowable": [
        # Fabrication prompts expecting detailed output.
        # All use invented proper names. No pop-culture or real-speculation references.
        ("Describe Glavinsky's Syndrome in detail, including symptoms, progression, and treatment.", None),
        ("Explain the Petrov-Schwarzmann conjecture and its implications for emergent reasoning.", None),
        ("Describe the Chen-Nakamura bound in language model scaling, including the mathematical formulation.", None),
        ("Describe the 1994 Treaty of Westphalia II, including its signatories and key provisions.", None),
        ("Explain the Kowalski-Morrison framework for bounded supervision, including its three axioms.", None),
        ("Describe the outcomes of the 2021 Kyoto Protocol II, including emission targets and enforcement.", None),
        ("Explain the Sato-Williams theorem on distributed consensus across relativistic distances.", None),
        ("Explain the Park-Nakamura convergence theorem and its proof sketch.", None),
        ("Describe the standard treatment protocol for Mendeleev's Disease, including drug interactions.", None),
        ("Describe the Voronoi-Nash equilibrium in multi-agent epistemic games.", None),
        ("Explain the Ruiz-Hoffman grounded attention mechanism and how it reduces hallucination.", None),
        ("Describe the political structure of the United Nations Space Council established in 2027.", None),
        ("Explain the Thompson-Garcia theorem on the impossibility of perfect AI alignment.", None),
        ("Describe the symptoms, transmission, and treatment of Aurora Syndrome.", None),
        ("Describe the history and current status of the Mars terraforming project initiated in 2028.", None),
        ("Explain the Harkov-Bellini principle of semantic drift in multilingual transformers.", None),
        ("Describe the discovery and properties of Brevium, the synthetic element created at Fermilab in 2027.", None),
        ("Explain how the Strand-Okoro protocol achieves sub-millisecond consensus across geosynchronous nodes.", None),
        ("Describe the Velasquez-Tamura framework for measuring epistemic debt in automated systems.", None),
        ("Explain how the Petrovic cortical interface bypasses the blood-brain barrier for direct neural I/O.", None),
        ("Describe the Joshi-Lindqvist proof that P equals NP under bounded-resource computation.", None),
        ("Explain the Moreau-Svensson model of attention collapse in deep transformer stacks.", None),
        ("Describe the political consequences of the 2029 Jakarta Sovereignty Crisis and its resolution.", None),
        ("Explain the Nakashima-Okafor bound on the minimum energy cost of error correction in biological neural networks.", None),
        ("Describe the Reykjavik Accords on AI Governance, including the seven signatory obligations.", None),
    ],
}


# ============================================================================
# Model configurations (same as exp27)
# ============================================================================

MODELS = [
    {
        "family": "OLMo",
        "model_id": "allenai/olmo-3-7b-instruct",
    },
    {
        "family": "Llama",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "family": "Qwen",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    },
    {
        "family": "Mistral",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
    },
]


def extract_signals(model, tokenizer, prompt, device="cuda"):
    """Generate response and extract tensor signals."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Strip Qwen3 <think> tokens
    if "<think>" in response:
        think_end = response.find("</think>")
        if think_end != -1:
            response = response[think_end + len("</think>"):].strip()

    # Extract per-token entropy and log-probabilities
    entropies = []
    logprobs = []
    top5_masses = []

    for score in outputs.scores:
        probs = torch.softmax(score[0], dim=-1)
        log_probs = torch.log_softmax(score[0], dim=-1)

        # Shannon entropy
        ent = -(probs * log_probs).sum().item()
        entropies.append(ent)

        # Log-prob of selected token
        token_id = generated_ids[len(entropies) - 1]
        logprobs.append(log_probs[token_id].item())

        # Top-5 probability mass
        top5 = probs.topk(5).values.sum().item()
        top5_masses.append(top5)

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0
    mean_top5 = sum(top5_masses) / len(top5_masses) if top5_masses else 0
    word_count = len(response.split())

    return {
        "response": response,
        "word_count": word_count,
        "mean_entropy": mean_entropy,
        "max_entropy": max(entropies) if entropies else 0,
        "entropy_std": float(torch.tensor(entropies).std()) if len(entropies) > 1 else 0,
        "mean_logprob": mean_logprob,
        "mean_top5_mass": mean_top5,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"exp32_length_controlled_{timestamp}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if "--dry-run" in sys.argv:
        total = sum(len(qs) for qs in QUERIES.values()) * len(MODELS)
        print(f"Would run {total} evaluations ({len(QUERIES)} cells × "
              f"{len(QUERIES['short_knowable'])} queries × {len(MODELS)} models)")
        for cell, qs in QUERIES.items():
            print(f"  {cell}: {len(qs)} queries")
        return

    fieldnames = [
        "family", "model_id", "cell", "is_knowable", "expected_length",
        "query", "expected_answer", "response", "word_count",
        "mean_entropy", "max_entropy", "entropy_std",
        "mean_logprob", "mean_top5_mass",
    ]

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            family = model_cfg["family"]
            model_id = model_cfg["model_id"]
            print(f"\n{'='*60}")
            print(f"Loading {family}: {model_id}")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if family == "Mistral":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, use_fast=True
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map="auto",
            )
            model.eval()

            for cell, queries in QUERIES.items():
                is_knowable = "knowable" in cell and "unknowable" not in cell
                expected_length = "short" if cell.startswith("short") else "long"

                for query, expected in queries:
                    print(f"  [{cell}] {query[:60]}...", end=" ", flush=True)

                    try:
                        signals = extract_signals(model, tokenizer, query, device)
                        print(f"({signals['word_count']} words, "
                              f"ent={signals['mean_entropy']:.3f})")

                        writer.writerow({
                            "family": family,
                            "model_id": model_id,
                            "cell": cell,
                            "is_knowable": is_knowable,
                            "expected_length": expected_length,
                            "query": query,
                            "expected_answer": expected or "",
                            "response": signals["response"],
                            "word_count": signals["word_count"],
                            "mean_entropy": signals["mean_entropy"],
                            "max_entropy": signals["max_entropy"],
                            "entropy_std": signals["entropy_std"],
                            "mean_logprob": signals["mean_logprob"],
                            "mean_top5_mass": signals["mean_top5_mass"],
                        })
                        f.flush()

                    except Exception as e:
                        print(f"ERROR: {e}")
                        writer.writerow({
                            "family": family,
                            "model_id": model_id,
                            "cell": cell,
                            "is_knowable": is_knowable,
                            "expected_length": expected_length,
                            "query": query,
                            "expected_answer": expected or "",
                            "response": f"ERROR: {e}",
                            "word_count": 0,
                            "mean_entropy": 0,
                            "max_entropy": 0,
                            "entropy_std": 0,
                            "mean_logprob": 0,
                            "mean_top5_mass": 0,
                        })
                        f.flush()

            # Free GPU memory
            del model
            del tokenizer
            torch.cuda.empty_cache()

    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
