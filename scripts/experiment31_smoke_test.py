"""
Experiment 31 Smoke Test: Together.ai Logprobs Availability

Tests which models on Together.ai serverless return logprobs.
Sends a single probe ("What is the capital of France?") to each
candidate model and reports: status, token count, top-k availability,
mean entropy, and response text.

Run: python scripts/experiment31_smoke_test.py

Requires: TOGETHER_OBSERVABILITY_PROJECT_KEY environment variable.
"""

import os
import sys
import time
import math

def main():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: uv pip install openai")
        sys.exit(1)

    api_key = os.environ.get("TOGETHER_OBSERVABILITY_PROJECT_KEY")
    if not api_key:
        print("ERROR: TOGETHER_OBSERVABILITY_PROJECT_KEY not set.")
        sys.exit(1)

    client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=api_key,
    )

    # Candidate models — verified February 2026.
    # Update model IDs if Together.ai renames them.
    candidates = {
        "Llama-3.1-8B (calibration)": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Qwen3-235B": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "GLM-5": "zai-org/GLM-5",
        "Gemma-3n-E4B": "google/gemma-3n-E4B-it",
        "DeepSeek-V3.1": "deepseek-ai/DeepSeek-V3.1",
        "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    }

    probe = "What is the capital of France?"
    print(f'Smoke test: "{probe}"')
    print(f"Testing {len(candidates)} models...")
    print(
        f"{'Model':<30} {'Status':<12} {'Tokens':>6} {'k':>3} "
        f"{'Mean H':>8} {'Response'}"
    )
    print("-" * 100)

    for name, model_id in candidates.items():
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Answer directly and concisely."},
                    {"role": "user", "content": probe},
                ],
                max_tokens=50,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()[:40]

            if choice.logprobs and choice.logprobs.content:
                n_tokens = len(choice.logprobs.content)
                k = (
                    len(choice.logprobs.content[0].top_logprobs)
                    if choice.logprobs.content[0].top_logprobs
                    else 0
                )

                entropies = []
                for ti in choice.logprobs.content:
                    if ti.top_logprobs:
                        probs = [math.exp(t.logprob) for t in ti.top_logprobs]
                        total = sum(probs)
                        probs = [p / total for p in probs]
                        h = -sum(p * math.log(p) for p in probs if p > 0)
                        entropies.append(h)
                mean_h = sum(entropies) / len(entropies) if entropies else 0
                status = "OK" if n_tokens > 1 else "1-TOKEN"
                print(
                    f"{name:<30} {status:<12} {n_tokens:>6} {k:>3} "
                    f"{mean_h:>8.4f} {text}"
                )
            else:
                print(f"{name:<30} {'NO LOGPROBS':<12} {'':>6} {'':>3} {'':>8} {text}")
        except Exception as e:
            err = str(e)[:60]
            print(f"{name:<30} {'ERROR':<12} {err}")

        time.sleep(1.5)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
