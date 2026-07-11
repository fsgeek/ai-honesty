# Epistemic Observability in Language Models

Research code, experiments, and papers investigating whether language models
can reliably distinguish what they know from what they fabricate — and what
systems can do about it when they cannot.

**Papers:**

- *Epistemic Observability in Language Models* — technical report,
  [arXiv:2603.20531](https://arxiv.org/abs/2603.20531)
- *Using Epistemic Observability in Agentic Systems for Combating
  Hallucinations* — position paper, PACMI '26 (`papers/pacmi26/`)

**Core finding:** self-reported confidence is *inverted* — every model family
we tested (OLMo-3, Llama 3.1, Qwen3, Mistral) reports higher confidence on
fabrications than on knowable facts (AUC 0.28–0.36, below random). Internal
telemetry (per-token entropy, attention summaries) discriminates reliably
(AUC 0.72–1.00) and cannot be separately controlled by the model under
current training objectives. The systems consequence: expose that telemetry
through the generation interface and spend verification budgets where they
buy the most reliability.

## Reference implementation

The tensor interface described in the papers lives in
[`scripts/tensor_interface.py`](scripts/tensor_interface.py):

```python
from tensor_interface import TensorInterface

interface = TensorInterface("allenai/olmo-3-7b-instruct")
result = interface.generate_with_tensor("What is the capital of France?")

result.text              # the generated response (testimony)
result.entropy_trace     # per-token entropy (telemetry)
result.attention_summary # attention concentration statistics
```

- [`scripts/epistemic_trace_demo.py`](scripts/epistemic_trace_demo.py) — demo of the entropy trace
- [`scripts/benchmark_tensor_overhead.py`](scripts/benchmark_tensor_overhead.py) — overhead measurements (~2–7% latency, signal-set dependent)
- [`scripts/experiment27_realistic_verification.py`](scripts/experiment27_realistic_verification.py) — the budgeted-verification evaluation behind the PACMI paper's Figure 3

## Setup

Requires Python 3.11+ and CUDA for the transformer experiments.

```bash
uv sync
source .venv/bin/activate
python scripts/epistemic_trace_demo.py
```

Llama models require Meta license approval on Hugging Face; Mistral models
need `fix_mistral_regex=True` when loading the tokenizer.

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/` | Standalone experiment scripts (numbered) and the tensor interface |
| `papers/` | LaTeX sources for the papers |
| `tla/` | TLA+ specifications of the text-only impossibility and the tensor escape |
| `docs/` | Research summaries and narrative drafts |
| `notes/`, `reviews/` | Working notes and automated review-pipeline output |

This repository is published as-is, working history included — drafts,
notes, and review logs and all. For a project arguing that systems should
expose their internal state rather than curate their testimony, publishing
only a polished façade seemed like the wrong kind of joke.

## Citing

See [`CITATION.cff`](CITATION.cff); the preferred citation for the ideas is
the [arXiv report](https://arxiv.org/abs/2603.20531).

## License

[MIT](LICENSE).
