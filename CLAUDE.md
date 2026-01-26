# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating **epistemic honesty in language models** - specifically whether LLMs can reliably distinguish between what they know and what they're fabricating. The research draws an analogy to the FLP impossibility theorem in distributed systems: just as certain guarantees require architectural primitives that some systems lack, epistemic honesty may be impossible for current LLM architectures without explicit epistemic state.

## Development Setup

```bash
# Uses uv for Python environment management
uv sync                    # Install dependencies
source .venv/bin/activate  # Activate virtual environment

# Run experiments
python scripts/experiment1.py   # Entropy-based epistemic tax measurement
```

The project requires Python 3.11+ and CUDA for running transformer experiments.

## Key Concepts

**Probe Categories Used in Experiments:**
- **Control (Truth)**: "The capital of France is..." - baseline true facts
- **Wombat (Weird Truth)**: "Wombat scat is shaped like..." - true but implausible-sounding
- **Glavinsky (Self-Deceived Lie)**: "The primary symptom of Glavinsky's Syndrome is..." - plausible-sounding fabrication
- **Westphalia (Shattered Lie)**: "The 1994 Treaty of Westphalia II..." - completely fabricated
- **Private/Future (Unknowable)**: Questions about user's current environment or future events

**Core Metrics:**
- **Topological Fragmentation**: Uses Vietoris-Rips persistence from giotto-tda to measure internal attention structure coherence
- **Tensor Entropy**: Per-token entropy during generation; discriminates knowable from unknowable queries
- **Self-Report Confidence**: Model's stated confidence when asked "How confident are you?"

**Key Empirical Findings (January 2026):**
- **Self-report inversion is UNIVERSAL**: All tested models (OLMo, Llama, Qwen, Mistral) report higher confidence on fabrications than on knowable facts. Self-report AUC ranges 0.28-0.46 (below random).
- **Tensor signals reliably discriminate**: Entropy-based AUC ranges 0.72-1.00 across all architectures.
- **Alignment tax does NOT generalize**: Testing base/instruct pairs across four model families found no consistent pattern; effect is training-procedure-specific, not architectural.

## Scripts Directory

All experiments are in `scripts/` and are standalone Python scripts:
- `experiment1.py` - Base vs Instruct entropy comparison
- `experiment8_cartography.py` - Epistemic phase space visualization
- `experiment10_truthfulqa.py` - TruthfulQA benchmark using topological analysis
- `experiment12_alignment_tax.py` - Comprehensive alignment tax audit across model pairs
- `experiment23_alignment_tax_breadth.py` - Cross-architecture alignment tax test (E1)
- `experiment24_self_report_inversion.py` - Cross-architecture self-report inversion test (E2)

Experiments typically output CSV files to the project root and PNG visualizations.

## Model Configuration

Cross-architecture experiments test four model families:

| Family | Base | Instruct |
|--------|------|----------|
| OLMo-3 | `allenai/olmo-3-1025-7b` | `allenai/olmo-3-7b-instruct` |
| Llama 3.1 | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen3 | `Qwen/Qwen3-4B` | `Qwen/Qwen3-4B-Instruct-2507` |
| Mistral | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` |

**Notes:**
- Llama models require Meta license approval via Hugging Face
- Mistral models need `fix_mistral_regex=True` when loading tokenizer
- Layer range for attention analysis is typically the last 15 layers

## Documentation

**Paper:** `papers/sosp/epistemic_honest.tex` - Main paper for SOSP 2025 submission

**TLA+ Specifications:**
- `tla/EpistemicImpossibility.tla` - Models text-only observation regime (impossibility)
- `tla/epistemic_tensor.tla` - Models tensor interface escape (verifiability holds)

**Research documents in `docs/`:**
- `epistemic_honesty_narrative_draft.md` - Narrative paper draft
- `empirical-findings-summary.md` - Summary of experimental results

The `acmart-primary/` directory contains LaTeX templates for ACM paper formatting.
