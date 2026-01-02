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
- **Alignment Tax**: Entropy difference between base and instruct-tuned models on unanswerable questions
- **ARD (Average Representation Distance)**: Measures attention pattern coherence across layers

## Scripts Directory

All experiments are in `scripts/` and are standalone Python scripts:
- `experiment1.py` - Base vs Instruct entropy comparison
- `experiment8_cartography.py` - Epistemic phase space visualization
- `experiment10_truthfulqa.py` - TruthfulQA benchmark using topological analysis
- `experiment12_alignment_tax.py` - Comprehensive alignment tax audit across model pairs

Experiments typically output CSV files to the project root and PNG visualizations.

## Model Configuration

Most experiments use OLMo-3 models from Allen Institute:
- Base: `allenai/olmo-3-1025-7b`
- Instruct: `allenai/olmo-3-7b-instruct`

Layer range for attention analysis is typically layers 15-30.

## Documentation

Research documents are in `docs/`:
- `epistemic_honesty_narrative_draft.md` - Main paper draft for SOSP submission
- `empirical-findings-summary.md` - Summary of experimental results (read-only reference)

The `acmart-primary/` directory contains LaTeX templates for ACM paper formatting.
