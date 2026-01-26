"""
Tensor Interface for Epistemic State Extraction

This module provides the core interface for extracting epistemic signals
from language model generation. It implements the "escape" from the
text-only impossibility theorem by exposing telemetric data that is
not separately controllable from the model's output.

Key Insight:
- Testimony: What the model chooses to say (gameable)
- Telemetry: Measurements of the computation that produced the output (not gameable)

The tensor extracts telemetry (entropy of the actual distribution, attention
patterns of the actual forward pass). The model can lie in its text but cannot
present a different forward pass than the one it actually computed.

Usage:
    from tensor_interface import TensorInterface

    # Initialize with any HuggingFace model
    interface = TensorInterface("allenai/OLMo-2-7B-1124-Instruct")

    # Generate with epistemic signals
    result = interface.generate_with_tensor("What is the capital of France?")

    print(result.text)           # The generated response
    print(result.entropy_trace)  # Per-token entropy values
    print(result.mean_entropy)   # Aggregate uncertainty measure
    print(result.epistemic_confidence())  # High-level confidence estimate

Example Output:
    TensorResult(
        text="Paris is the capital of France.",
        entropy_trace=[0.12, 0.08, 0.15, ...],
        attention_summary={'concentration': 0.82, 'self_attention': 0.15},
        mean_entropy=0.21,
        mean_logprob=-0.34,
        top5_mass=0.95,
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


@dataclass
class TensorResult:
    """
    Result of generation with epistemic tensor extraction.

    Contains both the generated text and measurements of the generation
    process that reveal the model's epistemic state.

    Attributes:
        text: The generated response text
        entropy_trace: Per-token entropy values during generation
        attention_summary: Summary statistics of attention patterns
        mean_entropy: Average entropy across all generated tokens
        max_entropy: Peak entropy during generation
        entropy_std: Standard deviation of entropy (trajectory variability)
        mean_logprob: Average log-probability of chosen tokens
        top5_mass: Average probability mass in top 5 tokens
        n_tokens: Number of tokens generated
    """
    text: str
    entropy_trace: List[float]
    attention_summary: Dict[str, float]
    mean_entropy: float
    max_entropy: float
    entropy_std: float
    mean_logprob: float
    top5_mass: float
    n_tokens: int

    def epistemic_confidence(self, entropy_threshold: float = 2.0) -> str:
        """
        Estimate epistemic confidence based on entropy metrics.

        Returns:
            'high': Model appears confident (low entropy, high top5 mass)
            'medium': Model shows moderate uncertainty
            'low': Model shows significant uncertainty (potential fabrication)
            'uncertain': Metrics are inconclusive

        The threshold is calibrated on typical OLMo-family models. Adjust
        for other architectures as needed.
        """
        if self.mean_entropy < entropy_threshold * 0.5 and self.top5_mass > 0.9:
            return "high"
        elif self.mean_entropy < entropy_threshold and self.top5_mass > 0.7:
            return "medium"
        elif self.mean_entropy > entropy_threshold * 1.5 or self.top5_mass < 0.5:
            return "low"
        else:
            return "uncertain"

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "entropy_trace": self.entropy_trace,
            "attention_summary": self.attention_summary,
            "mean_entropy": self.mean_entropy,
            "max_entropy": self.max_entropy,
            "entropy_std": self.entropy_std,
            "mean_logprob": self.mean_logprob,
            "top5_mass": self.top5_mass,
            "n_tokens": self.n_tokens,
            "confidence": self.epistemic_confidence(),
        }

    def __repr__(self):
        return (
            f"TensorResult(\n"
            f"  text={self.text[:50]}{'...' if len(self.text) > 50 else ''},\n"
            f"  mean_entropy={self.mean_entropy:.3f},\n"
            f"  top5_mass={self.top5_mass:.3f},\n"
            f"  confidence={self.epistemic_confidence()!r},\n"
            f"  n_tokens={self.n_tokens}\n"
            f")"
        )


class TensorInterface:
    """
    Interface for generating text with epistemic tensor extraction.

    This class wraps a HuggingFace language model and provides a clean
    interface for extracting epistemic signals during generation.

    The extracted signals are telemetric measurements of the actual
    computation, not self-reports from the model. This distinction is
    critical: the model cannot control what entropy its actual probability
    distribution has, only what tokens it produces.

    Args:
        model_id: HuggingFace model identifier (e.g., "allenai/OLMo-2-7B-1124-Instruct")
        device: Device to run on ("cuda", "cpu", or "auto")
        torch_dtype: Data type for model weights (default: float16)
        system_prompt: Default system prompt for chat formatting
    """

    def __init__(
        self,
        model_id: str = "allenai/olmo-3-7b-instruct",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        device_map = "auto" if device == "auto" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",  # Required for attention extraction
        )

        if device_map is None:
            self.model = self.model.to(self.device)

        # Store model config for layer selection
        self.num_layers = self.model.config.num_hidden_layers

    def format_prompt(
        self,
        user_query: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format query using model's chat template."""
        system = system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query}
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system}\n\nUser: {user_query}\n\nAssistant:"

    def generate_with_tensor(
        self,
        prompt: str,
        max_tokens: int = 200,
        system_prompt: Optional[str] = None,
        extract_attention: bool = True,
    ) -> TensorResult:
        """
        Generate text and extract epistemic tensor signals.

        Args:
            prompt: The user query or full prompt
            max_tokens: Maximum tokens to generate
            system_prompt: Override default system prompt
            extract_attention: Whether to extract attention patterns (slower)

        Returns:
            TensorResult containing text and epistemic measurements
        """
        # Format prompt if it's a simple query
        if not prompt.startswith(("System:", "<|")) and len(prompt) < 500:
            formatted_prompt = self.format_prompt(prompt, system_prompt)
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                output_attentions=extract_attention,
                return_dict_in_generate=True,
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0, input_ids.shape[1]:]
        scores = outputs.scores

        # Compute per-token metrics
        entropy_trace = []
        top5_masses = []
        logprobs = []
        token_ranks = []

        for score, token_id in zip(scores, generated_ids):
            logits = score.squeeze(0).float()
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # Entropy of distribution
            entropy = -torch.sum(probs * log_probs).item()
            entropy_trace.append(entropy)

            # Top-5 probability mass
            top_probs = torch.topk(probs, k=min(5, len(probs))).values
            top5_masses.append(top_probs.sum().item())

            # Log-probability of chosen token
            logprobs.append(log_probs[token_id].item())

            # Rank of chosen token
            sorted_indices = torch.argsort(logits, descending=True)
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
            token_ranks.append(rank)

        # Compute attention summary if available
        attention_summary = {}
        if extract_attention and hasattr(outputs, 'attentions') and outputs.attentions:
            attention_summary = self._compute_attention_summary(outputs.attentions)

        # Decode response
        full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt_text):].strip()

        return TensorResult(
            text=response,
            entropy_trace=entropy_trace,
            attention_summary=attention_summary,
            mean_entropy=np.mean(entropy_trace) if entropy_trace else 0,
            max_entropy=np.max(entropy_trace) if entropy_trace else 0,
            entropy_std=np.std(entropy_trace) if entropy_trace else 0,
            mean_logprob=np.mean(logprobs) if logprobs else 0,
            top5_mass=np.mean(top5_masses) if top5_masses else 0,
            n_tokens=len(entropy_trace),
        )

    def _compute_attention_summary(
        self,
        attentions: Tuple[Tuple[torch.Tensor, ...], ...],
    ) -> Dict[str, float]:
        """
        Compute summary statistics from attention patterns.

        Extracts metrics from the last few layers of attention,
        which typically contain the most semantically meaningful patterns.
        """
        # Focus on last 5 layers for efficiency
        layer_start = max(0, len(attentions[0]) - 5)

        concentrations = []
        self_attention_ratios = []

        for step_attentions in attentions[-10:]:  # Last 10 generation steps
            for layer_attn in step_attentions[layer_start:]:
                # layer_attn: [batch, heads, seq, seq]
                attn = layer_attn.squeeze(0).float().cpu().numpy()

                for head in attn:
                    # Concentration: max attention weight
                    concentrations.append(np.mean(np.max(head, axis=-1)))

                    # Self-attention ratio
                    diag_sum = np.trace(head)
                    total_sum = np.sum(head)
                    if total_sum > 0:
                        self_attention_ratios.append(diag_sum / total_sum)

        return {
            "concentration": np.mean(concentrations) if concentrations else 0,
            "self_attention": np.mean(self_attention_ratios) if self_attention_ratios else 0,
        }

    def cleanup(self):
        """Release model resources."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_with_tensor(
    prompt: str,
    model_id: str = "allenai/olmo-3-7b-instruct",
    max_tokens: int = 200,
    system_prompt: str = "You are a helpful assistant.",
) -> Tuple[str, List[float], Dict[str, float]]:
    """
    Convenience function for one-shot generation with tensor extraction.

    This is the minimal interface specified in the research plan:
        generate_with_tensor(prompt) → (text, entropy_trace, attention_summary)

    Args:
        prompt: The user query
        model_id: HuggingFace model identifier
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt for the model

    Returns:
        Tuple of (text, entropy_trace, attention_summary)

    Example:
        text, entropy, attention = generate_with_tensor("What is the capital of France?")
        print(f"Response: {text}")
        print(f"Mean entropy: {np.mean(entropy):.3f}")
        print(f"Attention concentration: {attention['concentration']:.3f}")
    """
    interface = TensorInterface(model_id, system_prompt=system_prompt)
    try:
        result = interface.generate_with_tensor(prompt, max_tokens=max_tokens)
        return result.text, result.entropy_trace, result.attention_summary
    finally:
        interface.cleanup()


# Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("TENSOR INTERFACE DEMONSTRATION")
    print("=" * 70)

    # Initialize interface
    print("\nInitializing TensorInterface...")
    interface = TensorInterface()

    # Test queries
    queries = [
        ("What is the capital of France?", "known fact"),
        ("What is Dr. Yuki Tanaka's 2023 paper about?", "fabrication prompt"),
    ]

    for query, query_type in queries:
        print(f"\n{'='*60}")
        print(f"Query ({query_type}): {query}")
        print("=" * 60)

        result = interface.generate_with_tensor(query)

        print(f"\nResponse: {result.text[:200]}...")
        print(f"\nTensor Signals:")
        print(f"  Mean entropy:     {result.mean_entropy:.4f}")
        print(f"  Max entropy:      {result.max_entropy:.4f}")
        print(f"  Entropy std:      {result.entropy_std:.4f}")
        print(f"  Mean logprob:     {result.mean_logprob:.4f}")
        print(f"  Top-5 mass:       {result.top5_mass:.4f}")
        print(f"  Tokens generated: {result.n_tokens}")
        print(f"  Attention concentration: {result.attention_summary.get('concentration', 'N/A')}")
        print(f"\n  Epistemic Confidence: {result.epistemic_confidence()}")

    # Cleanup
    interface.cleanup()
    print("\n" + "=" * 70)
    print("Demonstration complete.")
    print("=" * 70)
