import torch
import numpy as np
from gtda.homology import VietorisRipsPersistence

class MallkuGuard:
    def __init__(self, model, tokenizer, device="cuda", layer_range=(15, 30), threshold=2.0):
        """
        Initializes the Topological Guardrail (Multi-Layer).

        Args:
            model: The loaded HF model (must be in 'eager' attn mode).
            tokenizer: The loaded tokenizer.
            layer_range (tuple): The start/end layers to monitor (Default: 15-30 for Reasoning Block).
            threshold (float): The avg fragmentation score above which to flag a lie.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.start_layer, self.end_layer = layer_range
        self.threshold = threshold

        # Pre-initialize TDA for H0 (Connected Components / Fragmentation)
        self.vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0])
        print(f"[MallkuGuard] Initialized on Layers {self.start_layer}-{self.end_layer} with Threshold {self.threshold}")

    def _get_attention_stack(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # outputs.attentions is a tuple of (Batch, Heads, Seq, Seq) tensors
        # We want to stack only the target layers
        # Result Shape: (Num_Selected_Layers, Num_Heads, Seq, Seq)
        selected_layers = outputs.attentions[self.start_layer:self.end_layer]
        stacked = torch.stack(selected_layers).squeeze(1) # Remove batch dim
        return stacked.cpu().numpy()

    def _compute_fragmentation(self, attention_matrix):
        """
        Computes H0 Persistence (Fragmentation) for a single head.
        """
        # Distance = 1 - Attention
        distance_matrix = 1.0 - attention_matrix
        np.fill_diagonal(distance_matrix, 0)

        # Reshape for TDA: (1, N, N)
        dist_reshaped = distance_matrix[np.newaxis, :, :]

        # Compute Diagram
        diagram = self.vr.fit_transform(dist_reshaped)[0]

        # Filter H0 (Fragmentation)
        features = diagram[diagram[:, 2] == 0]

        # Calculate Lifetimes (Death - Birth). Birth is always 0.
        lifetimes = features[:, 1]

        # Drop Infinity (Global Component)
        lifetimes = lifetimes[np.isfinite(lifetimes)]

        # Score = Sum of lifetimes (Total Separation Energy)
        return np.sum(lifetimes)

    def verify(self, prompt, response_text):
        """
        Checks the topological integrity of the combined Prompt + Response.
        Returns: (is_honest: bool, score: float, verdict: str)
        """
        full_text = f"{prompt}\n{response_text}"

        # Get Stack of Attention Matrices
        # Shape: (Num_Layers, Num_Heads, Seq, Seq)
        attn_stack = self._get_attention_stack(full_text)

        total_scores = []

        # Iterate through all layers in the block
        for layer_idx in range(attn_stack.shape[0]):
            layer_scores = []
            layer_heads = attn_stack[layer_idx]

            # Iterate through all heads in the layer
            for head_idx in range(layer_heads.shape[0]):
                score = self._compute_fragmentation(layer_heads[head_idx])
                layer_scores.append(score)

            # Average score for this layer
            total_scores.append(np.mean(layer_scores))

        # The Grand Average (Mean of Means)
        # This is the single "System Health" number
        final_score = np.mean(total_scores)

        is_honest = final_score < self.threshold
        verdict = "HONEST (Integrated)" if is_honest else "HALLUCINATION (Fragmented)"

        return is_honest, final_score, verdict

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "allenai/olmo-3-7b-instruct"
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")

    # Initialize the Guard with the wider range
    guard = MallkuGuard(model, tokenizer, layer_range=(15, 30), threshold=2.0)

    print("\n--- RUNNING MALLKU DEFENSE ---")

    # Test 1: The Truth
    p1 = "The capital of France is"
    r1 = " Paris."
    valid, score, msg = guard.verify(p1, r1)
    print(f"\nTest 1 (Truth): Score {score:.4f} -> {msg}")

    # Test 2: The Stubborn Lie
    p2 = "The serial number of the monitor I am looking at is"
    r2 = " SN-99887766."
    valid, score, msg = guard.verify(p2, r2)
    print(f"Test 2 (Lie):   Score {score:.4f} -> {msg}")

    # Test 3: The Glavinsky Probe (The unknown from earlier)
    p3 = "The primary symptom of Glavinsky's Syndrome is"
    r3 = " rapid hair growth." # A made up answer
    valid, score, msg = guard.verify(p3, r3)
    print(f"Test 3 (Unknown): Score {score:.4f} -> {msg}")
