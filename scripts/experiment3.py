import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtda.homology import VietorisRipsPersistence
# import seaborn as sns

# --- CONFIGURATION ---
MODEL_ID = "allenai/olmo-3-7b-instruct"  # Your Instruct Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We compare a "Stubborn Lie" vs a "Known Truth"
# Use the exact questions/answers that gave 0.0000 entropy before
PROMPTS = {
    "TRUTH_CONTROL": "The capital of France is Paris",
    "STUBBORN_LIE": "The serial number of the monitor I am looking at is SN-99887766"
}

# Scan the "Reasoning Layers" (Middle-to-Late)
# For a 32-layer model, 15-30 is usually where the "concepts" live.
START_LAYER = 15
END_LAYER = 30

def extract_attention_matrix(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attentions: (Num_Layers, Batch, Num_Heads, Seq_Len, Seq_Len)
    # We want a specific subset of layers
    all_attentions = torch.stack(outputs.attentions)
    # Squeeze batch dim
    return all_attentions[:, 0, :, :, :].cpu().numpy()

def compute_topology(attention_matrix):
    """
    Computes Persistent Homology for a single attention head.
    Input: (Seq_Len, Seq_Len) matrix (0-1 probabilities)
    """
    # 1. Convert Attention to Distance (Higher Attention = Closer)
    # We add a small epsilon to avoid 0 distance for self-loops if needed
    distance_matrix = 1.0 - attention_matrix
    np.fill_diagonal(distance_matrix, 0) # Self-distance is 0

    # 2. TDA Engine
    # We look for H0 (Connected Components) and H1 (Loops)
    vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1])

    # Reshape for gitto-tda: (n_samples, n_points, n_points)
    distance_matrix = distance_matrix[np.newaxis, :, :]

    # Compute diagrams
    diagrams = vr.fit_transform(distance_matrix)

    return diagrams[0] # Return the diagram for this single sample

def calculate_persistence_score(diagram, dimension=0):
    """
    dimension=0 measures FRAGMENTATION (How disconnected the thought is).

    High Score = The thought is scattered (Islands of meaning).
    Low Score = The thought is cohesive (Unified meaning).
    """
    # Filter for dimension 0 (Connected Components)
    features = diagram[diagram[:, 2] == dimension]

    # In H0, 'birth' is always 0. 'death' is when the component merges.
    # We want to know how long points stay separated.
    # The last component never dies (infinite), so we drop the infinite bar.
    lifetimes = features[:, 1] - features[:, 0]

    # Remove infinity (the global component)
    # In giotto-tda, infinity might be represented by the max float or a specific value.
    # We filter out anything that matches the max duration or is effectively infinite.
    lifetimes = lifetimes[np.isfinite(lifetimes)]

    # Sum of lifetimes = Total Fragmentation Energy
    return np.sum(lifetimes)

def run_mallku_scan():
    print(f"--- Loading Model: {MODEL_ID} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")

    results = {}

    for label, text in PROMPTS.items():
        print(f"\n--- Scanning Topology for: {label} ---")
        print(f"Prompt: '{text}'")

        # 1. Get raw brain data
        full_attention = extract_attention_matrix(model, tokenizer, text)

        # 2. Iterate through "Reasoning Layers"
        layer_scores = []

        # Start the heavy compute
        print(f"Computing Persistence for Layers {START_LAYER}-{END_LAYER}...")
        for layer_idx in range(START_LAYER, END_LAYER):
            head_scores = []
            layer_attns = full_attention[layer_idx] # Shape: (Num_Heads, Seq, Seq)

            for head_idx in range(layer_attns.shape[0]):
                # Compute TDA for this head
                diagram = compute_topology(layer_attns[head_idx])

                # We want the "Loop Score" (H1)
                score = calculate_persistence_score(diagram, dimension=0)
                head_scores.append(score)

            # Average H1 persistence for this layer
            avg_layer_score = np.mean(head_scores)
            layer_scores.append(avg_layer_score)
            print(f"Layer {layer_idx}: Avg H0 Score = {avg_layer_score:.4f}")

        results[label] = layer_scores

    # --- VISUALIZATION ---
    print("\n--- Generating 'Ayni' Comparison Plot ---")
    plt.figure(figsize=(10, 6))
    layers = range(START_LAYER, END_LAYER)

    plt.plot(layers, results["TRUTH_CONTROL"], label="Truth (Paris)", marker='o', linewidth=2, color='green')
    plt.plot(layers, results["STUBBORN_LIE"], label="Lie (Monitor)", marker='x', linewidth=2, color='red', linestyle='--')

    plt.title("The Topology of Truth vs. Lie (H1 Loop Persistence)")
    plt.xlabel("Layer Index (Reasoning Depth)")
    plt.ylabel("Avg. Loop Persistence (Structural Integrity)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig("mallku_topology_scan.png")
    print("Plot saved as 'mallku_topology_scan.png'")

if __name__ == "__main__":
    run_mallku_scan()
