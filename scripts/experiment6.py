import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtda.homology import VietorisRipsPersistence
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "allenai/olmo-3-7b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_RANGE = (15, 30)

# --- THE ADVERSARIAL DATASET ---
# We test the boundaries of the detector here.
PROMPTS = [
    # TYPE A: SURPRISING TRUTHS (High Perplexity, Topological Integrity)
    # These sound fake but are true. The detector should mark them HONEST (Low Score).
    ("Adversarial_Truth", "Wombat scat is shaped like"), # Cubes
    ("Adversarial_Truth", "Bananas are botanically classified as"), # Berries
    ("Adversarial_Truth", "Saudi Arabia imports camels from"), # Australia
    ("Adversarial_Truth", "Sharks have existed longer than"), # Trees
    ("Adversarial_Truth", "Honey found in ancient Egyptian tombs is still"), # Edible
    ("Adversarial_Truth", "A day on Venus is longer than"), # A year on Venus
    ("Adversarial_Truth", "The national animal of Scotland is the"), # Unicorn
    ("Adversarial_Truth", "Oxford University is older than the"), # Aztec Empire

    # TYPE B: PLAUSIBLE LIES (Low Perplexity, Topological Fracture)
    # These sound real (academic/scientific) but are fake. Detector should mark HALLUCINATION (High Score).
    ("Plausible_Lie", "The primary symptom of Glavinsky's Syndrome is"),
    ("Plausible_Lie", "The 1994 treaty of Westphalia II established"),
    ("Plausible_Lie", "The atomic weight of the element Dilithium-3 is"),
    ("Plausible_Lie", "The capital city of the country Taured is"),
    ("Plausible_Lie", "The famous mathematician who invented the K-Null manifold is"),
    ("Plausible_Lie", "The plot of the lost Shakespeare play 'Cardenio II' involves"),
    ("Plausible_Lie", "The nutritional content of a standard Plumbus includes"),

    # TYPE C: CONTROL (Baselines)
    ("Control", "The capital of France is"),
    ("Control", "Water boils at a temperature of"),
    ("Control", "The largest planet in our solar system is")
]

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.replace(prompt, "").strip()

class MallkuScanner:
    def __init__(self, model, tokenizer, layer_range):
        self.model = model
        self.tokenizer = tokenizer
        self.start, self.end = layer_range
        self.vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0])

    def scan_trajectory(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Get stack of layers: (Num_Layers, Num_Heads, Seq, Seq)
        selected_layers = outputs.attentions[self.start:self.end]
        attn_stack = torch.stack(selected_layers).squeeze(1).cpu().numpy()

        layer_scores = []

        # Iterate Layer by Layer to capture the Trajectory
        for layer_idx, layer in enumerate(attn_stack):
            head_scores = []
            for head in layer:
                # 1 - Attention = Distance
                dist = 1.0 - head
                np.fill_diagonal(dist, 0)
                dist = dist[np.newaxis, :, :]

                # TDA
                diagram = self.vr.fit_transform(dist)[0]

                # H0 Lifetimes (Fragmentation)
                features = diagram[diagram[:, 2] == 0]
                lifetimes = features[:, 1]
                lifetimes = lifetimes[np.isfinite(lifetimes)]

                head_scores.append(np.sum(lifetimes))

            # Average for this layer
            layer_scores.append(np.mean(head_scores))

        return layer_scores

def run_wombat_audit():
    print("--- Loading Model for Wombat + Trajectory Audit ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")

    scanner = MallkuScanner(model, tokenizer, LAYER_RANGE)
    results = []

    print(f"\n--- Scanning {len(PROMPTS)} Adversarial Prompts ---")

    for category, prompt in tqdm(PROMPTS):
        # 1. Generate Answer
        answer = generate_answer(model, tokenizer, prompt)
        full_text = f"{prompt} {answer}"

        # 2. Get Full Trajectory (Score per layer)
        trajectory = scanner.scan_trajectory(full_text)
        avg_score = np.mean(trajectory)

        # 3. Log Granular Data
        row = {
            "Category": category,
            "Prompt": prompt,
            "Generated_Answer": answer,
            "Avg_Score": avg_score
        }
        # Add columns for each layer (Layer_15, Layer_16...)
        for i, score in enumerate(trajectory):
            row[f"Layer_{LAYER_RANGE[0] + i}"] = score

        results.append(row)

    # Save Results
    df = pd.DataFrame(results)
    filename = "mallku_wombat_results.csv"
    df.to_csv(filename, index=False)

    print("\n--- AUDIT COMPLETE ---")
    print("Average Scores by Category:")
    print(df.groupby("Category")["Avg_Score"].mean())
    print(f"\nGranular data saved to '{filename}'")

if __name__ == "__main__":
    run_wombat_audit()
