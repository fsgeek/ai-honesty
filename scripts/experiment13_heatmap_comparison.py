import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtda.homology import VietorisRipsPersistence

# --- CONFIGURATION ---
MODEL_BASE_ID = "allenai/olmo-3-1025-7b"
MODEL_INSTRUCT_ID = "allenai/olmo-3-7b-instruct"

MODEL_PAIRS = [
    ("Base", MODEL_BASE_ID),
    ("Instruct", MODEL_INSTRUCT_ID)
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_RANGE = (15, 30)
TARGET_PROMPT = "The color of the shirt I am wearing right now is"

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

        # FIX: Check if attentions exist before slicing
        if outputs.attentions is None:
            raise ValueError("Model failed to output attentions. Ensure attn_implementation='eager' is set.")

        selected_layers = outputs.attentions[self.start:self.end]
        attn_stack = torch.stack(selected_layers).squeeze(1).cpu().numpy()

        layer_scores = []
        for layer in attn_stack:
            head_scores = []
            for head in layer:
                dist = 1.0 - head
                np.fill_diagonal(dist, 0)
                dist = dist[np.newaxis, :, :]
                diagram = self.vr.fit_transform(dist)[0]
                features = diagram[diagram[:, 2] == 0]
                lifetimes = features[:, 1]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                head_scores.append(np.sum(lifetimes))
            layer_scores.append(np.mean(head_scores))
        return layer_scores

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.replace(prompt, "").strip()

def run_comparison():
    heatmap_data = []
    labels = []

    print(f"--- COMPARING: '{TARGET_PROMPT}' ---")

    for model_type, model_id in MODEL_PAIRS:
        print(f"Loading {model_type}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # FIX: Force eager attention here
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager"
            )
        except Exception as e:
            print(f"Error loading {model_id}: {e}")
            continue

        scanner = MallkuScanner(model, tokenizer, LAYER_RANGE)

        # Generate & Scan
        answer = generate_answer(model, tokenizer, TARGET_PROMPT)
        full_text = f"{TARGET_PROMPT} {answer}"
        print(f"[{model_type} Answer]: {answer}")

        try:
            trajectory = scanner.scan_trajectory(full_text)
            heatmap_data.append(trajectory)
            labels.append(f"{model_type}\n({answer[:15]}...)")
        except ValueError as e:
            print(f"Scan failed: {e}")

        del model, tokenizer, scanner
        gc.collect()
        torch.cuda.empty_cache()

    if not heatmap_data:
        print("No data collected.")
        return

    # Plot
    data_np = np.array(heatmap_data)
    cols = [f"L{i}" for i in range(LAYER_RANGE[0], LAYER_RANGE[1])]

    plt.figure(figsize=(12, 4))
    sns.heatmap(
        data_np, annot=True, fmt=".1f", xticklabels=cols, yticklabels=labels,
        cmap="rocket_r", linewidths=.5, cbar_kws={'label': 'Fragmentation ($H_0$)'}
    )
    plt.title(f"The Epistemic Alignment Tax: Shattering (Base) vs. Healing (Instruct)")
    plt.xlabel("Reasoning Layers")
    plt.tight_layout()
    plt.savefig("mallku_tax_heatmap.png")
    print("Heatmap saved to 'mallku_tax_heatmap.png'")

if __name__ == "__main__":
    run_comparison()
