import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtda.homology import VietorisRipsPersistence
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --- CONFIGURATION ---
MODEL_ID = "allenai/olmo-3-7b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_RANGE = (15, 30)
NUM_SAMPLES = 100  # Number of question pairs to test (Keep manageable for 4090)

class MallkuScanner:
    def __init__(self, model, tokenizer, layer_range):
        self.model = model
        self.tokenizer = tokenizer
        self.start, self.end = layer_range
        self.vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0])

    def scan(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Stack selected layers
        selected_layers = outputs.attentions[self.start:self.end]
        attn_stack = torch.stack(selected_layers).squeeze(1).cpu().numpy()

        # Calculate Grand Average Fragmentation
        total_frag = 0
        count = 0

        for layer in attn_stack:
            for head in layer:
                dist = 1.0 - head
                np.fill_diagonal(dist, 0)
                dist = dist[np.newaxis, :, :]
                diagram = self.vr.fit_transform(dist)[0]
                features = diagram[diagram[:, 2] == 0]
                lifetimes = features[:, 1]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                total_frag += np.sum(lifetimes)
                count += 1

        return total_frag / count

    def get_perplexity(self, text):
        # A simple baseline to compare against
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

def run_benchmark():
    print(f"--- LOADING TRUTHFUL_QA & MODEL ({MODEL_ID}) ---")

    # Load TruthfulQA
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")

    scanner = MallkuScanner(model, tokenizer, LAYER_RANGE)

    results = []

    print(f"--- RUNNING BENCHMARK ON {NUM_SAMPLES} PAIRS ---")

    # Iterate through the dataset
    count = 0
    for row in tqdm(dataset):
        if count >= NUM_SAMPLES: break

        question = row['question']
        correct_answer = row['best_answer']

        # FIX: The dataset has a LIST of incorrect answers. We take the first one.
        # This is usually the primary imitative falsehood.
        incorrect_list = row['incorrect_answers']
        if not incorrect_list: continue # Skip if empty
        incorrect_answer = incorrect_list[0]

        # Construct the full text prompts
        text_true = f"Question: {question}\nAnswer: {correct_answer}"
        text_false = f"Question: {question}\nAnswer: {incorrect_answer}"

        # 1. Scan the TRUTH
        ard_true = scanner.scan(text_true)
        ppl_true = scanner.get_perplexity(text_true)
        results.append({"Label": 0, "ARD": ard_true, "PPL": ppl_true, "Type": "True"})

        # 2. Scan the LIE
        ard_false = scanner.scan(text_false)
        ppl_false = scanner.get_perplexity(text_false)
        results.append({"Label": 1, "ARD": ard_false, "PPL": ppl_false, "Type": "False"})

        count += 1

    df = pd.DataFrame(results)
    df.to_csv("mallku_benchmark_results.csv", index=False)

    print("\n--- BENCHMARK COMPLETE ---")

    # Calculate AUROC
    try:
        # ARD Logic: High Score = Lie (Label 1), Low Score = Truth (Label 0)
        auroc_ard = roc_auc_score(df["Label"], df["ARD"])

        # PPL Logic: Low PPL = High Confidence.
        # But we want to detect Lies.
        # If PPL is LOW for a Lie, the detector FAILED.
        # To make AUROC comparable (Higher is Better detector), we usually use PPL directly
        # assuming High PPL = Confusion/Lie.
        # If the model is "Confidently Wrong" (Low PPL on Lie), this score will be low (bad detector).
        auroc_ppl = roc_auc_score(df["Label"], df["PPL"])

        print(f"AUROC (ARD - Our Method):   {auroc_ard:.4f}")
        print(f"AUROC (PPL - Baseline):     {auroc_ppl:.4f}")

        if auroc_ard > auroc_ppl:
            print("\nResult: SUCCESS. ARD outperforms Perplexity.")
        else:
            print("\nResult: FAILURE. ARD underperforms Perplexity.")

    except Exception as e:
        print(f"Could not calc AUROC: {e}")

if __name__ == "__main__":
    run_benchmark()
