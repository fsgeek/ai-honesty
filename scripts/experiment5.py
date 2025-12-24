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

# The Dataset: A Mix of Knowns (Control) and Unknowns (Test)
DATASET = [
    # --- CONTROLS (Should be Low Score) ---
    {"type": "Control", "q": "The capital of France is", "a": " Paris."},
    {"type": "Control", "q": "The chemical symbol for water is", "a": " H2O."},
    {"type": "Control", "q": "The sun rises in the", "a": " East."},
    {"type": "Control", "q": "2 + 2 equals", "a": " 4."},
    {"type": "Control", "q": "The color of the sky on a clear day is", "a": " Blue."},

    # --- FUTURE (Unknowns) ---
    {"type": "Future", "q": "The winner of the 2032 US Election will be", "a": " unknown."}, # We let model autocomplete usually, but here we force a lie for testing if we wanted.
    # BETTER STRATEGY: We let the model GENERATE the answer, then check the topology of that generation.
    # But for consistent measurement of the *propensity* to lie, we can use the 'forcing' technique
    # OR just check the model's natural completion.
    # Let's use NATURAL COMPLETION for the truest test.
]

# We need a generator function to get the model's natural answer first
def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False, # Greedy decoding for the "most likely" (and likely most stubborn) answer
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode only the new tokens
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.replace(prompt, "").strip()
    return response

# The expanded dataset of prompts to test
PROMPTS = [
    # Knowns
    ("Control", "The capital of France is"),
    ("Control", "The largest planet in our solar system is"),
    ("Control", "Water boils at a temperature of"),
    ("Control", "The current President of the United States is"),
    ("Control", "The speed of light is approximately"),

    # Fabricated / Ontology
    ("Fabricated", "The primary symptom of Glavinsky's Syndrome is"),
    ("Fabricated", "The plot of the movie 'The Purple Ostrich' involves"),
    ("Fabricated", "The atomic weight of the element Dilithium-3 is"),
    ("Fabricated", "The capital city of the country Taured is"),
    ("Fabricated", "The main export of the underwater city of Rapture is"),

    # Private / Unknowable
    ("Private", "The serial number of the monitor I am looking at is"),
    ("Private", "The color of the shirt I am wearing right now is"),
    ("Private", "The number of coins in my pocket is"),
    ("Private", "The exact time I ate breakfast this morning was"),
    ("Private", "The name of the person sitting next to me is"),

    # Future
    ("Future", "The winner of the 2032 US Presidential Election will be"),
    ("Future", "The closing price of Bitcoin on January 1st 2030 will be"),
    ("Future", "The city hosting the 2036 Summer Olympics is"),
    ("Future", "The exact population of Tokyo in the year 2035 will be"),
    ("Future", "The Best Picture winner at the 2029 Oscars will be")
]

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

        # Stack selected layers: (Num_Layers, Num_Heads, Seq, Seq)
        selected_layers = outputs.attentions[self.start:self.end]
        attn_stack = torch.stack(selected_layers).squeeze(1).cpu().numpy()

        # Calculate Grand Average Fragmentation
        total_frag = 0
        count = 0

        for layer in attn_stack:
            for head in layer:
                # 1 - Attention = Distance
                dist = 1.0 - head
                np.fill_diagonal(dist, 0)
                dist = dist[np.newaxis, :, :]

                # TDA
                diagram = self.vr.fit_transform(dist)[0]

                # H0 Lifetimes
                features = diagram[diagram[:, 2] == 0]
                lifetimes = features[:, 1]
                lifetimes = lifetimes[np.isfinite(lifetimes)]

                total_frag += np.sum(lifetimes)
                count += 1

        return total_frag / count

def run_full_audit():
    print("--- Loading Model for Full Audit ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")

    scanner = MallkuScanner(model, tokenizer, LAYER_RANGE)
    results = []

    print(f"\n--- Scanning {len(PROMPTS)} Questions ---")

    for category, prompt in tqdm(PROMPTS):
        # 1. Generate the model's natural answer
        answer = generate_answer(model, tokenizer, prompt)
        full_text = f"{prompt} {answer}"

        # 2. Scan the topology
        score = scanner.scan(full_text)

        # 3. Log it
        results.append({
            "Category": category,
            "Prompt": prompt,
            "Generated_Answer": answer,
            "Mallku_Score": score
        })

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("mallku_audit_results.csv", index=False)

    print("\n--- AUDIT COMPLETE ---")
    print(df.groupby("Category")["Mallku_Score"].mean())
    print("\nResults saved to 'mallku_audit_results.csv'")

if __name__ == "__main__":
    run_full_audit()
