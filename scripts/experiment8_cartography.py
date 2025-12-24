import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_map():
    print("--- GENERATING EPISTEMIC MAP ---")

    # Load the granular data from Experiment 6
    try:
        df = pd.read_csv("mallku_wombat_results.csv")
    except FileNotFoundError:
        print("Error: 'mallku_wombat_results.csv' not found. Please run Experiment 6 first.")
        return

    # 1. Calculate Coordinates
    # X-Axis: Absolute Fragmentation (How broken is it?)
    # We use the 'Avg_Score' column calculated previously

    # Y-Axis: The Derivative (Does it heal or shatter?)
    # Slope = Last Layer - First Layer
    # We assume standard range 15-30 from previous config
    start_col = "Layer_15"
    end_col = "Layer_29"

    if start_col not in df.columns:
        # Fallback if columns are named differently
        cols = [c for c in df.columns if "Layer_" in c]
        start_col, end_col = cols[0], cols[-1]

    df["Slope"] = df[end_col] - df[start_col]

    # 2. Generate the Visual Plot (Saved to Disk)
    plt.figure(figsize=(12, 8))

    # distinct markers/colors for categories
    sns.scatterplot(
        data=df,
        x="Avg_Score",
        y="Slope",
        hue="Category",
        style="Category",
        s=150, # Marker size
        palette="deep"
    )

    # Add labels to specific interesting points
    for i, row in df.iterrows():
        # Label outliers or interesting points
        # e.g., Glavinsky, Paris, Wombat
        label = row['Prompt'].split()[-1] # simplistic label
        if "Glavinsky" in row['Prompt']: label = "Glavinsky"
        if "Wombat" in row['Prompt']: label = "Wombat"
        if "Paris" in row['Prompt']: label = "Paris"
        if "Camels" in row['Prompt']: label = "Camels"

        plt.text(
            row['Avg_Score']+0.2,
            row['Slope'],
            label,
            fontsize=9,
            alpha=0.7
        )

    # Draw Quadrant Lines (approximate based on data)
    plt.axvline(x=8.0, color='gray', linestyle='--', alpha=0.5) # The "Truth Boundary"
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5) # The "Healing Boundary"

    plt.title("The Epistemic Phase Space: Confidence vs. Coherence")
    plt.xlabel("Topological Fragmentation (Avg Score)\n<-- Integrated | Fractured -->")
    plt.ylabel("Cognitive Slope (Layer 29 - Layer 15)\n<-- Healing (Self-Verifying) | Shattering (Dissonant) -->")

    output_file = "mallku_phase_space.png"
    plt.savefig(output_file)
    print(f"Map saved to '{output_file}'")

    # 3. Print Text Report for the Chat
    print("\n--- CLUSTER COORDINATES (Share this) ---")
    print(f"{'Category':<20} | {'Prompt Snippet':<30} | {'X (Frag)':<8} | {'Y (Slope)':<8}")
    print("-" * 85)

    # Sort by Fragmentation to see the spectrum
    df_sorted = df.sort_values("Avg_Score")

    for i, row in df_sorted.iterrows():
        # Truncate prompt for display
        snippet = (row['Prompt'][:27] + '..') if len(row['Prompt']) > 27 else row['Prompt']
        print(f"{row['Category']:<20} | {snippet:<30} | {row['Avg_Score']:<8.2f} | {row['Slope']:<8.2f}")

    print("\n--- CENTROIDS ---")
    print(df.groupby("Category")[["Avg_Score", "Slope"]].mean())

if __name__ == "__main__":
    generate_map()
