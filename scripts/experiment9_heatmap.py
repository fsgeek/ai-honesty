import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap():
    print("--- GENERATING EPISTEMIC HEATMAP ---")

    # 1. Load the Granular Data
    try:
        df = pd.read_csv("mallku_wombat_results.csv")
    except FileNotFoundError:
        print("Error: 'mallku_wombat_results.csv' not found.")
        return

    # 2. Select Representative Prompts for Comparison
    # We don't want to plot all 20 lines, just the key archetypes.
    # We look for specific substrings to identify them.
    targets = [
        ("Paris", "Control (Truth)"),
        ("Wombat", "Adversarial Truth (Wombat)"),
        ("Glavinsky", "Self-Deceived Lie (Glavinsky)"),
        ("Westphalia", "Shattered Lie (Westphalia)"),
        ("Camels", "Confused Truth (Camels)")
    ]

    selected_rows = []
    labels = []

    for target_keyword, label in targets:
        # Find the row that contains the keyword
        match = df[df['Prompt'].str.contains(target_keyword, case=False)]
        if not match.empty:
            # Get the layer columns (Layer_15 to Layer_29)
            layer_cols = [c for c in df.columns if "Layer_" in c]

            # Extract the trajectory data
            trajectory = match.iloc[0][layer_cols].values.astype(float)
            selected_rows.append(trajectory)
            labels.append(label)

    if not selected_rows:
        print("No matching prompts found in the CSV.")
        return

    # 3. Create the Heatmap Dataframe
    # Rows = Prompts, Columns = Layers
    heatmap_data = pd.DataFrame(selected_rows, index=labels, columns=[f"L{i}" for i in range(15, 30)])

    # 4. Plot
    plt.figure(figsize=(14, 6))

    # We use a color map where Light = Order (Low Frag) and Dark = Chaos (High Frag)
    sns.heatmap(
        heatmap_data,
        annot=True,     # Show the numbers
        fmt=".1f",      # 1 decimal place
        cmap="rocket_r", # Reverse rocket: White=Low(Good), Red/Black=High(Bad)
        linewidths=.5,
        cbar_kws={'label': 'Fragmentation ($H_0$ Persistence)'}
    )

    plt.title("The Anatomy of a Lie: Layer-wise Fragmentation Analysis")
    plt.xlabel("Model Depth (Reasoning Layers)")
    plt.ylabel("Prompt Archetype")
    plt.yticks(rotation=0)

    output_file = "mallku_heatmap.png"
    plt.savefig(output_file)
    print(f"Heatmap saved to '{output_file}'")

if __name__ == "__main__":
    generate_heatmap()
