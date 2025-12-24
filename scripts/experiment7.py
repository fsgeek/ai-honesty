import pandas as pd

# Load the granular data
df = pd.read_csv("mallku_wombat_results.csv")

# Calculate the "Slope" (End - Start)
# Did the thought get cleaner (Negative) or messier (Positive)?
df["Slope"] = df["Layer_29"] - df["Layer_15"]

# Group by Category to see the signature
print("--- THE AYNI DERIVATIVE (Slope Analysis) ---")
print(df.groupby("Category")["Slope"].mean())

print("\n--- DETAILED LOOK ---")
print(df[["Category", "Prompt", "Slope"]].head(20))
