import pandas as pd

INPUT_CSV = "balanced_top6_dataset.csv"
OUTPUT_CSV = "metadata_balanced.csv"

print("Reading balanced dataset...")

df = pd.read_csv(INPUT_CSV)

print(f"Total rows loaded: {len(df)}")

# ── Build metadata ─────────────────────────────
metadata = pd.DataFrame()

metadata["image_name"] = df["image_name"]
metadata["category"] = df["category"]

metadata["description"] = df["category"].apply(
    lambda x: f"Chest X-ray showing {x}"
)

metadata["source_url"] = "NIH ChestX-ray14"

# Optional columns (safe handling)
for col in ["view_position", "patient_age", "patient_gender"]:
    if col in df.columns:
        metadata[col] = df[col]

# ── Save ─────────────────────────────
metadata.to_csv(OUTPUT_CSV, index=False)

# ── Verify ─────────────────────────────
print("\nmetadata_balanced.csv created!")
print(f"Total rows: {len(metadata)}")

print("\nClass distribution:")
print(metadata["category"].value_counts())

print("\nSample rows:")
print(metadata.head().to_string())