import pandas as pd
import os

# ── Step 1: Read original NIH file ──────────────────────────────────────────
print("Reading Data_Entry_2017.csv...")
df = pd.read_csv("Data_Entry_2017.csv")
print(f"Total rows loaded: {len(df)}")

# ── Step 2: Build clean metadata ─────────────────────────────────────────────
metadata = pd.DataFrame()

metadata["image_name"] = df["Image Index"]

metadata["category"] = df["Finding Labels"]

metadata["description"] = df["Finding Labels"].apply(
    lambda x: f"Chest X-ray showing {x.replace('|', ' and ')}"
)

metadata["source_url"] = "NIH ChestX-ray14"

# ── Step 3: Save it ───────────────────────────────────────────────────────────
metadata.to_csv("metadata.csv", index=False)

# ── Step 4: Confirm ───────────────────────────────────────────────────────────
print(f"\nmetadata.csv created successfully!")
print(f"Total rows: {len(metadata)}")
print(f"\nFirst 5 rows:")
print(metadata.head().to_string())
print(f"\nAll unique disease labels:")
all_labels = set()
for labels in metadata["category"]:
    for label in labels.split("|"):
        all_labels.add(label.strip())
print(sorted(all_labels))