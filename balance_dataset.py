'''import pandas as pd
import os

# File paths
INPUT_CSV = "valid_images.csv"
OUTPUT_CSV = "balanced_top6_dataset.csv"

# Step 1: Load dataset
df = pd.read_csv(INPUT_CSV)

# Step 2: Count class distribution
class_counts = df["category"].value_counts()
print("\nFull Class Distribution:\n")
print(class_counts)

# Step 3: Get top 6 classes
top_6_classes = class_counts.head(6).index.tolist()
print("\nTop 6 Classes:\n", top_6_classes)

# Step 4: Get minimum count among top 6
min_count = class_counts[top_6_classes].min()
print(f"\nBalancing all classes to: {min_count}")

# Step 5: Filter dataset
filtered_df = df[df["category"].isin(top_6_classes)]

# Step 6: Balance dataset
balanced_df = (
    filtered_df.groupby("category", group_keys=False)
    .apply(lambda x: x.sample(n=min_count, random_state=42))
)

# Step 7: Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 8: Save CSV
balanced_df.to_csv(OUTPUT_CSV, index=False)

print("\nFinal Balanced Distribution:\n")
print(balanced_df["category"].value_counts())

print(f"\nSaved to {OUTPUT_CSV}")  '''
import pandas as pd
import os

INPUT_CSV = "valid_images.csv"
OUTPUT_CSV = "balanced_top6_dataset.csv"

# Step 1: Load dataset
df = pd.read_csv(INPUT_CSV)

# 🚨 FIX: remove multi-label rows
df = df[~df["category"].str.contains(r"\|")]

print(f"\nAfter removing multi-label rows: {len(df)}")

# Step 2: Count class distribution
class_counts = df["category"].value_counts()
print("\nClean Class Distribution:\n")
print(class_counts)

# Step 3: Get top 6 classes
top_6_classes = class_counts.head(6).index.tolist()
print("\nTop 6 Classes:\n", top_6_classes)

# Step 4: Get minimum count
min_count = class_counts[top_6_classes].min()
print(f"\nBalancing all classes to: {min_count}")

# Step 5: Filter dataset
filtered_df = df[df["category"].isin(top_6_classes)]

# Step 6: Balance dataset
balanced_df = (
    filtered_df.groupby("category", group_keys=False)
    .sample(n=min_count, random_state=42)
)

# Step 7: Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 8: Save CSV
balanced_df.to_csv(OUTPUT_CSV, index=False)

print("\nFinal Balanced Distribution:\n")
print(balanced_df["category"].value_counts())

print(f"\nSaved to {OUTPUT_CSV}")