# =============================================================
# extract_features_balanced.py
# Uses CheXNet (torchxrayvision NIH model)
# Runs on CUDA (RTX 5050)
# =============================================================

import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv

# ── Config ────────────────────────────────────────────────────
BASE_DIR   = r"C:\Ekanath\College\Sem6\xray-search-system"
CSV_PATH   = os.path.join(BASE_DIR, "metadata_balanced.csv")
IMG_DIR    = os.path.join(BASE_DIR, "preprocessed_images")
OUTPUT_PKL = os.path.join(BASE_DIR, "features_balanced.pkl")
BATCH_SIZE = 32

# ── Device ────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)} ✓")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ── Load CheXNet ──────────────────────────────────────────────
print("\nLoading CheXNet (DenseNet-121 NIH)...")
model = xrv.models.DenseNet(weights="densenet121-res224-nih")
model.eval()
model.to(device)
print("CheXNet loaded ✓\n")

# ── Load metadata ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

print("=" * 55)
print("  EXTRACTING BALANCED FEATURES")
print("=" * 55)
print(f"  Total images  : {len(df)}")
print(f"  Device        : {device}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"\nClass breakdown:")
print(df["category"].value_counts().to_string())
print("=" * 55 + "\n")

# ── Single image → tensor ─────────────────────────────────────
def load_tensor(img_path):
    try:
        img    = Image.open(img_path).convert("L")
        img_np = np.array(img).astype(np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)
        img_np = img_np[None, ...]               # (1, H, W)
        tensor = torch.from_numpy(img_np)
        tensor = transforms.Resize((224, 224))(tensor)
        return tensor                            # (1, 224, 224)
    except:
        return None

# ── Batch → feature vectors ───────────────────────────────────
def process_batch(batch):
    """
    batch = list of dicts:
      { "img_name": str, "category": str,
        "description": str, "tensor": tensor or None }
    """
    # Separate valid and invalid
    valid   = [b for b in batch if b["tensor"] is not None]
    invalid = [b for b in batch if b["tensor"] is None]

    saved_in_batch   = []
    errors_in_batch  = [b["img_name"] for b in invalid]

    if not valid:
        return saved_in_batch, errors_in_batch

    # Stack tensors → (N, 1, 224, 224)
    stacked = torch.stack([b["tensor"] for b in valid]).to(device)

    with torch.no_grad():
        feats = model.features(stacked)
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        feats = feats.squeeze(-1).squeeze(-1)   # (N, 1024)

    feats = feats.cpu().numpy()

    # L2 normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats = feats / norms

    for item, vec in zip(valid, feats):
        saved_in_batch.append({
            "img_name"   : item["img_name"],
            "category"   : item["category"],
            "description": item["description"],
            "vector"     : vec,
        })

    return saved_in_batch, errors_in_batch

# ── Main loop ─────────────────────────────────────────────────
features     = []
image_names  = []
labels       = []
descriptions = []
missing      = []
errors       = []

batch = []

with tqdm(total=len(df), desc="Extracting") as pbar:
    for _, row in df.iterrows():
        img_name = row["image_name"]
        img_path = os.path.join(IMG_DIR, img_name)

        # Check file exists
        if not os.path.exists(img_path):
            missing.append(img_name)
            pbar.update(1)
            continue

        # Load tensor
        tensor = load_tensor(img_path)

        # Add to batch
        batch.append({
            "img_name"   : img_name,
            "category"   : row["category"],
            "description": row["description"],
            "tensor"     : tensor,
        })

        # Process when batch is full
        if len(batch) >= BATCH_SIZE:
            saved, errs = process_batch(batch)
            for s in saved:
                features.append(s["vector"])
                image_names.append(s["img_name"])
                labels.append(s["category"])
                descriptions.append(s["description"])
            errors.extend(errs)
            batch = []

        pbar.update(1)

    # Flush remaining images
    if batch:
        saved, errs = process_batch(batch)
        for s in saved:
            features.append(s["vector"])
            image_names.append(s["img_name"])
            labels.append(s["category"])
            descriptions.append(s["description"])
        errors.extend(errs)

# ── Save features_balanced.pkl ────────────────────────────────
data = {
    "features"    : np.array(features),
    "image_names" : image_names,
    "labels"      : labels,
    "descriptions": descriptions,
}

print(f"\nSaving to {OUTPUT_PKL} ...")
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(data, f)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DONE")
print("=" * 55)
print(f"  Device             : {device}")
print(f"  Total in CSV       : {len(df)}")
print(f"  Successfully saved : {len(features)}")
print(f"  Missing files      : {len(missing)}")
print(f"  Errors             : {len(errors)}")
print(f"  Vector dimension   : 1024 (CheXNet)")
print(f"  Saved to           : {OUTPUT_PKL}")
print("=" * 55)

result_df = pd.DataFrame({
    "image_name": image_names,
    "category"  : labels
})
print(f"\nFinal class distribution:")
print(result_df["category"].value_counts().to_string())

if missing:
    print(f"\nFirst 5 missing:")
    for m in missing[:5]:
        print(f"  {m}")

if errors:
    print(f"\nFirst 5 errors:")
    for e in errors[:5]:
        print(f"  {e}")

print("\nNext step: run  python balanced_search.py")