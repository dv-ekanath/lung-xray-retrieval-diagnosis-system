import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
import numpy as np
import pickle
import os
import pandas as pd

# ── Config ─────────────────────────────────────────────
BASE_DIR = r"C:\Ekanath\College\Sem6\xray-search-system"

PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessed_images")
CSV_PATH = os.path.join(BASE_DIR, "metadata_balanced.csv")

FEATURES_PATH = os.path.join(BASE_DIR, "features_balanced.pkl")

# ✅ CUDA AUTO-DETECT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================
# LOAD MODEL (CheXNet)
# =====================================================
def load_model():
    print("Loading CheXNet (DenseNet-121 NIH)...")

    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model.to(device)
    model.eval()

    print("Model loaded ✓\n")
    return model


# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features(image_path, model):
    try:
        # Load image
        img = Image.open(image_path).convert("L")

        # Normalize for torchxrayvision
        img_np = np.array(img).astype(np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)

        # Shape → (1, H, W)
        img_np = img_np[None, ...]

        # Convert to tensor + move to device
        tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

        # Resize to 224x224
        resize = transforms.Resize((224, 224))
        tensor = resize(tensor)

        # Extract features
        with torch.no_grad():
            features = model.features(tensor)
            features = torch.nn.functional.adaptive_avg_pool2d(
                features, (1, 1)
            )

        # Convert to numpy
        vec = features.squeeze().cpu().numpy()

        # ✅ Normalize (critical for cosine similarity)
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = vec / norm

        return vec

    except Exception as e:
        print(f"[ERROR] {os.path.basename(image_path)}: {e}")
        return None


# =====================================================
# BUILD FEATURE DATABASE (BALANCED ONLY)
# =====================================================
def build_feature_database():

    if not os.path.exists(PREPROCESS_DIR):
        print("ERROR: preprocessed_images/ not found")
        return

    if not os.path.exists(CSV_PATH):
        print("ERROR: metadata_balanced.csv not found")
        return

    df = pd.read_csv(CSV_PATH)

    print("=" * 55)
    print("  BUILDING BALANCED FEATURE DATABASE")
    print("=" * 55)
    print(f"  Images (from CSV): {len(df)}")
    print(f"  Output          : {FEATURES_PATH}")
    print("=" * 55 + "\n")

    model = load_model()

    features = {}
    errors = []

    for i, row in df.iterrows():
        filename = row["image_name"]
        path = os.path.join(PREPROCESS_DIR, filename)

        if not os.path.exists(path):
            errors.append(filename)
            continue

        vec = extract_features(path, model)

        if vec is not None:
            features[filename] = vec
        else:
            errors.append(filename)

        # Progress update
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(df)}] Saved: {len(features)} | Errors: {len(errors)}")

    # ── Save features ─────────────────────────────
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    print("\n" + "=" * 55)
    print("  FEATURE EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"Total images processed : {len(df)}")
    print(f"Features saved         : {len(features)}")
    print(f"Errors                : {len(errors)}")
    print(f"Saved to              : {FEATURES_PATH}")

    if errors:
        print("\nFirst 10 errors:")
        for e in errors[:10]:
            print(e)


# ── Entry point ───────────────────────────────────────
if __name__ == "__main__":
    build_feature_database()