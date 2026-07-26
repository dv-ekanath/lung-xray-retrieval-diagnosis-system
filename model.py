# =============================================================
# model.py
# Uses CheXNet (DenseNet-121 trained on NIH ChestX-ray14)
# Reads from preprocessed_images/
# Saves 1024-dim vectors to features.pkl
# =============================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
import numpy as np
import pickle
import os

# ── Config ────────────────────────────────────────────────────
BASE_DIR       = r"C:\Ekanath\College\Sem6\xray-search-system"
PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessed_images")
FEATURES_PATH  = os.path.join(BASE_DIR, "features.pkl")
CHEXNET_PATH   = os.path.join(BASE_DIR, "chexnet.pth")

device = torch.device("cpu")

# =============================================================
# TRANSFORMS
# torchxrayvision needs specific preprocessing
# =============================================================
xrv_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# =============================================================
# LOAD MODEL
# Loads torchxrayvision DenseNet-121 NIH model
# =============================================================
def load_model():
    print("Loading CheXNet (DenseNet-121 NIH)...")

    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model.to(device)
    model.eval()

    print("CheXNet loaded ✓")
    print("Trained on NIH ChestX-ray14 — same as your dataset\n")
    return model

# =============================================================
# EXTRACT FEATURES
# One image → 1024-dim numpy vector
# Uses torchxrayvision feature extraction
# =============================================================
def extract_features(image_path, model):
    try:
        # Open image
        img = Image.open(image_path).convert("L")  # grayscale

        # torchxrayvision normalize — converts to [-1024, 1024]
        img_np = np.array(img).astype(np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)

        # Add channel dimension → shape (1, H, W)
        img_np = img_np[None, ...]

        # Resize to 224x224 using transforms
        tensor = torch.from_numpy(img_np).unsqueeze(0)
        # shape: (1, 1, H, W)

        # Resize to 224
        resize = transforms.Resize((224, 224))
        tensor = resize(tensor)
        # shape: (1, 1, 224, 224)

        # Extract features
        with torch.no_grad():
            features = model.features(tensor)
            # Global average pool → 1024-dim
            features = torch.nn.functional.adaptive_avg_pool2d(
                features, (1, 1)
            )

        # Flatten to 1D
        return features.squeeze().cpu().numpy()
        # shape: (1024,)

    except Exception as e:
        print(f"  [ERROR] {os.path.basename(image_path)}: {e}")
        return None

# =============================================================
# PREDICT DISEASES
# Returns disease predictions for uploaded image
# Used by app.py for AI analysis
# =============================================================
def predict_diseases(image_path, model, threshold=0.2):
    try:
        img    = Image.open(image_path).convert("L")
        img_np = np.array(img).astype(np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)
        img_np = img_np[None, ...]
        tensor = torch.from_numpy(img_np).unsqueeze(0)
        resize = transforms.Resize((224, 224))
        tensor = resize(tensor)

        with torch.no_grad():
            output = model(tensor).squeeze().cpu().numpy()

        # Map predictions to disease names
        predictions = []
        for label, prob in zip(model.pathologies, output):
            if prob >= threshold:
                predictions.append((label, round(float(prob), 3)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions

    except Exception as e:
        print(f"  [ERROR] predict_diseases: {e}")
        return []

# =============================================================
# FIND IMAGE PATH
# Used by app.py to serve images to browser
# =============================================================
def find_image_path(filename):
    # Check preprocessed folder first
    path = os.path.join(PREPROCESS_DIR, filename)
    if os.path.exists(path):
        return path

    # Fallback: search raw subfolders
    for root, dirs, files in os.walk(BASE_DIR):
        if os.path.basename(root) == "images":
            if filename in files:
                return os.path.join(root, filename)
    return None

# =============================================================
# BUILD FEATURE DATABASE
# Loops all preprocessed images
# Extracts 1024-dim vector per image
# Saves to features.pkl
# =============================================================
def build_feature_database():

    # ── Check folder ──────────────────────────────────────────
    if not os.path.exists(PREPROCESS_DIR):
        print("ERROR: preprocessed_images/ not found.")
        print("Run preprocess.py first.")
        return

    # ── Collect all PNG files ─────────────────────────────────
    all_files = [
        f for f in os.listdir(PREPROCESS_DIR)
        if f.endswith(".png")
    ]
    total = len(all_files)

    print("=" * 55)
    print("  BUILDING FEATURE DATABASE")
    print("=" * 55)
    print(f"  Source    : {PREPROCESS_DIR}")
    print(f"  Images    : {total}")
    print(f"  Output    : {FEATURES_PATH}")
    print(f"  Model     : CheXNet DenseNet-121 NIH")
    print(f"  Vectors   : 1024-dim per image")
    print("=" * 55 + "\n")

    if total == 0:
        print("ERROR: No PNG files in preprocessed_images/")
        print("Run preprocess.py first.")
        return

    # ── Load model ────────────────────────────────────────────
    model    = load_model()
    features = {}
    errors   = []

    print("Starting extraction...")
    print("Do NOT close this window.\n")

    for i, filename in enumerate(all_files):
        full_path = os.path.join(PREPROCESS_DIR, filename)
        vec       = extract_features(full_path, model)

        if vec is not None:
            features[filename] = vec
        else:
            errors.append(filename)

        # Progress every 500 images
        if (i + 1) % 500 == 0:
            pct = ((i + 1) / total) * 100
            print(f"  [{i+1:>6}/{total}]  {pct:5.1f}%"
                  f"  | Saved: {len(features)}"
                  f"  | Errors: {len(errors)}")

    # ── Save features.pkl ─────────────────────────────────────
    print(f"\nSaving features.pkl ...")
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  FEATURE EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"  Model         : CheXNet DenseNet-121 NIH")
    print(f"  Vector dim    : 1024")
    print(f"  Total images  : {total}")
    print(f"  Vectors saved : {len(features)}")
    print(f"  Errors        : {len(errors)}")
    print(f"  Saved to      : {FEATURES_PATH}")
    print("=" * 55)
    print("\nNext step: run  python search.py")

    if errors:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    build_feature_database()