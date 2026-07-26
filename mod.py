import torch
import torch.nn as nn
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

BATCH_SIZE = 32   # 🔥 You can tune this (16 / 32 / 64)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================
# LOAD MODEL
# =============================================================
def load_model():
    print("Loading CheXNet (DenseNet-121 NIH)...")

    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print("CheXNet loaded ✓\n")
    return model


# =============================================================
# PREPROCESS IMAGE → TENSOR
# =============================================================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")

    img_np = np.array(img).astype(np.float32)
    img_np = xrv.datasets.normalize(img_np, 255)

    img_np = img_np[None, ...]  # (1, H, W)
    tensor = torch.from_numpy(img_np)

    resize = transforms.Resize((224, 224))
    tensor = resize(tensor)

    return tensor


# =============================================================
# BATCH FEATURE EXTRACTION
# =============================================================
def extract_features_batch(image_paths, model):
    tensors = []

    for path in image_paths:
        try:
            tensor = preprocess_image(path)
            tensors.append(tensor)
        except:
            tensors.append(None)

    # Filter valid tensors
    valid_data = [(p, t) for p, t in zip(image_paths, tensors) if t is not None]

    if len(valid_data) == 0:
        return {}, image_paths

    paths, tensors = zip(*valid_data)

    batch = torch.stack(tensors).to(device)  # 🔥 move batch to GPU

    with torch.no_grad():
        features = model.features(batch)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))

    features = features.squeeze(-1).squeeze(-1)  # (B, 1024)
    features = features.cpu().numpy()

    result = {}
    for path, vec in zip(paths, features):
        filename = os.path.basename(path)
        result[filename] = vec

    failed = [os.path.basename(p) for p, t in zip(image_paths, tensors) if t is None]

    return result, failed


# =============================================================
# BUILD FEATURE DATABASE (CUDA + BATCHING)
# =============================================================
def build_feature_database():

    if not os.path.exists(PREPROCESS_DIR):
        print("ERROR: preprocessed_images/ not found.")
        return

    all_files = [
        f for f in os.listdir(PREPROCESS_DIR)
        if f.endswith(".png")
    ]

    total = len(all_files)

    print("=" * 55)
    print("  CUDA FEATURE EXTRACTION")
    print("=" * 55)
    print(f"Images    : {total}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device    : {device}")
    print("=" * 55 + "\n")

    if total == 0:
        print("No images found.")
        return

    model = load_model()

    features = {}
    errors = []

    print("Starting extraction...\n")

    for i in range(0, total, BATCH_SIZE):

        batch_files = all_files[i:i + BATCH_SIZE]
        batch_paths = [os.path.join(PREPROCESS_DIR, f) for f in batch_files]

        batch_features, failed = extract_features_batch(batch_paths, model)

        features.update(batch_features)
        errors.extend(failed)

        # Progress
        done = i + len(batch_files)
        pct = (done / total) * 100

        print(f"[{done:>6}/{total}] {pct:5.1f}% | Saved: {len(features)} | Errors: {len(errors)}")

    # Save
    print("\nSaving features.pkl ...")
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"Saved vectors : {len(features)}")
    print(f"Errors        : {len(errors)}")
    print(f"Path          : {FEATURES_PATH}")
    print("=" * 55)


# =============================================================
# ENTRY
# =============================================================
if __name__ == "__main__":
    build_feature_database()
print(torch.cuda.get_device_name(0))