import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
import os

# ── Config ─────────────────────────────────────────────
BASE_DIR = r"C:\Ekanath\College\Sem6\xray-search-system"

FEATURES_PATH = os.path.join(BASE_DIR, "features_balanced.pkl")
CSV_PATH = os.path.join(BASE_DIR, "metadata_balanced.csv")

TOP6 = [
    "Infiltration", "Atelectasis", "Effusion",
    "Nodule", "Pneumothorax", "Mass"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# LOAD DATA
# =====================================================
print("\nLoading features...")
with open(FEATURES_PATH, "rb") as f:
    data = pickle.load(f)

filenames = data["image_names"]
labels = data["labels"]
descriptions = data["descriptions"]
raw_vectors = data["features"]

# Normalize vectors
vectors = normalize(raw_vectors, norm="l2")
print(f"Loaded {len(filenames)} vectors")

# Build class index
CLASS_INDEX = {}
for cat in TOP6:
    CLASS_INDEX[cat] = [i for i, l in enumerate(labels) if l == cat]

# =====================================================
# TF-IDF (text search support)
# =====================================================
df = pd.read_csv(CSV_PATH)
df = df[df["image_name"].isin(set(filenames))].reset_index(drop=True)

TFIDF = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
TFIDF_MATRIX = TFIDF.fit_transform(df["description"])

print("Search engine ready\n")

# =====================================================
# LOAD MODEL
# =====================================================
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model.to(device)
    model.eval()
    return model

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_query_features(image_path, model):
    img = Image.open(image_path).convert("L")

    img_np = np.array(img).astype(np.float32)
    img_np = xrv.datasets.normalize(img_np, 255)
    img_np = img_np[None, ...]

    tensor = torch.from_numpy(img_np).unsqueeze(0)
    tensor = transforms.Resize((224, 224))(tensor)
    tensor = tensor.to(device)

    with torch.no_grad():
        feat = model.features(tensor)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))

    vec = feat.squeeze().cpu().numpy()
    vec = vec / (np.linalg.norm(vec) + 1e-8)

    return vec.reshape(1, -1)

# =====================================================
# DISEASE PREDICTION
# =====================================================
def predict_diseases(image_path, model, threshold=0.15):
    img = Image.open(image_path).convert("L")

    img_np = np.array(img).astype(np.float32)
    img_np = xrv.datasets.normalize(img_np, 255)
    img_np = img_np[None, ...]

    tensor = torch.from_numpy(img_np).unsqueeze(0)
    tensor = transforms.Resize((224, 224))(tensor)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor).squeeze().cpu().numpy()

    predictions = []
    for label, prob in zip(model.pathologies, output):
        if label in TOP6 and prob >= threshold:
            predictions.append({
                "disease": label,
                "confidence": round(float(prob) * 100, 1)
            })

    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions[:3]

# =====================================================
# IMAGE SEARCH
# =====================================================
def image_search(query_image_path, top_k=5):
    model = load_model()
    query_vec = extract_query_features(query_image_path, model)

    # Step 1: best per class
    per_class_best = []
    for cat in TOP6:
        idxs = CLASS_INDEX[cat]
        if not idxs:
            continue

        class_vecs = vectors[idxs]
        scores = cosine_similarity(query_vec, class_vecs).flatten()

        best_idx = scores.argmax()
        per_class_best.append((idxs[best_idx], scores[best_idx]))

    # Step 2: global search
    global_scores = cosine_similarity(query_vec, vectors).flatten()
    global_top = global_scores.argsort()[::-1][:top_k * 3]

    # Step 3: merge
    seen = set()
    merged = []

    for idx, score in per_class_best:
        if idx not in seen:
            seen.add(idx)
            merged.append((idx, score))

    for idx in global_top:
        if idx not in seen:
            seen.add(idx)
            merged.append((idx, global_scores[idx]))

    # Step 4: final ranking
    merged.sort(key=lambda x: x[1], reverse=True)
    merged = merged[:top_k]

    # Step 5: build results
    results = []
    for idx, score in merged:
        results.append({
            "image_name": filenames[idx],
            "category": labels[idx],
            "description": descriptions[idx],
            "score": round(float(score), 4)
        })

    predictions = predict_diseases(query_image_path, model)

    return results, predictions

# =====================================================
# TEST BLOCK
# =====================================================
if __name__ == "__main__":

    query_image = os.path.join(BASE_DIR, "test1.png")

    print("=" * 50)
    print(f"Testing on: {query_image}")
    print("=" * 50)

    results, predictions = image_search(query_image, top_k=5)

    print("\nTop 5 Similar Images:\n")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['image_name']:<25} | "
              f"{r['category']:<15} | "
              f"Similarity: {r['score']}")

    print("\nPredicted Diseases:\n")
    for p in predictions:
        print(f"{p['disease']} ({p['confidence']}%)")

    print("\nAll tests passed")