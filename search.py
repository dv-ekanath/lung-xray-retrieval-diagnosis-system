₹₹import pickle
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
import torchvision.transforms as transforms

# Load features
with open("features.pkl", "rb") as f:
    database = pickle.load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xrv.models.DenseNet(weights="densenet121-res224-nih")
model.to(device)
model.eval()


# Extract feature from query image
def extract_feature(image_path):
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

    return feat.squeeze().cpu().numpy()


# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Search
def search(query_path, top_k=5):
    query_vec = extract_feature(query_path)

    scores = []
    for filename, vec in database.items():
        sim = cosine_similarity(query_vec, vec)
        scores.append((filename, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]


# Run
if __name__ == "__main__":
    query_image = "test1.png"   # 🔥 change this

    results = search(query_image, top_k=5)

    print("\nTop matches:")
    for fname, score in results:
        print(f"{fname}  | similarity: {score:.4f}")