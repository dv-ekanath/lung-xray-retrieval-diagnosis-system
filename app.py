from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import json

from balanced_search import image_search

app = Flask(__name__)
CORS(app)

BASE_DIR = r"C:\Ekanath\College\Sem6\xray-search-system"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD KNOWLEDGE BASE
# =========================
with open("disease_knowledge.json") as f:
    KNOWLEDGE = json.load(f)

# =========================
# RISK LOGIC
# =========================
def get_risk_level(predictions):
    if not predictions:
        return "Low"

    conf = float(predictions[0]["confidence"])

    # Thresholds: Low (0-50], Medium (50-75], High (>75)
    if conf > 75:
        return "High"
    if conf > 50:
        return "Medium"
    return "Low"


def get_disease_knowledge(disease, risk):
    """
    Supports both schema versions:
    1) disease -> risk -> details
    2) disease -> details
    """
    if not disease or disease not in KNOWLEDGE:
        return {}

    entry = KNOWLEDGE[disease]
    if isinstance(entry, dict) and ("Low" in entry or "Medium" in entry or "High" in entry):
        return entry.get(risk, {}).copy()
    if isinstance(entry, dict):
        return entry.copy()
    return {}

# =========================
# SEARCH API
# =========================
@app.route("/search-image", methods=["POST"])
def search_image_api():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    filename = str(uuid.uuid4()) + ".png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # 🔹 Step 1: Core retrieval
        results, predictions = image_search(filepath, top_k=5)

        main_disease = predictions[0]["disease"] if predictions else None
        risk = get_risk_level(predictions)

        # 🔹 Step 2: Local structured knowledge retrieval
        knowledge = get_disease_knowledge(main_disease, risk)

        # 🔹 Step 3: Final response
        return jsonify({
            "results": results,
            "predictions": predictions,
            "risk": risk,
            "knowledge": knowledge
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# SERVE IMAGES
# =========================
@app.route("/image/<filename>")
def get_image(filename):
    return send_from_directory("preprocessed_images", filename)


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)