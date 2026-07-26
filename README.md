# Lung Diseases X-Ray Image Retrieval and Symptom-Driven Diagnosis System

An AI-based chest X-ray analysis and diagnosis support system. It combines two modules into a single web platform: an **image-based analysis module** that predicts disease from an uploaded X-ray and retrieves visually similar past cases, and a **symptom-based diagnosis module** that suggests possible diseases from user-entered symptoms.

## Overview

Users can upload an X-ray image to get a predicted disease, risk level, the top 5 most visually similar cases from the dataset, and relevant medical insights/FAQ. Alternatively, users can describe symptoms in free text and get a ranked list of possible diseases along with supporting medical information — all backed by a shared knowledge base.

## System Architecture

**Analysis Module (image-based):**
```
Upload X-Ray → Preprocessing (resize, normalize) → DenseNet (CheXNet) prediction
            → CLIP feature extraction → Cosine similarity search → Top-5 similar X-rays
            → Predicted Disease + Risk Level + Medical Insights + FAQ (from Knowledge Base)
```

**Diagnosis Module (symptom-based):**
```
Enter Symptoms → Text preprocessing (lowercase, tokenize, remove stopwords)
              → TF-IDF vectorizer → Similarity matching vs symptom-disease knowledge base
              → Predicted Possible Diseases → Medical Insights (from Knowledge Base)
```

Both modules share a centralized **knowledge base** (disease info, symptoms, medical insights, FAQ) to produce a unified final output.

## Methodology

### 1. Preprocessing / Dataset Balancing
The raw NIH ChestX-ray14 dataset is heavily imbalanced (e.g. "No Finding" dominates with 60,353 samples vs. ~2,000–9,500 for disease classes), which biases retrieval. The dataset is balanced to 1,367 samples per class across 6 disease categories (Infiltration, Atelectasis, Effusion, Nodule, Pneumothorax, Mass) for fair retrieval performance.

### 2. Analysis Module (Image-Based)
- **Feature extraction:** CheXNet (DenseNet-121, trained on NIH ChestX-ray14) — chosen over ResNet-50 because ResNet-50 is trained on generic photos, while CheXNet understands lung-specific patterns from the same dataset.
- Pipeline: input PNG → resize/normalize to 224×224 → DenseNet-121 (121 layers) → average pooling → 1024-dim L2-normalized feature vector → saved to `features.pkl`.
- **Disease prediction:** DenseNet model predicts the most probable disease class, including "No Finding."
- **Image retrieval:** CLIP-based embeddings compared via cosine similarity against precomputed dataset embeddings; top 5 most similar X-rays returned with similarity scores.
- **Risk level & insights:** Qualitative risk level (Low/Medium/High) and medical insights/FAQ retrieved from the knowledge base for the predicted disease.

### 3. Diagnosis Module (Symptom-Based)
- User enters symptoms (e.g. fever, cough, chest pain) in free text.
- TF-IDF vectorization + similarity matching against a symptom-disease knowledge base ranks possible diseases.
- Relevant medical insights are retrieved for the top predicted disease(s).

### 4. Web Application
Built with Flask (backend) and HTML/CSS/JS (frontend), with separate tabs for the Analysis and Diagnosis modules.

## Metrics

Evaluated using similarity-based retrieval metrics (cosine similarity, Precision@5, Recall@5, F1 Score, MAP) on the top-5 retrieval results.

- Per-class Precision@5 is strongest for visually distinctive conditions (e.g. Pneumothorax) and weaker for classes with high visual overlap (Mass, Nodule).
- CheXNet (proposed) consistently outperforms a ResNet-50 baseline across Precision@5, Recall@5, F1 Score, and MAP — confirming the benefit of a medically-optimized backbone over a generic one.

## Project structure

```
.
├── medical-xray-ui/        # React frontend
│   ├── src/                 # App.js, styles, etc.
│   ├── public/
│   └── build/                 # generated — not tracked
├── static/                  # Flask static assets (css/js)
├── templates/                # Flask HTML templates (index.html, diagnosis.html)
├── app.py                    # Flask entrypoint
├── search.py / balanced_search.py       # similarity search logic
├── model.py / balanced_model.py         # model loading/inference
├── symptom_model.py                     # TF-IDF symptom-based diagnosis
├── preprocess.py                        # image preprocessing
├── extract_features_balanced.py         # builds CLIP/CheXNet feature embeddings
├── create_metadata.py / create_metadata_balanced.py  # builds metadata CSVs
├── balance_dataset.py                   # dataset balancing utility
├── disease_knowledge.json               # disease reference / FAQ knowledge base
├── requirements.txt
└── images_001 ... images_012/           # raw NIH ChestX-ray14 dataset (not tracked, see below)
```

## Not included in this repo

The `.gitignore` excludes large/generated/secret files. You'll need to regenerate or re-download these locally:

| File/Folder | How to get it |
|---|---|
| `chexnet.pth` | Run `python download_chexnet.py` |
| `images_001`–`images_012` | Download the NIH ChestX-ray14 dataset ([source](https://nihcc.app.box.com/v/ChestXray-NIHCC)) |
| `features.pkl` / `features_balanced.pkl` | Run `python extract_features_balanced.py` after downloading images |
| `metadata.csv` / `metadata_balanced.csv` | Run `python create_metadata.py` / `create_metadata_balanced.py` |
| `secret/` | Create locally with your own API keys/config — never commit this |

## Setup

### Backend (Flask)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend (React)

```bash
cd medical-xray-ui
npm install
npm start
```

## Notes

- `verify_environment.py` can be used to sanity-check your local setup before running the pipeline.
- `mod.py` / `test.py` / `test1.png` / `test2.png` — scratch/testing scripts and sample images used during development.
