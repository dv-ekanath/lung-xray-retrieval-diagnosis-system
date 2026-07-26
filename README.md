# Medical X-Ray Search

A medical chest X-ray similarity search / diagnosis-assist tool. Combines a Flask backend
(feature extraction + search over a CheXNet-derived embedding space) with a React frontend.

## Project structure

```
.
├── medical-xray-ui/        # React frontend
│   ├── src/                 # App.js, styles, etc.
│   ├── public/
│   └── build/                # generated — not tracked
├── static/                  # Flask static assets (css/js)
├── templates/                # Flask HTML templates (index.html, diagnosis.html)
├── app.py                    # Flask entrypoint
├── search.py / balanced_search.py       # similarity search logic
├── model.py / balanced_model.py         # model loading/inference
├── symptom_model.py                     # symptom-based scoring
├── preprocess.py                        # image preprocessing
├── extract_features_balanced.py         # builds feature embeddings
├── create_metadata.py / create_metadata_balanced.py  # builds metadata CSVs
├── balance_dataset.py                   # dataset balancing utility
├── disease_knowledge.json               # disease reference data
├── requirements.txt
└── images_001 ... images_012/           # raw NIH ChestX-ray14 dataset (not tracked, see below)
```

## Not included in this repo

The `.gitignore` excludes large/generated/secret files. You'll need to regenerate or
re-download these locally:

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
- `mod.py` / `test.py` — [add a short note here on what these are for, if kept long-term]
