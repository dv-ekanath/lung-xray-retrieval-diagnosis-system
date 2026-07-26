from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Disease knowledge base
disease_data = {
    "No Finding": "healthy no symptoms normal chest xray",
    "Infiltration": "fever cough chest pain breathing difficulty infection lungs",
    "Atelectasis": "collapsed lung shortness of breath shallow breathing",
    "Effusion": "fluid in lungs chest pain breathing difficulty heaviness",
    "Nodule": "small lung growth mild cough sometimes no symptoms",
    "Pneumothorax": "collapsed lung sudden chest pain sharp breathing difficulty"
}

# Step 2: Prepare data
diseases = list(disease_data.keys())
texts = list(disease_data.values())

# Step 3: Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Step 4: Prediction function
def predict_disease_from_text(user_input):
    user_vec = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vec, X)[0]

    results = sorted(
        zip(diseases, similarity),
        key=lambda x: x[1],
        reverse=True
    )

    return results[:3]  # top 3 matches     