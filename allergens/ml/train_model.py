import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import cross_val_score
import joblib
import torch

data = pd.read_csv("allergens/data/allergen_data.csv")
print(data.columns)

data = data.dropna(subset=["Ingredient"])
X = data['Ingredient'].fillna("unknown")
y = data['Allergen'].str.strip("[]").str.replace("'", "").str.split(", ")
y = y.apply(lambda allergens: [allergen.lower() for allergen in allergens])

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def generate_bert_embedding(text):
    """
    Generate BERT embeddings for a given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Generating BERT embeddings...")
X_bert = np.array([generate_bert_embedding(text) for text in X])

X_combined = hstack([X_tfidf, X_bert])

mlb = MultiLabelBinarizer()
y_transformed = mlb.fit_transform(y)

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier()
lr_model = LogisticRegression(max_iter=500, solver='liblinear')

ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('lr', lr_model)],
    voting='soft'
)

print("Training ensemble model with BERT and TF-IDF...")
model = OneVsRestClassifier(ensemble_model)
model.fit(X_combined, y_transformed)

joblib.dump(model, "allergens/ml/allergen_bert_tfidf_ensemble_model.pkl")
joblib.dump(vectorizer, "allergens/ml/vectorizer.pkl")
joblib.dump(mlb, "allergens/ml/mlb.pkl")
print("Model, vectorizer, and label binarizer saved successfully!")

scores = cross_val_score(model, X_combined, y_transformed, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.2f}")
