import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

data = pd.read_csv("allergens/data/allergen_data.csv")

X = data["Ingredient List"]
y = data["Allergens"].apply(eval) 

mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y)

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

model = OneVsRestClassifier(LogisticRegression())
model.fit(X_tfidf, y_binary)

joblib.dump(model, "allergens/ml/allergen_model.pkl")
joblib.dump(vectorizer, "allergens/ml/vectorizer.pkl")
joblib.dump(mlb, "allergens/ml/mlb.pkl")
print("Model training completed and files saved.")
