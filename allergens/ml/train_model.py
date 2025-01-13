import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

data = pd.read_csv("allergens/data/allergen_data.csv")
print(data.columns)


data = data.dropna(subset=["Ingredient"]) 
X = data['Ingredient'].fillna("unknown")  
y = data['Allergen'].str.strip("[]").str.replace("'", "").str.split(", ")
y = data['Allergen'].str.strip("[]").str.replace("'", "").str.split(", ")
y = y.apply(lambda allergens: [allergen.lower() for allergen in allergens])


vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

mlb = MultiLabelBinarizer()
y_transformed = mlb.fit_transform(y)

model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
model.fit(X_transformed, y_transformed)
joblib.dump(model, "allergens/ml/allergen_model.pkl")
joblib.dump(vectorizer, "allergens/ml/vectorizer.pkl")
joblib.dump(mlb, "allergens/ml/mlb.pkl")
print("Model, vectorizer, and label binarizer saved successfully!")
