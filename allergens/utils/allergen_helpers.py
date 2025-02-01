import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import inflect
import joblib
from transformers import AutoTokenizer, AutoModel

# Initialize models
MODEL_PATH = "allergens/ml/allergen_bert_tfidf_ensemble_model.pkl"
VECTORIZER_PATH = "allergens/ml/vectorizer.pkl" 
MLB_PATH = "allergens/ml/mlb.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
mlb = joblib.load(MLB_PATH)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
inflect_engine = inflect.engine()

def normalize_allergen(allergen):
    allergen = allergen.lower()
    return inflect_engine.singular_noun(allergen) or allergen

def generate_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      padding=True, max_length=512)
    print("model2")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def detect_allergens_from_ingredients(user_allergens, ingredients):
    try:
        user_allergens = [normalize_allergen(a) for a in user_allergens]
        ingredient_text = ", ".join(ingredients)
        
        X_tfidf = vectorizer.transform([ingredient_text])
        X_bert = np.array([generate_bert_embedding(ingredient_text)])
        X_combined = hstack([X_tfidf, X_bert])
        
        user_allergen_embeddings = [generate_bert_embedding(allergen) 
                                  for allergen in user_allergens]
        
        predicted_allergens_binary = model.predict(X_combined)
        predicted_allergens = [normalize_allergen(a) for a in 
                             mlb.inverse_transform(predicted_allergens_binary)[0]]
        
        detected_allergens = set()
        for allergen, allergen_embedding in zip(user_allergens, user_allergen_embeddings):
            similarity = cosine_similarity(X_bert, [allergen_embedding])[0][0]
            if similarity > 0.8:
                detected_allergens.add(allergen)
                
            for predicted_allergen in predicted_allergens:
                if fuzz.ratio(allergen, predicted_allergen) > 85:
                    detected_allergens.add(predicted_allergen)
                    
        detected_allergens.update(set(predicted_allergens).intersection(set(user_allergens)))
        print("model3")
        return {
            "detected_allergens": list(detected_allergens),
            "safe": not bool(detected_allergens)
        }
        
    except Exception as e:
        return {"error": str(e), "safe": False}
