from SafeChoice.celery import app  
from .utils.allergen_helpers import normalize_allergen,generate_bert_embedding

from celery import shared_task, group
import time 
from celery.result import AsyncResult

from openai import OpenAI
from dotenv import load_dotenv
import os 
import json 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)

@shared_task
def identify_harmful_ingredients(ingredient_text):
    prompt = f"""You are an expert dietician with extensive knowledge of ingredients and their effects on health. You are particularly focused on identifying harmful ingredients in processed food. A client has come to you with the following profile:
    Age: 20 years old
    Height: 160 cm
    Weight: 60 kg 
    Physical Activity: Low intensity
    Medical Conditions:
    Blood sugar (fasting): 90 mg/dL
    Blood pressure: 120/75 mmHg
    Total cholesterol: 200 mg/dL, LDL: 135 mg/dL, HDL: 65 mg/dL
    Triglycerides: 140 mg/dL
    SGOT (AST): 25 U/L, SGPT (ALT): 35 U/L, GGT: 40 U/L
    
    Please identify the hazardous ingredients in the following list and explain the risks they pose to this individual. Your response should follow this format:
    {{
        "hazard": {{
            "value": [
                {{
                    "name": "Name of the ingredient",
                    "value": "A brief, simple explanation (2–3 sentences max) about the health risks this ingredient poses, why it's harmful, and how it could affect this individual's medical conditions (avoid complex chemical terms)."
                }}
            ]
        }},
        "long": {{
            "value": [
                {{
                    "key1": "A summary of the long-term health risks these ingredients, when combined, could cause with regular consumption (e.g., diabetes, heart disease, liver damage, etc.).",
                    "key2": "A further explanation of how these ingredients affect overall health and contribute to chronic conditions over time."
                }}
            ]
        }},
        "recommend": {{
            "value": "A suggestion for how often this individual can consume foods with these ingredients (e.g., Maximum of once a week)."
        }}
    }}
    
    Ingredients to analyze: {ingredient_text}
    make sure to list out ingredients that have high chances of ill effects for our users. Don't list all the ingredients, instead make sure we consider the health of above user
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        print('content from gpt')
        
        try:
            json_str = content.strip('```json').strip('```').strip()
            return json.loads(json_str)
        except json.JSONDecodeError as json_err:
            print(f"Error parsing OpenAI response as JSON: {json_err}")
            return {"hazard": {"value": []}, "recommend": "Unable to analyze ingredients at this time."}
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return {"hazard": {"value": []}, "recommend": "Unable to analyze ingredients at this time."}

import inflect
import joblib
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import torch 
from scipy.sparse import hstack

MODEL_PATH = "allergens/ml/allergen_bert_tfidf_ensemble_model.pkl"
VECTORIZER_PATH = "allergens/ml/vectorizer.pkl" 
MLB_PATH = "allergens/ml/mlb.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
mlb = joblib.load(MLB_PATH)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
inflect_engine = inflect.engine()
@shared_task
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

def analyze_ingredients_and_allergens(user_allergens, ingredients):
    task_group = group(
        identify_harmful_ingredients.s(ingredients), 
        detect_allergens_from_ingredients.s(user_allergens, ingredients)
    )
    
    result = task_group.apply_async()
    return result

def get_task_responses(result):
    responses = result.get()  
    return responses

    
@app.task(soft_time_limit=60)
@app.task(track_started=True)
@app.task(rate_limit="10/m")
@app.task(bind=True, max_retries=3)
def send_email(self):
    try:
        # Logic to send email
        print("Task 3 is running")
    except Exception as exc:
        # Retry the task if it fails
        raise self.retry(exc=exc)
    

def check_task_status(request, task_id):
    result = AsyncResult(task_id)
    print({'status': result.status, 'result': result.result})

