import json
import os
import re
import base64
import requests
import cv2
import joblib
import numpy as np
import pandas as pd
import inflect
import torch
from PIL import Image
from pyzbar.pyzbar import decode
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from openai import OpenAI
from dotenv import load_dotenv
from allergens.models import Users


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
UPLOAD_DIR = "./uploads"
UPLOAD_DIRS = "./uploaded_images"
MODEL_PATH = "allergens/ml/allergen_bert_tfidf_ensemble_model.pkl"
VECTORIZER_PATH = "allergens/ml/vectorizer.pkl" 
MLB_PATH = "allergens/ml/mlb.pkl"

# Create upload directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIRS, exist_ok=True)

# Load ML models
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
mlb = joblib.load(MLB_PATH)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
inflect_engine = inflect.engine()



def BarcodeReader(image_path):
    """Read barcode from image"""
    img = cv2.imread(image_path)
    detectedBarcodes = decode(img)
    
    if not detectedBarcodes:
        return "error:barcode not detected"
    
    barcode_data = []
    for barcode in detectedBarcodes:
        barcode_data.append({
            "data": barcode.data.decode("utf-8"),
            "type": barcode.type
        })
        
    non_url_data = [item['data'] for item in barcode_data 
                    if 'data' in item and not is_url(item['data'])]
    
    return non_url_data[0] if non_url_data else None

def is_url(data):
    """Check if string is URL"""
    return re.match(r'^https?://', data) is not None

def crop_image(file_path):
    """Crop image into square"""
    with Image.open(file_path) as img:
        width, height = img.size
        box_size = min(width, height)
        
        left = (width - box_size) / 2
        top = (height - box_size) / 2
        right = left + box_size
        bottom = top + box_size
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_file_path = os.path.join(UPLOAD_DIR, "cropped_image.jpg")
        cropped_img.save(cropped_file_path)
        
    return cropped_file_path

def normalize_allergen(allergen):
    """Normalize allergen text"""
    allergen = allergen.lower()
    return inflect_engine.singular_noun(allergen) or allergen

def generate_bert_embedding(text):
    """Generate BERT embeddings"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def detect_allergens_from_ingredients(user_allergens, ingredients):
    """Detect allergens in ingredients"""
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
        
        return {
            "detected_allergens": list(detected_allergens),
            "safe": not bool(detected_allergens)
        }
        
    except Exception as e:
        return {"error": str(e), "safe": False}

def mock_get_ingredients(barcode_data):
    """Get product info from Open Food Facts API"""
    try:
        url = f"https://world.openfoodfacts.net/api/v2/product/{barcode_data}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != 1:
            return None
            
        product = data["product"]
        nutrients = product.get("nutriments", {})
        
        ingredients_list = [ing.strip() for ing in 
                          product.get("ingredients_text", "").split(",")]
        
        nutrients_data = {
            "value": [
                {"name": "energy", "value": nutrients.get("energy-kcal_100g", 0)},
                {"name": "Fat", "value": nutrients.get("fat_100g", 0)},
                {"name": "Carbohydrates", "value": nutrients.get("carbohydrates_100g", 0)},
                {"name": "Fruits&vegetables&nuts", "value": nutrients.get("fruits-vegetables-nuts-estimate-from-ingredients_100g", 0)},
                {"name": "Proteins", "value": nutrients.get("proteins_100g", 0)},
                {"name": "Saturated Fat", "value": nutrients.get("saturated-fat_100g", 0)},
                {"name": "Sodium", "value": nutrients.get("sodium_100g", 0)},
                {"name": "Sugar", "value": nutrients.get("sugars_100g", 0)},
                {"name": "Fiber", "value": nutrients.get("fiber_100g", 0)},
                {"name": "Salt", "value": nutrients.get("salt_100g", 0)}
            ]
        }
        
        nutrients_display = {
            "value": [
                {"name": "energy", "value": f'{nutrients.get("energy-kcal_100g", 0)} Kcal'},
                {"name": "Fat", "value": f'{nutrients.get("fat_100g", 0)} g'},
                {"name": "Carbohydrates", "value": f'{nutrients.get("carbohydrates_100g", 0)} g'},
                {"name": "Fruits&vegetables&nuts", "value": nutrients.get("fruits-vegetables-nuts-estimate-from-ingredients_100g", 0)},
                {"name": "Proteins", "value": f'{nutrients.get("proteins_100g", 0)} g'},
                {"name": "Saturated Fat", "value": f'{nutrients.get("saturated-fat_100g", 0)} g'},
                {"name": "Sodium", "value": f'{nutrients.get("sodium_100g", 0)} g'},
                {"name": "Sugar", "value": f'{nutrients.get("sugars_100g", 0)} g'}
            ]
        }
        
        return (ingredients_list, product["brands"], 
                product.get("product_name",""),
                product.get("image_small_url", "No image available"),
                nutrients_display, nutrients_data)
                
    except Exception as e:
        print(f"Error fetching ingredients: {str(e)}")
        return None

def identify_harmful_ingredients(ingredient_text):
    """Analyze ingredients for health risks using OpenAI"""
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
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()
        
        try:
            json_str = content.strip('```json').strip('```').strip()
            return json.loads(json_str)
        except json.JSONDecodeError as json_err:
            print(f"Error parsing OpenAI response as JSON: {json_err}")
            return {
                "hazard": {"value": []},
                "long": {"value": []},
                "recommend": {"value": "Unable to analyze ingredients at this time."}
            }
            
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return {
            "hazard": {"value": []},
            "long": {"value": []},
            "recommend": {"value": "Unable to analyze ingredients at this time."}
        }

@csrf_exempt
def upload_base64(request):
    """Handle image upload and analysis"""
    try:
        if request.method != 'POST':
            return JsonResponse(
                {"error": "Invalid HTTP method. Use POST."}, 
                status=405
            )

        data = json.loads(request.body)
        image_data = data.get("image")
        user_id = data.get("userid")


        
        if not image_data:
            return JsonResponse(
                {"error": "No image data provided"}, 
                status=400
            )

        try:
            image_bytes = base64.b64decode(image_data)
        except base64.binascii.Error:
            return JsonResponse(
                {"error": "Invalid Base64 data"}, 
                status=400
            )

        file_path = os.path.join(UPLOAD_DIRS, "uploaded_image.jpg")
        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)

        cropped_file_path = crop_image(file_path)
        barcode_info = BarcodeReader(cropped_file_path)
        
        if barcode_info == "error:barcode not detected":
            return JsonResponse(
                {"status": "error", "message": "Barcode not detected"}, 
                status=400
            )

        ingredients, brand, name, image, nutrients, Nutri = mock_get_ingredients(barcode_info)
        gen_openai = identify_harmful_ingredients(ingredients)
        user = Users.objects.get(user_id=user_id)
        user_allergens = user.disease
        # user_allergens = user_allergens.split(",")

        user_input = {
            'sugar_level': float(user.sugar),
            'cholesterol_level': float(user.cholestrol),
            'blood_pressure': float(user.bp),
            'bmi': float(user.bmi),
            'age': int(user.age),
            'heart_rate': float(user.heartrate),
            'sugar_in_product': Nutri["value"][7]["value"],
            'salt_in_product': Nutri["value"][9]["value"],
            'saturated_fat_in_product': Nutri["value"][5]["value"],
            'carbohydrates_in_product': Nutri["value"][2]["value"]
        }
        



        allergen_detection_result = detect_allergens_from_ingredients(
            user_allergens, 
            ingredients
        )
        
        predictions = 23

        result = {
            "status": "success",
            "code": barcode_info,
            "brandName": brand,
            "name": name,
            "ingredients": ingredients,
            "image": image,
            "nutrients": nutrients,
            "Nutri": Nutri,
            "score": predictions,
            "allergens": allergen_detection_result.get("detected_allergens", []),
            "safe": allergen_detection_result.get("safe", True),
            "hazard": gen_openai["hazard"],
            "Long": gen_openai["long"],
            "Recommend": "Maximum of once a week",
            "generated_text": gen_openai,
        }

        return JsonResponse(result, status=200)

    except Exception as e:
        return JsonResponse(
            {"error": f"Failed to decode and save image: {str(e)}"}, 
            status=400
        )
