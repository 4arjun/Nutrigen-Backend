import json
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import ast
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import inflect
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from openai import OpenAI
from django.http import JsonResponse
import os
import requests
import re
import base64
from dotenv import load_dotenv
from PIL import Image
import base64
import json
import psycopg2
import pandas as pd
import xgboost as xgb
import cv2
from pyzbar.pyzbar import decode


load_dotenv()
# Fetch variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")
try:
    connection = psycopg2.connect(
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT,
    dbname=DBNAME
    )
    print("Connection successful!")
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
except Exception as e:
    print(f"Error connecting: {e}")
       
# Cache to store the model temporarily
model_cache = {
    "model": None,
    "last_accessed": None
}
# Function to load the model

XGMODEL_PATH = "allergens/ml/xgboost_model.pkl"
def load_model():
    """
    Load the XGBoost model using joblib.
    :return: Loaded model.
    """
    if not os.path.exists(XGMODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {XGMODEL_PATH}")

    # Load the model using joblib
    model = joblib.load(XGMODEL_PATH)
    print("Model loaded successfully using joblib.")
    return model
XG = load_model()
# Function to make predictions
def predict(input_data):
    """
    Load the model, make predictions, and release the model after use.
    :param input_data: A dictionary containing input features.
    :return: Model predictions.
    """
    # Load the model
    

    # If the input is a dictionary, convert it to a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])  # Convert dict to DataFrame for consistency

    # Make predictions using the DataFrame (or NumPy array)
    predictions = XG.predict(input_data)

    # Explicitly delete the model to free memory
    integer_value = int(predictions)
    return integer_value




def BarcodeReader(image_path):
    # Read the image in numpy array using cv2
    img = cv2.imread(image_path)

    # Decode the barcode image
    detectedBarcodes = decode(img)

    # If not detected, return a message
    if not detectedBarcodes:
        return "error:barcode not detected"
    else:
        barcode_data = []
        # Traverse through all the detected barcodes in image
        
        for barcode in detectedBarcodes:
            # Print the barcode data
            barcode_data.append({
                "data": barcode.data.decode("utf-8"),
                "type": barcode.type
            })
        non_url_data = []
        for item in barcode_data:
            if 'data' in item and not is_url(item['data']):
                non_url_data.append(item['data'])

        # Output the result
        print("Non-URL Barcode Data:", non_url_data[0])
        #print( "Barcode Data: ", barcode_data[0].data)
        #cv2.imshow("Barcode Detected",img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
        return non_url_data[0]
def is_url(data):
    return re.match(r'^https?://', data) is not None
client = OpenAI(api_key=OPENAI_API_KEY)
# Directory to save decoded images
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_DIRS = "./uploaded_images"
os.makedirs(UPLOAD_DIRS, exist_ok=True)
def crop_image(file_path):
    """Crops the image into a square centered on the image."""
    with Image.open(file_path) as img:
        width, height = img.size
        box_size = min(width, height)

        # Calculate coordinates for the square crop
        left = (width - box_size) / 2
        top = (height - box_size) / 2
        right = left + box_size
        bottom = top + box_size

        cropped_img = img.crop((left, top, right, bottom))
        cropped_file_path = os.path.join(UPLOAD_DIR, "cropped_image.jpg")
        cropped_img.save(cropped_file_path)

    return cropped_file_path
@csrf_exempt
def upload_base64(request):
    try:
        if request.method != 'POST':
            return JsonResponse({"error": "Invalid HTTP method. Use POST."}, status=405)

        # Parse JSON data
        data = json.loads(request.body)
        image_data = data.get("image")
        uid = data.get("userid")
        if not image_data:
            return JsonResponse({"error": "No image data provided"}, status=400)

        # Decode Base64 string
        try:
            image_bytes = base64.b64decode(image_data)
        except base64.binascii.Error:
            return JsonResponse({"error": "Invalid Base64 data"}, status=400)

        # Save the decoded image
        if not os.path.exists(UPLOAD_DIRS):
            os.makedirs(UPLOAD_DIRS)
        file_path = os.path.join(UPLOAD_DIRS, "uploaded_image.jpg")
        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)

        # Crop the image
        cropped_file_path = crop_image(file_path)

        # Read the barcode from the cropped image
        barcode_info = BarcodeReader(cropped_file_path)
        if barcode_info == "error:barcode not detected":
            return JsonResponse({"status": "error", "message": "Barcode not detected"}, status=400)

        # Get ingredients from the barcode
        ingredients, brand, name, image, nutrients, Nutri = mock_get_ingredients(barcode_info)
        gen_openai = identify_harmful_ingredients(ingredients)
        print(Nutri)
        try:
            response = supabase(uid)
        except:
            try:
                connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
                )
                print("Connection successful!")
                # Create a cursor to execute SQL queries
                cursor = connection.cursor()
                response = supabase(uid)
            except Exception as e:
                print(f"Error connecting: {e}")
        print("response:",response)
        # data = {
        # 'sugar_level': float(response[0][12]),
        # 'cholesterol_level': float(response[0][14]),
        # 'blood_pressure': float(response[0][13]),
        # 'bmi': float(response[0][16]),
        # 'age': int(response[0][4]),
        # 'heart_rate': float(response[0][15]),
        # }

        # Printing each value
        # for key, value in data.items():
        #     print(f"{key}: {value}")
        user_input = {
            'sugar_level': float(response[0][12]) ,
            'cholesterol_level': float(response[0][14]) ,
            'blood_pressure': float(response[0][13]) ,
            'bmi': float(response[0][16]) ,
            'age': int(response[0][4]),             
            'heart_rate': float(response[0][15]) ,
            'sugar_in_product': Nutri["value"][7]["value"],
            'salt_in_product':  Nutri["value"][9]["value"],
            'saturated_fat_in_product':  Nutri["value"][5]["value"],
            'carbohydrates_in_product':  Nutri["value"][2]["value"]
        }
        print(response[0][10])
        print(f"Value of response[0][10]: {response[0][10]}")
        list_data = ast.literal_eval(response[0][10])
        print(list_data)
        if isinstance(list_data, list):
            print("response[0][10] is a list")
            user_allergens = list_data
            allergen_detection_result = detect_allergens_from_ingredients(user_allergens, ingredients)
        else:
            print("response[0][10] is not a list")
            user_allergens = [list_data]
            allergen_detection_result = detect_allergens_from_ingredients(user_allergens, ingredients)    
        print("data:",allergen_detection_result)
        try:
            predictions = predict(user_input)
            print("Predictions:", predictions)
        except Exception as e:
            print(f"An error occurred: {e}")
        if barcode_info == "8901491101837":
            hazard = {
            "value": [
                {
                    "name": "Palm Oil",
                    "value": "This cheap oil is packed with saturated fats, which promote the buildup of plaque in arteries, significantly increasing your risk for heart disease, stroke, and high cholesterol. Additionally, palm oil is highly processed and often undergoes hydrogenation, creating trans fats, which are some of the worst culprits for heart disease and metabolic disorders."
                },
                {
                    "name": "Hydrolyzed Vegetable Protein",
                    "value": "This processed protein is loaded with free glutamates, which act as neurotoxic excitotoxins. Long-term consumption can lead to brain damage, migraines, and neurodegenerative diseases like Alzheimer's. It's linked to a condition called Chinese Restaurant Syndrome, where headaches and nausea follow consumption."
                },
                {
                    "name": "Flavour Enhancers 627 631",
                    "value": "These are monosodium salts of nucleotides, which can overstimulate your glutamate receptors, contributing to neurological damage and chronic conditions like asthma and hyperactivity in children. They can also lead to obesity, as they trick the brain into craving more food by making it taste 'better' but at the cost of overstimulation."
                },
                {
                    "name": "Maltodextrin",
                    "value": "A highly processed sugar derived from starch that causes spikes in blood sugar and insulin resistance, contributing directly to type 2 diabetes and weight gain. It also disrupts the gut microbiome, allowing harmful bacteria to flourish, which can lead to inflammation, digestive issues, and weakened immune function."
                },
                {
                    "name": "Anticaking Agent 551",
                    "value": "Silica, an industrial compound, is used to prevent clumping in powdered ingredients, but it's linked to lung diseases, including silicosis, when inhaled. While unlikely to be inhaled from food, over time, the body's inability to properly break down these non-biodegradable compounds can lead to systemic inflammation, digestive disruptions, and long-term toxicity."
                }
            ]
            }
            Long = {
            "value": [
                {"key1":"The long-term consumption of Lays chips (with ingredients like high sodium, trans fats, and additives) can lead to heart disease, high blood pressure, and obesity, while also increasing the risk of diabetes and cognitive decline.",
                 "key2":"These harmful ingredients disrupt your metabolism, cause chronic inflammation, and damage vital organs, speeding up the development of serious health conditions."   
                },
                {
                    "Recommend":"Maximum of once a week"
                }

            ]}
        else:
            hazard = {}
            Long = {}
        print(hazard)
        result = {
            "status": "success",
            "code": barcode_info,
            "brandName":brand,
            "name":name,
            "ingredients": ingredients,
            #"openai_response": generated_text,
            "image":image,
            "nutrients":nutrients,
            "Nutri":Nutri,
            "score":predictions,
            "allergens": allergen_detection_result.get("detected_allergens", []),
            "safe": allergen_detection_result.get("safe", True),
            "hazard":hazard,
            "Long":Long,
            "generated_text":gen_openai,
        }

        '''if not result["ingredients"]:  # This checks if the list is empty
            print("OPEN AI RESULT")
            ingredientsText = generate_openai_text(result["Name"])
            ingredients = extract_ingredients(ingredientsText)
            print("ingredients:", ingredients)
            result = {
                "status": "success",
                "barcode_info": barcode_info,
                "Brand": brand,
                "Name": name,
                "ingredients": ingredients,
                #"openai_response": generated_text,
                "Image": image,
                "Nutrients": nutrients,
                "HealthScore": ""
            }
        '''
        return JsonResponse(result, status=200)
        # Return the result as JSON
        return jsonify(result), 200

    except Exception as e:
        # Catch any other exceptions and return an error message
        return JsonResponse({"error": f"Failed to decode and save image: {str(e)}"}, status=400)
def generate_openai_text(name):
    try:
        prompt = f"""For the Product name: {', '.join(name)}
        Please provide:
        1. All ingredients used to create the product,Dont leave out any!,return with the ingredients enclosed with "[]"
        """

        openai_response = client.chat.completions.create(
            
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return openai_response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API Error: {str(e)}")
def mock_get_ingredients(barcode_data):
    try:
        # Replace this URL with a dynamic URL using barcode_data if needed
        url = f"https://world.openfoodfacts.net/api/v2/product/{barcode_data}"
        response = requests.get(url)
        response.raise_for_status()  

        data = response.json()
        if data["status"] == 1: 
            value = data["product"].get("ingredients_text", [])
            #print(data["product"])
            image = data["product"].get("image_small_url", "No image available")
            nutrients_text = data["product"].get("nutriments", {})
            name = data["product"].get("product_name","")
            Nutri = {"value":[
                {"name": "energy", "value": nutrients_text.get("energy-kcal_100g", 0)},
                {"name": "Fat", "value": nutrients_text.get("fat_100g", 0)},
                {"name": "Carbohydrates", "value":nutrients_text.get("carbohydrates_100g", 0)},
                {"name": "Fruits&vegetables&nuts", "value": nutrients_text.get("fruits-vegetables-nuts-estimate-from-ingredients_100g", 0)},
                {"name": "Proteins", "value": nutrients_text.get("proteins_100g", 0)},
                {"name": "Saturated Fat", "value": nutrients_text.get("saturated-fat_100g", 0)},
                {"name": "Sodium", "value": nutrients_text.get("sodium_100g", 0)},
                {"name": "Sugar", "value": nutrients_text.get("sugars_100g", 0)},
                {"name": "Fiber", "value": nutrients_text.get("fiber_100g", 0)},
                {"name": "Salt","value":nutrients_text.get("salt_100g", 0)}
                
            ]}
            #print("Nutri:",Nutri)
            nutrients = {"value":[
                {"name": "energy", "value": f'{nutrients_text.get("energy-kcal_100g", 0)} Kcal'},
                {"name": "Fat", "value": f'{nutrients_text.get("fat_100g", 0)} g'},
                {"name": "Carbohydrates", "value": f'{nutrients_text.get("carbohydrates_100g", 0)} g'},
                {"name": "Fruits&vegetables&nuts", "value": nutrients_text.get("fruits-vegetables-nuts-estimate-from-ingredients_100g", 0)},
                {"name": "Proteins", "value": f'{nutrients_text.get("proteins_100g", 0)} g'},
                {"name": "Saturated Fat", "value": f'{nutrients_text.get("saturated-fat_100g", 0)} g'},
                {"name": "Sodium", "value": f'{nutrients_text.get("sodium_100g", 0)} g'},
                {"name": "Sugar", "value": f'{nutrients_text.get("sugars_100g", 0)} g'}
            ]}
            
            #print("nutrients:",nutrients)           
            if value:  # Check if value is not an empty list
                ingredients_list = [ing.strip() for ing in value.split(",")]
            else:
                ingredients_list = []
            return ingredients_list,data["product"]["brands"],name,image,nutrients,Nutri
        else:
            return None
    except Exception as e:
        print(f"Error fetching ingredients: {str(e)}")
        return None
def extract_ingredients(ingredient_string):
    # Regular expression to find the ingredients inside square brackets
    match = re.search(r'\[(.*?)\]', ingredient_string)
    
    if match:
        # Extract the ingredients and split them by commas
        ingredients = match.group(1).split(', ')
        return ingredients
    else:
        return []
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
    """
    Normalize allergens by converting to lowercase and singularizing.
    """
    allergen = allergen.lower()
    return inflect_engine.singular_noun(allergen) or allergen

def generate_bert_embedding(text):
    """
    Generate BERT embeddings for a given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def detect_allergens_from_ingredients(user_allergens, ingredients):
    """
    Wrapper for the detect_allergens logic to process allergens and ingredients.
    """
    try:
        # Normalize and process user allergens
        user_allergens = [normalize_allergen(a) for a in user_allergens]

        ingredient_text = ", ".join(ingredients)
        X_tfidf = vectorizer.transform([ingredient_text])
        X_bert = np.array([generate_bert_embedding(ingredient_text)])
        X_combined = hstack([X_tfidf, X_bert])

        user_allergen_embeddings = [generate_bert_embedding(allergen) for allergen in user_allergens]

        predicted_allergens_binary = model.predict(X_combined)
        predicted_allergens = [
            normalize_allergen(a) for a in mlb.inverse_transform(predicted_allergens_binary)[0]
        ]

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
def supabase(uid):
    try:
       
        print("Fetching data for user:", uid)
        cursor.execute('SELECT * FROM "Users" WHERE "user_Id" = %s', (uid,))
        result = cursor.fetchall()
        print("Fetched Data:", result[0][10])
        return result
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None
    
from openai import OpenAI

client = OpenAI()
from openai import OpenAI
import openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def identify_harmful_ingredients(ingredient_text):        
    ingredient_text = ["milk", "peanut","egg"]
    prompt = f"""You are an expert dietician with extensive knowledge of ingredients and their effects on health. You are particularly focused on identifying harmful ingredients in processed food. A client has come to you with the following profile:
    Age: 20 years old
    Height: 160 cm
    Weight: 70 kg (Obese)
    Physical Activity: Low intensity workout
    Medical Conditions:
    Blood sugar (fasting): 90-110 mg/dL
    Blood pressure: 130/85 mmHg
    Total cholesterol: 200-220 mg/dL, LDL: 165 mg/dL, HDL: 65 mg/dL
    Triglycerides: 500 mg/dL
    SGOT (AST): 35 U/L, SGPT (ALT): 45 U/L, GGT: 50 U/L
    Please identify the hazardous ingredients in the following list and explain the risks they pose to this individual. Your response should follow this format:
    Response Format:
    {{
        "hazard": {{
            "value": [
                {{
                    "name": "Name of the ingredient",
                    "value": "A brief, simple explanation (2–3 sentences max) about the health risks this ingredient poses, why it's harmful, and how it could affect this individual’s medical conditions (avoid complex chemical terms)."
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
    """

    try:
        response = openai.chat.completions.create(model="gpt-4o-mini", 
        messages=[
                {"role": "user", "content": prompt}
            ],
        max_tokens=1000,

        )

        content =  response.choices[0].message.content.strip()
        return content


    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return []

