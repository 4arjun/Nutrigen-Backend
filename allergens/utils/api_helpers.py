import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)

def mock_get_ingredients(barcode_data):
    try:
        url = f"https://world.openfoodfacts.org/api/v2/product/{barcode_data}"

        response = requests.get(url)
        
        print(response.json())
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
        
        try:
            json_str = content.strip('```json').strip('```').strip()
            return json.loads(json_str)
        except json.JSONDecodeError as json_err:
            print(f"Error parsing OpenAI response as JSON: {json_err}")
            return {"hazard": {"value": []}, "recommend": "Unable to analyze ingredients at this time."}
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return {"hazard": {"value": []}, "recommend": "Unable to analyze ingredients at this time."}
