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
        
        #print(response.json())
        print("openfoodfacts")
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
