import os
from openai import OpenAI
from dotenv import load_dotenv
import json 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)

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
