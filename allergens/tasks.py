import json 

from celery import shared_task, group
from celery.result import AsyncResult
from celery.exceptions import SoftTimeLimitExceeded

from .utils.openai_helpers import identify_harmful_ingredients
from .utils.allergen_helpers import detect_allergens_from_ingredients

@shared_task(soft_time_limit=30, track_started=True, rate_limit="10/m")
def celery_call_gpt(ingredient_text):
    try:
        return identify_harmful_ingredients(ingredient_text)
    
    except SoftTimeLimitExceeded:
        print("Task exceeded soft time limit!")
        return {"status": "error", "message": "Soft time limit exceeded"}
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"status": "error", "message": str(e)}
        

@shared_task(soft_time_limit=30, track_started=True)
def celery_call_allergen_model (user_allergens, ingredients):
    try:
        return detect_allergens_from_ingredients(user_allergens, ingredients)
    
    except SoftTimeLimitExceeded:
        print("Task exceeded soft time limit!")
        return {"status": "error", "message": "Soft time limit exceeded"}
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"status": "error", "message": str(e)}

   

def check_task_status(task_id):
    result = AsyncResult(task_id)
    print({'status': result.status, 'result': result.result})

