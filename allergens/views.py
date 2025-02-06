import os
import json
import base64
from dotenv import load_dotenv

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from celery.result import AsyncResult

from .utils.image_helpers import rotate_image, crop_image
from .utils.barcode_helpers import BarcodeReader
from .utils.api_helpers import mock_get_ingredients
from .utils.file_uploader import save_to_temp_file

from .tasks import analyze_ingredients_and_allergens, get_task_responses
from allergens.models import Users

load_dotenv()

UPLOAD_DIR = "./uploads"
UPLOAD_DIRS = "./uploaded_images"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIRS, exist_ok=True)

@csrf_exempt
def upload_base64(request):
    print("views")
    
    try :
        image_data = None
        user_id = None
        if request.method != 'POST':
            return JsonResponse(
                {"error": "Invalid HTTP method. Use POST."}, 
                status=405
            )

        if request.content_type == 'application/json':
            print("application/json")
            data = json.loads(request.body)
            image_data = data.get("image")
            user_id = data.get("userid")
            
        elif request.content_type =='multipart/form-data':
            print("multipart/form-data")
            image_file = request.FILES.get('image') 
            user_id = request.POST.get('userid')
            image_data = save_to_temp_file(image_file)            
        
        if not image_data:
                return JsonResponse(
                    {"error": "No image data provided"}, 
                    status=401
                )
        
        try:
            image_bytes = base64.b64decode(image_data)
        except base64.binascii.Error:
            return JsonResponse(
                {"error": "Invalid Base64 data"}, 
                status=401
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
        

        #todo
        # gen_openai = identify_harmful_ingredients.apply_async(args=[ingredients])
        # allergen_detection_result = detect_allergens_from_ingredients.apply_async(
        #     args=[user_allergens, 
        #     ingredients]
        # )
        
        
        
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
            "allergens": responses[1].get("detected_allergens", []),
            "safe": responses[1].get("safe", True),
            "hazard": responses[0]["hazard"],
            "Long": responses[0]["long"],
            "Recommend": "Maximum of once a week",
            "generated_text": responses[0],
        }

        return JsonResponse(result, status=200)

    except Exception as e:
        return JsonResponse(
            {"error": f"Failed to decode and save image: {str(e)}"}, 
            status=400
        )
