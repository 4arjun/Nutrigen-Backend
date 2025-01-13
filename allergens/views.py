import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

MODEL_PATH = "allergens/ml/allergen_model.pkl"
VECTORIZER_PATH = "allergens/ml/vectorizer.pkl"
MLB_PATH = "allergens/ml/mlb.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
mlb = joblib.load(MLB_PATH)

@csrf_exempt
def detect_allergens(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_allergens = data.get("user_allergens", [])
        ingredients = data.get("ingredients", [])

        ingredient_text = ", ".join(ingredients)

        X_input = vectorizer.transform([ingredient_text])

        predicted_allergens_binary = model.predict(X_input)
        predicted_allergens = mlb.inverse_transform(predicted_allergens_binary)[0]

        detected_allergens = set(predicted_allergens).intersection(set(user_allergens))

        print("Predicted Allergens:", predicted_allergens)

        print({
            "detected_allergens": list(detected_allergens),
            "safe": not bool(detected_allergens)
        })

        return JsonResponse({
            "detected_allergens": list(detected_allergens),
            "safe": not bool(detected_allergens)
        })

    return JsonResponse({"error": "Invalid request"}, status=400)
