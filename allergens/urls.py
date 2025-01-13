from django.urls import path
from .views import detect_allergens

urlpatterns = [
    path("detect/", detect_allergens, name="detect_allergens"),
]
