from django.urls import path
from .views import detect_allergens
from .views import upload_base64

urlpatterns = [
    path("detect/", detect_allergens, name="detect_allergens"),
    path("upload-base64",upload_base64,name="upload_base64")
]
