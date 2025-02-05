from django.urls import path
from .views import upload_base64

urlpatterns = [
    path("upload-base64/",upload_base64,name="upload_base64")
]
