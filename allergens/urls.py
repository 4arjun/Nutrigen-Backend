from django.urls import path
from .views import upload_base64
from .consumers import MyWebSocketConsumer

urlpatterns = [
    path("upload-base64/",upload_base64,name="upload_base64")
]

websocket_urlpatterns = [
    path('ws/chat/', MyWebSocketConsumer.as_asgi()),
]