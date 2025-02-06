import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.security.websocket import AllowedHostsOriginValidator
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SafeChoice.settings')

django_application = get_asgi_application()
 

from .urls import websocket_urlpatterns #dont move this line up

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
	"websocket": AllowedHostsOriginValidator(
		AuthMiddlewareStack(
			URLRouter(websocket_urlpatterns)
		)
	),
	
})