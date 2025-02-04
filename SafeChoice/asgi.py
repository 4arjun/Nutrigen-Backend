import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SafeChoice.settings')

application = get_asgi_application()

#todo : 
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from channels.sessions import SessionMiddlewareStack
from excelplay_dalalbull.routing import websocket_urlpatterns
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
	"websocket": AllowedHostsOriginValidator(
		SessionMiddlewareStack(
			URLRouter(websocket_urlpatterns)
		)
	),
	
})