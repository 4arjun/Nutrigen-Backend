"""
ASGI config for SafeChoice project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SafeChoice.settings')

application = get_asgi_application()
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