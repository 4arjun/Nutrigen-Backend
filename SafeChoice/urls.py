"""
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.urls import path
from allergens.views import upload_base64

urlpatterns = [
    path('admin/', admin.site.urls),
    path("upload-base64/",upload_base64,name="upload_base64")
]

