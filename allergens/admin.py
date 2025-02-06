from django.contrib import admin
from .models import Users, History, Saved

# Register your models here
admin.site.register(Users)
admin.site.register(History)
admin.site.register(Saved)
