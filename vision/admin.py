from django.contrib import admin

# Register your models here.
from .models import Gesture,DatasetSample,PredictionHistory

admin.site.register(Gesture)
admin.site.register(PredictionHistory)
admin.site.register(DatasetSample)
