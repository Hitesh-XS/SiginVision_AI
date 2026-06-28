from django.contrib import admin

# Register your models here.
from .models import Gesture,DatasetSample,PredictionHistory,DynamicDatasetSample,SentenceHistory

admin.site.register(Gesture)
admin.site.register(PredictionHistory)
admin.site.register(DatasetSample)
admin.site.register(DynamicDatasetSample)
admin.site.register(SentenceHistory)
