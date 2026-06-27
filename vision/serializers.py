from rest_framework import serializers
from .models import *
class GestureSerializer(serializers.ModelSerializer):
    class Meta:
        model=Gesture
        fields='__all__'

class DatasetSampleSerializer(serializers.ModelSerializer):
    class Meta:
        model=DatasetSample
        fields='__all__'
class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model=PredictionHistory
        fields='__all__'
