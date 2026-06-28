from django.db import models

# Create your models here.
class Gesture(models.Model):
    name = models.CharField(max_length=100,unique=True)
    gesture_type=models.CharField(
        max_length=20,
        choices=[
            ("static","Static"),
            ("dynamic","Dynamic")
        ],
        default="static"
    )
    created_at=models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.name

class DatasetSample(models.Model):
    gesture = models.ForeignKey(Gesture,on_delete=models.CASCADE)
    landmark=models.JSONField()
    image=models.ImageField(upload_to="samples/",null=True,blank=True)
    created_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.gesture.name} sample"



class PredictionHistory(models.Model):
    predicted_gesture = models.CharField(max_length=100)
    confidence = models.FloatField()
    landmark = models.JSONField(null=True,blank=True)
    created_at=models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.predicted_gesture}-{self.confidence}"

class DynamicDatasetSample(models.Model):
    gesture = models.ForeignKey(Gesture, on_delete=models.CASCADE)
    frames = models.JSONField()
    frame_count = models.IntegerField(default=30)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.gesture.name} dynamic sample"
