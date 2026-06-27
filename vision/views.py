from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import Gesture,DatasetSample
from .serializers import GestureSerializer
from .services.hand_landmark_service import HandLandmarkService

from django.shortcuts import render
from django.db.models import Count
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser

@api_view(['GET'])
def gesture_list(request):
    gestures=Gesture.objects.all()
    serializer=GestureSerializer(gestures,many=True)
    return Response(serializer.data)


@api_view(['POST'])
@parser_classes([MultiPartParser,FormParser])
def collect_sample(request):
    gesture_id=request.data.get('gesture_id')
    image=request.FILES.get("image")
    if not gesture_id:
       return Response(
           {"error":"gesture_id"},
           status=status.HTTP_400_BAD_REQUEST
       )
    if not image:
        return Response(
            {"error":"image is required"},
            status=status.HTTP_400_BAD_REQUEST
        )
    try:
        gesture=Gesture.objects.get(id=gesture_id)
    except Gesture.DoesNotExist:
        return Response(
            {"error":"gesture does not exist"},
            status=status.HTTP_404_NOT_FOUND
        )

    service = HandLandmarkService()
    landmarks = service.extract_landmarks_from_image(image)

    if landmarks is None:
        return Response(
            {"error": "No landmarks detected"},
            status=400
        )

    image.seek(0)

    sample = DatasetSample.objects.create(
        gesture=gesture,
        landmark=landmarks,
        image=image
    )

    return Response({
        "message": "Sample collected successfully",
        "sample_id": sample.id,
        "gesture": gesture.name,
        "landmark_count": len(landmarks)
    })

def collector_page(request):
    return render(request,'vision/collector.html')

@api_view(["GET"])
def sample_counts(request):
    data=(
        Gesture.objects
        .annotate(sample_count=Count("datasetsample"))
        .values("id","name","gesture_type","sample_count")
        .order_by("name")

    )
    return Response(list(data))