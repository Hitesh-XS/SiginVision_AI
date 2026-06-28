from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import Gesture, DatasetSample, DynamicDatasetSample
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
from .services.static_prediction_service import StaticPredictionService



@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict_static(request):
    image = request.FILES.get("image")

    if not image:
        return Response({
            "success": False,
            "hand_detected": False,
            "error": "image is required",
            "gesture": None,
            "raw_gesture": None,
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "threshold_percent": 80.0,
            "landmark_count": 0,
            "top_predictions": []
        }, status=400)

    try:
        service = StaticPredictionService()
        result = service.predict_from_image(image)

        threshold = 0.80

        if not result.get("success"):
            return Response({
                "success": False,
                "hand_detected": False,
                "error": result.get("error", "No hand detected"),
                "gesture": None,
                "raw_gesture": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "threshold_percent": threshold * 100,
                "landmark_count": 0,
                "top_predictions": []
            }, status=200)

        confidence = result.get("confidence", 0.0)
        margin = result.get("margin", 0.0)
        raw_gesture = result.get("gesture")

        confidence_threshold = 0.60
        margin_threshold = 0.05

        if confidence < confidence_threshold or margin < margin_threshold:
            final_gesture = "Unknown"
        else:
            final_gesture = raw_gesture

        return Response({
            "success": True,
            "hand_detected": True,
            "gesture": final_gesture,
            "raw_gesture": raw_gesture,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "margin": round(margin, 4),
            "margin_percent": round(margin * 100, 2),
            "threshold_percent": confidence_threshold * 100,
            "margin_threshold_percent": margin_threshold * 100,
            "landmark_count": result.get("landmark_count", 0),
            "top_predictions": result.get("top_predictions", []),
            "hand_quality": result.get("hand_quality", {})
        })

    except Exception as e:
        return Response({
            "success": False,
            "hand_detected": False,
            "error": str(e),
            "gesture": None,
            "raw_gesture": None,
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "threshold_percent": 80.0,
            "landmark_count": 0,
            "top_predictions": []
        }, status=500)

def predict_page(request):
    return render(request, "vision/predict.html")

@api_view(["POST"])
def create_gesture(request):
    name = request.data.get("name")
    gesture_type = request.data.get("gesture_type", "static")

    if not name:
        return Response(
            {"error": "gesture name is required"},
            status=400
        )

    name = name.strip()

    gesture, created = Gesture.objects.get_or_create(
        name=name,
        defaults={"gesture_type": gesture_type}
    )

    return Response({
        "id": gesture.id,
        "name": gesture.name,
        "gesture_type": gesture.gesture_type,
        "created": created
    })

from .services.dynamic_prediction_service import DynamicPredictionService
def dynamic_predict_page(request):
    return render(request, "vision/dynamic_predict.html")


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict_dynamic(request):
    image_files = request.FILES.getlist("frames")

    if not image_files:
        return Response({
            "success": False,
            "error": "frames are required"
        }, status=400)

    try:
        service = DynamicPredictionService()
        result = service.predict_from_image_files(image_files, frame_count=30)

        if not result.get("success"):
            return Response({
                "success": False,
                "error": result.get("error", "Dynamic prediction failed"),
                "gesture": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "valid_frames": result.get("valid_frames", 0),
                "top_predictions": []
            }, status=200)

        confidence = result.get("confidence", 0.0)
        margin = result.get("margin", 0.0)
        raw_gesture = result.get("gesture")

        confidence_threshold = 0.60
        margin_threshold = 0.05

        if confidence < confidence_threshold or margin < margin_threshold:
            final_gesture = "Unknown"
        else:
            final_gesture = raw_gesture

        return Response({
            "success": True,
            "gesture": final_gesture,
            "raw_gesture": raw_gesture,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "margin": round(margin, 4),
            "margin_percent": round(margin * 100, 2),
            "threshold_percent": confidence_threshold * 100,
            "margin_threshold_percent": margin_threshold * 100,
            "valid_frames": result.get("valid_frames", 0),
            "top_predictions": result.get("top_predictions", [])
        })

    except Exception as e:
        return Response({
            "success": False,
            "error": str(e)
        }, status=500)

@api_view(["GET"])
def dynamic_sample_counts(request):
    data = []

    gestures = Gesture.objects.filter(gesture_type="dynamic").order_by("name")

    for gesture in gestures:
        data.append({
            "id": gesture.id,
            "name": gesture.name,
            "gesture_type": gesture.gesture_type,
            "sample_count": DynamicDatasetSample.objects.filter(gesture=gesture).count()
        })

    return Response(data)

def signvision_app_page(request):
    return render(request, "vision/signvision_app.html")

