import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from django.conf import settings

from vision.services.hand_landmark_service import HandLandmarkService
from vision.services.landmark_utils import normalize_landmarks


class StaticPredictionService:
    model = None
    label_map = None

    def __init__(self):
        self.model_path = Path(settings.BASE_DIR) / "ml_models" / "static_gesture_model.keras"
        self.label_map_path = Path(settings.BASE_DIR) / "ml_models" / "static_label_map.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map file not found: {self.label_map_path}")

        if StaticPredictionService.model is None:
            StaticPredictionService.model = tf.keras.models.load_model(str(self.model_path))

        if StaticPredictionService.label_map is None:
            with open(self.label_map_path, "r") as f:
                StaticPredictionService.label_map = json.load(f)

        self.hand_service = HandLandmarkService()

    def predict_from_image(self, image_file):
        hand_data = self.hand_service.extract_landmarks_with_quality_from_image(image_file)

        if not hand_data["success"]:
            return {
                "success": False,
                "error": hand_data["reason"],
                "gesture": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "second_confidence": 0.0,
                "margin": 0.0,
                "margin_percent": 0.0,
                "landmark_count": 0,
                "top_predictions": [],
                "hand_quality": hand_data
            }

        landmarks = hand_data["landmarks"]
        normalized = normalize_landmarks(landmarks)
        x = np.array([normalized], dtype=np.float32)

        predictions = StaticPredictionService.model.predict(x, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1]
        top1 = int(top_indices[0])
        top2 = int(top_indices[1]) if len(top_indices) > 1 else top1

        confidence = float(predictions[top1])
        second_confidence = float(predictions[top2])
        margin = confidence - second_confidence

        gesture = StaticPredictionService.label_map.get(str(top1), "Unknown")

        top_predictions = []

        for idx in top_indices[:3]:
            idx = int(idx)
            top_predictions.append({
                "gesture": StaticPredictionService.label_map.get(str(idx), "Unknown"),
                "confidence": round(float(predictions[idx]), 4),
                "confidence_percent": round(float(predictions[idx]) * 100, 2)
            })

        return {
            "success": True,
            "error": None,
            "gesture": gesture,
            "confidence": confidence,
            "confidence_percent": confidence * 100,
            "second_confidence": second_confidence,
            "margin": margin,
            "margin_percent": margin * 100,
            "landmark_count": len(landmarks),
            "top_predictions": top_predictions,
            "hand_quality": hand_data
        }

    def _extract_quality_from_image_file(self, image_file):
        import cv2
        import numpy as np

        file_bytes = image_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return self.hand_service.extract_landmarks_with_quality(image)