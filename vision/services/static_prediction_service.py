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
        landmarks = self.hand_service.extract_landmarks_from_image(image_file)

        if landmarks is None:
            return {
                "success": False,
                "error": "No hand landmarks detected",
                "gesture": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "landmark_count": 0,
                "top_predictions": []
            }

        normalized = normalize_landmarks(landmarks)
        x = np.array([normalized], dtype=np.float32)

        predictions = StaticPredictionService.model.predict(x, verbose=0)[0]

        class_index = int(np.argmax(predictions))
        confidence = float(predictions[class_index])
        confidence_percent = confidence * 100

        gesture = StaticPredictionService.label_map.get(str(class_index), "Unknown")

        top_indices = np.argsort(predictions)[::-1][:3]

        top_predictions = []

        for idx in top_indices:
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
            "confidence_percent": confidence_percent,
            "landmark_count": len(landmarks),
            "top_predictions": top_predictions
        }