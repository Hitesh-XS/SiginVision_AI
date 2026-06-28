import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from django.conf import settings

from vision.services.hand_landmark_service import HandLandmarkService
from vision.services.landmark_utils import normalize_landmark_sequence


class DynamicPredictionService:
    model = None
    label_map = None

    def __init__(self):
        self.model_path = Path(settings.BASE_DIR) / "ml_models" / "dynamic_gesture_model.keras"
        self.label_map_path = Path(settings.BASE_DIR) / "ml_models" / "dynamic_label_map.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Dynamic model file not found: {self.model_path}")

        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Dynamic label map file not found: {self.label_map_path}")

        if DynamicPredictionService.model is None:
            DynamicPredictionService.model = tf.keras.models.load_model(str(self.model_path))

        if DynamicPredictionService.label_map is None:
            with open(self.label_map_path, "r") as f:
                DynamicPredictionService.label_map = json.load(f)

        self.hand_service = HandLandmarkService()

    def _resample_sequence(self, sequence, target_length):
        if len(sequence) == target_length:
            return sequence

        indices = np.linspace(0, len(sequence) - 1, target_length).astype(int)
        return [sequence[i] for i in indices]

    def predict_from_image_files(self, image_files, frame_count=30):
        sequence = []

        for image_file in image_files:
            landmarks = self.hand_service.extract_landmarks_from_image(image_file)

            if landmarks is not None and len(landmarks) == 63:
                sequence.append(landmarks)

        if len(sequence) < 15:
            return {
                "success": False,
                "error": f"Not enough valid hand frames. Got {len(sequence)}, need at least 15.",
                "gesture": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "valid_frames": len(sequence),
                "top_predictions": []
            }

        sequence = self._resample_sequence(sequence, frame_count)
        normalized_sequence = normalize_landmark_sequence(sequence)

        x = np.array([normalized_sequence], dtype=np.float32)

        predictions = DynamicPredictionService.model.predict(x, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1]

        top1 = int(top_indices[0])
        top2 = int(top_indices[1]) if len(top_indices) > 1 else top1

        confidence = float(predictions[top1])
        second_confidence = float(predictions[top2])
        margin = confidence - second_confidence

        gesture = DynamicPredictionService.label_map.get(str(top1), "Unknown")

        top_predictions = []

        for idx in top_indices[:3]:
            idx = int(idx)
            top_predictions.append({
                "gesture": DynamicPredictionService.label_map.get(str(idx), "Unknown"),
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
            "valid_frames": len(sequence),
            "top_predictions": top_predictions
        }