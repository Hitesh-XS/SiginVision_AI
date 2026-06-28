import cv2
import mediapipe as mp
import numpy as np


class HandLandmarkService:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50
        )

    def extract_landmarks_from_image(self, image_file):
        file_bytes = image_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return None

        return self.extract_landmarks_from_frame(image)

    def extract_landmarks_from_frame(self, frame):
        data = self.extract_landmarks_with_quality(frame)

        if not data["success"]:
            return None

        return data["landmarks"]

    def extract_landmarks_with_quality_from_image(self, image_file):
        file_bytes = image_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "success": False,
                "reason": "invalid image",
                "landmarks": None,
                "box_area": 0.0
            }

        return self.extract_landmarks_with_quality(image)

    def extract_landmarks_with_quality(self, frame):
        if frame is None:
            return {
                "success": False,
                "reason": "empty frame",
                "landmarks": None,
                "box_area": 0.0
            }

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_image)

        if not result.multi_hand_landmarks:
            return {
                "success": False,
                "reason": "no hand detected",
                "landmarks": None,
                "box_area": 0.0
            }

        hand_landmarks = result.multi_hand_landmarks[0]

        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)
        box_area = box_width * box_height

        # Only reject extremely tiny fake detections
        if box_area < 0.003:
            return {
                "success": False,
                "reason": f"hand too small: {box_area:.4f}",
                "landmarks": None,
                "box_area": box_area
            }

        landmarks = []

        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            ])

        return {
            "success": True,
            "reason": "valid hand",
            "landmarks": landmarks,
            "box_area": box_area
        }

    def draw_landmarks(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame