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
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
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

    def extract_landmarks_with_quality(self, frame):
        if frame is None:
            return {
                "success": False,
                "reason": "empty frame",
                "landmarks": None
            }

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_image)

        if not result.multi_hand_landmarks:
            return {
                "success": False,
                "reason": "no hand detected",
                "landmarks": None
            }

        hand_landmarks = result.multi_hand_landmarks[0]

        handedness_score = 0.0

        if result.multi_handedness:
            handedness_score = float(result.multi_handedness[0].classification[0].score)

        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        box_width = max_x - min_x
        box_height = max_y - min_y
        box_area = box_width * box_height

        if handedness_score < 0.85:
            return {
                "success": False,
                "reason": f"weak handedness score: {handedness_score:.2f}",
                "landmarks": None
            }

        if box_area < 0.015:
            return {
                "success": False,
                "reason": f"hand too small or false detection: {box_area:.4f}",
                "landmarks": None
            }

        if box_area > 0.80:
            return {
                "success": False,
                "reason": f"invalid hand area: {box_area:.4f}",
                "landmarks": None
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
            "handedness_score": handedness_score,
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

    def extract_landmarks_with_quality_from_image(self, image_file):
        file_bytes = image_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "success": False,
                "reason": "invalid image",
                "landmarks": None
            }

        return self.extract_landmarks_with_quality(image)