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
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    def extract_landmarks_from_image(self, image_file):
        file_bytes = image_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_image)

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []

        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return landmarks