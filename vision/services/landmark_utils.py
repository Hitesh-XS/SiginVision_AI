import math


def normalize_landmarks(landmarks):
    points = []

    for i in range(0, len(landmarks), 3):
        points.append([
            float(landmarks[i]),
            float(landmarks[i + 1]),
            float(landmarks[i + 2])
        ])

    base_x, base_y, base_z = points[0]

    normalized = []

    for x, y, z in points:
        normalized.extend([
            x - base_x,
            y - base_y,
            z - base_z
        ])

    max_value = max(abs(v) for v in normalized)

    if max_value == 0:
        return normalized

    normalized = [v / max_value for v in normalized]

    return normalized