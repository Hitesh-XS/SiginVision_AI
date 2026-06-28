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

def normalize_landmark_sequence(sequence):
    normalized_sequence = []

    if not sequence:
        return normalized_sequence

    first_frame = sequence[0]
    base_x = float(first_frame[0])
    base_y = float(first_frame[1])
    base_z = float(first_frame[2])

    all_values = []

    for frame in sequence:
        current_frame = []

        for i in range(0, len(frame), 3):
            x = float(frame[i]) - base_x
            y = float(frame[i + 1]) - base_y
            z = float(frame[i + 2]) - base_z

            current_frame.extend([x, y, z])
            all_values.extend([x, y, z])

        normalized_sequence.append(current_frame)

    max_value = max(abs(v) for v in all_values) if all_values else 0

    if max_value == 0:
        return normalized_sequence

    final_sequence = []

    for frame in normalized_sequence:
        final_sequence.append([v / max_value for v in frame])

    return final_sequence