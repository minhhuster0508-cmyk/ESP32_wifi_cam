import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if MODEL_PATH.exists():
        return True

    print("Downloading hand_landmarker.task...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        print("Cannot download model:", error)
        print("Download manually and put it here:")
        print(MODEL_URL)
        print(MODEL_PATH)
        return False

    return True


def is_thumb_up(landmarks, handedness):
    if handedness == "Right":
        return landmarks[4].x < landmarks[3].x

    return landmarks[4].x > landmarks[3].x


def count_fingers(landmarks, handedness):
    count = 0

    if is_thumb_up(landmarks, handedness):
        count += 1

    finger_pairs = [
        (8, 6),    # index
        (12, 10),  # middle
        (16, 14),  # ring
        (20, 18),  # pinky
    ]

    for tip_id, pip_id in finger_pairs:
        if landmarks[tip_id].y < landmarks[pip_id].y:
            count += 1

    return count


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def draw_text(frame, finger_count, handedness):
    cv2.rectangle(frame, (20, 20), (285, 120), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
    )

    if handedness:
        cv2.putText(
            frame,
            handedness,
            (35, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )


def main():
    if not ensure_model():
        return

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                print("Cannot read camera frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000) + frame_index
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            finger_count = 0
            handedness = ""

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                if result.handedness and result.handedness[0]:
                    handedness = result.handedness[0][0].category_name

                finger_count = count_fingers(landmarks, handedness)
                draw_landmarks(frame, landmarks)

            draw_text(frame, finger_count, handedness)
            cv2.imshow("Finger Counter", frame)

            frame_index += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
