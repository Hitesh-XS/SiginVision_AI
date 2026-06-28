import time

import cv2
from django.core.management.base import BaseCommand

from vision.models import Gesture, DatasetSample
from vision.services.hand_landmark_service import HandLandmarkService


class Command(BaseCommand):
    help = "Collect custom gesture samples from webcam and save MediaPipe landmarks into DB"

    def add_arguments(self, parser):
        parser.add_argument(
            "--gesture",
            type=str,
            required=True,
            help="Gesture name, example: A, B, HELLO, STOP"
        )

        parser.add_argument(
            "--samples",
            type=int,
            default=300,
            help="Number of valid landmark samples to save"
        )

        parser.add_argument(
            "--duration",
            type=int,
            default=20,
            help="Collection duration in seconds"
        )

        parser.add_argument(
            "--camera",
            type=int,
            default=0,
            help="Camera index"
        )

        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete old samples for this gesture before collecting"
        )

    def handle(self, *args, **options):
        gesture_name = options["gesture"].strip()
        target_samples = options["samples"]
        duration = options["duration"]
        camera_index = options["camera"]
        clear = options["clear"]

        if target_samples <= 0:
            self.stderr.write(self.style.ERROR("samples must be positive"))
            return

        if duration <= 0:
            self.stderr.write(self.style.ERROR("duration must be positive"))
            return

        gesture, _ = Gesture.objects.get_or_create(
            name=gesture_name,
            defaults={"gesture_type": "static"}
        )

        if clear:
            deleted_count, _ = DatasetSample.objects.filter(gesture=gesture).delete()
            self.stdout.write(self.style.WARNING(
                f"Deleted {deleted_count} old samples for {gesture_name}"
            ))

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            self.stderr.write(self.style.ERROR("Could not open camera"))
            return

        service = HandLandmarkService()

        delay = duration / target_samples
        saved = 0
        failed = 0

        self.stdout.write(self.style.SUCCESS(
            f"Collecting gesture: {gesture_name}"
        ))
        self.stdout.write(
            f"Target samples: {target_samples}, duration: {duration}s, delay: {delay:.3f}s"
        )
        self.stdout.write("Press Q to stop early.")

        for i in range(3, 0, -1):
            self.stdout.write(f"Starting in {i}...")
            time.sleep(1)

        last_capture_time = 0

        while saved < target_samples:
            ret, frame = cap.read()

            if not ret:
                failed += 1
                continue

            frame = cv2.flip(frame, 1)

            display_frame = frame.copy()
            display_frame = service.draw_landmarks(display_frame)

            cv2.putText(
                display_frame,
                f"Gesture: {gesture_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Saved: {saved}/{target_samples} | Failed: {failed}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Move hand slowly: left/right/near/far/rotate",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("SignVision AI - Camera Collector", display_frame)

            current_time = time.time()

            if current_time - last_capture_time >= delay:
                landmarks = service.extract_landmarks_from_frame(frame)

                if landmarks is not None and len(landmarks) == 63:
                    DatasetSample.objects.create(
                        gesture=gesture,
                        landmark=landmarks
                    )
                    saved += 1
                else:
                    failed += 1

                last_capture_time = current_time

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.stdout.write(self.style.SUCCESS(
            f"Collection finished for {gesture_name}"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"Saved={saved}, failed={failed}"
        ))