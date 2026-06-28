import time

import cv2
from django.core.management.base import BaseCommand

from vision.models import Gesture, DynamicDatasetSample
from vision.services.hand_landmark_service import HandLandmarkService


class Command(BaseCommand):
    help = "Collect dynamic sign samples from webcam as landmark sequences"

    def add_arguments(self, parser):
        parser.add_argument(
            "--gesture",
            type=str,
            required=True,
            help="Dynamic gesture name, example: HELLO, THANK_YOU"
        )

        parser.add_argument(
            "--samples",
            type=int,
            default=50,
            help="Number of sequence samples to collect"
        )

        parser.add_argument(
            "--frames",
            type=int,
            default=30,
            help="Number of valid frames per sample"
        )

        parser.add_argument(
            "--seconds",
            type=float,
            default=2.0,
            help="Recording duration per sample"
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
            help="Delete old dynamic samples for this gesture before collecting"
        )

    def handle(self, *args, **options):
        gesture_name = options["gesture"].strip()
        total_samples = options["samples"]
        sequence_length = options["frames"]
        seconds_per_sample = options["seconds"]
        camera_index = options["camera"]
        clear = options["clear"]

        gesture, _ = Gesture.objects.get_or_create(
            name=gesture_name,
            defaults={"gesture_type": "dynamic"}
        )

        if gesture.gesture_type != "dynamic":
            gesture.gesture_type = "dynamic"
            gesture.save()

        if clear:
            deleted_count, _ = DynamicDatasetSample.objects.filter(gesture=gesture).delete()
            self.stdout.write(self.style.WARNING(
                f"Deleted {deleted_count} old dynamic samples for {gesture_name}"
            ))

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            self.stderr.write(self.style.ERROR("Could not open camera"))
            return

        service = HandLandmarkService()

        saved_samples = 0
        failed_samples = 0

        self.stdout.write(self.style.SUCCESS(
            f"Collecting dynamic gesture: {gesture_name}"
        ))
        self.stdout.write(
            f"Samples={total_samples}, frames/sample={sequence_length}, seconds/sample={seconds_per_sample}"
        )
        self.stdout.write("Press Q to quit.")

        for sample_no in range(1, total_samples + 1):
            self.stdout.write(self.style.NOTICE(
                f"Prepare sample {sample_no}/{total_samples}"
            ))

            for count in range(3, 0, -1):
                ret, frame = cap.read()

                if ret:
                    frame = cv2.flip(frame, 1)
                    cv2.putText(
                        frame,
                        f"Get ready: {count}",
                        (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        3
                    )
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture_name}",
                        (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow("SignVision AI - Dynamic Collector", frame)
                    cv2.waitKey(1)

                time.sleep(1)

            sequence = []
            start_time = time.time()
            interval = seconds_per_sample / sequence_length
            next_capture_time = start_time

            while len(sequence) < sequence_length:
                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                display_frame = service.draw_landmarks(display_frame)

                elapsed = time.time() - start_time

                cv2.putText(
                    display_frame,
                    f"Recording {sample_no}/{total_samples}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Frames: {len(sequence)}/{sequence_length}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Time: {elapsed:.1f}s",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv2.imshow("SignVision AI - Dynamic Collector", display_frame)

                current_time = time.time()

                if current_time >= next_capture_time:
                    landmarks = service.extract_landmarks_from_frame(frame)

                    if landmarks is not None and len(landmarks) == 63:
                        sequence.append(landmarks)

                    next_capture_time += interval

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    self.stdout.write(self.style.WARNING("Stopped by user"))
                    return

                if time.time() - start_time > seconds_per_sample + 3:
                    break

            if len(sequence) == sequence_length:
                DynamicDatasetSample.objects.create(
                    gesture=gesture,
                    frames=sequence,
                    frame_count=sequence_length
                )

                saved_samples += 1
                self.stdout.write(self.style.SUCCESS(
                    f"Saved dynamic sample {saved_samples}/{total_samples}"
                ))
            else:
                failed_samples += 1
                self.stderr.write(self.style.WARNING(
                    f"Failed sample {sample_no}: only {len(sequence)} valid frames"
                ))

        cap.release()
        cv2.destroyAllWindows()

        self.stdout.write(self.style.SUCCESS(
            f"Finished. saved={saved_samples}, failed={failed_samples}"
        ))