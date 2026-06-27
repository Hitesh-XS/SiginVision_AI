import os
from pathlib import Path

from django.core.management.base import BaseCommand

from vision.models import Gesture, DatasetSample
from vision.services.hand_landmark_service import HandLandmarkService


class Command(BaseCommand):
    help = "Import ASL image dataset by extracting MediaPipe hand landmarks"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Path to dataset folder containing class folders like A, B, C"
        )

        parser.add_argument(
            "--classes",
            nargs="+",
            required=True,
            help="Classes to import, example: A B C L Y"
        )

        parser.add_argument(
            "--limit",
            type=int,
            default=300,
            help="Maximum images per class"
        )

        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete old samples for selected classes before importing"
        )

    def handle(self, *args, **options):
        dataset_path = Path(options["dataset"])
        selected_classes = options["classes"]
        limit = options["limit"]
        clear = options["clear"]

        if not dataset_path.exists():
            self.stderr.write(self.style.ERROR(f"Dataset path not found: {dataset_path}"))
            return

        service = HandLandmarkService()

        total_saved = 0
        total_failed = 0

        for class_name in selected_classes:
            class_folder = dataset_path / class_name

            if not class_folder.exists():
                self.stderr.write(self.style.WARNING(f"Class folder not found: {class_folder}"))
                continue

            gesture, _ = Gesture.objects.get_or_create(
                name=class_name,
                defaults={"gesture_type": "static"}
            )

            if clear:
                deleted_count, _ = DatasetSample.objects.filter(gesture=gesture).delete()
                self.stdout.write(self.style.WARNING(
                    f"Deleted {deleted_count} old samples for {class_name}"
                ))

            image_files = []

            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_files.append(class_folder / file_name)

            image_files = image_files[:limit]

            self.stdout.write(self.style.NOTICE(
                f"Importing {len(image_files)} images for class {class_name}"
            ))

            saved = 0
            failed = 0

            for image_path in image_files:
                try:
                    with open(image_path, "rb") as image_file:
                        landmarks = service.extract_landmarks_from_image(image_file)

                    if landmarks is None:
                        failed += 1
                        continue

                    DatasetSample.objects.create(
                        gesture=gesture,
                        landmark=landmarks
                    )

                    saved += 1

                except Exception as e:
                    failed += 1
                    self.stderr.write(f"Failed: {image_path} | {e}")

            total_saved += saved
            total_failed += failed

            self.stdout.write(self.style.SUCCESS(
                f"{class_name}: saved={saved}, failed={failed}"
            ))

        self.stdout.write(self.style.SUCCESS(
            f"Import complete. Total saved={total_saved}, total failed={total_failed}"
        ))