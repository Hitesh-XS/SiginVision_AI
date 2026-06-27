import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

from vision.models import DatasetSample
from vision.services.landmark_utils import normalize_landmarks


class Command(BaseCommand):
    help = "Train static gesture recognition model using stored hand landmarks"

    def add_arguments(self, parser):
        parser.add_argument(
            "--classes",
            nargs="+",
            required=False,
            help="Classes to train, example: A B C L Y"
        )

        parser.add_argument(
            "--output",
            type=str,
            default="ml_models",
            help="Folder to save model and label map"
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=50
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=32
        )

    def handle(self, *args, **options):
        selected_classes = options["classes"]
        output_dir = Path(options["output"])
        epochs = options["epochs"]
        batch_size = options["batch_size"]

        output_dir.mkdir(parents=True, exist_ok=True)

        queryset = DatasetSample.objects.select_related("gesture").all()

        if selected_classes:
            queryset = queryset.filter(gesture__name__in=selected_classes)

        X = []
        y = []

        for sample in queryset:
            if not sample.landmark:
                continue

            if len(sample.landmark) != 63:
                continue

            normalized = normalize_landmarks(sample.landmark)

            X.append(normalized)
            y.append(sample.gesture.name)

        if len(X) == 0:
            self.stderr.write(self.style.ERROR("No valid samples found."))
            return

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        unique_classes = sorted(list(set(y)))

        self.stdout.write(self.style.NOTICE(f"Total samples: {len(X)}"))
        self.stdout.write(self.style.NOTICE(f"Classes: {unique_classes}"))

        for class_name in unique_classes:
            count = int(np.sum(y == class_name))
            self.stdout.write(f"{class_name}: {count} samples")

        if len(unique_classes) < 2:
            self.stderr.write(self.style.ERROR("Need at least 2 classes to train."))
            return

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        num_classes = len(label_encoder.classes_)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(63,)),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation="relu"),

            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=0.00001
        )

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        self.stdout.write(self.style.SUCCESS(f"Test Accuracy: {accuracy:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"Test F1 Score: {f1:.4f}"))

        model_path = output_dir / "static_gesture_model.keras"
        label_map_path = output_dir / "static_label_map.json"
        report_path = output_dir / "static_training_report.json"

        model.save(model_path)

        label_map = {
            str(index): class_name
            for index, class_name in enumerate(label_encoder.classes_)
        }

        with open(label_map_path, "w") as f:
            json.dump(label_map, f, indent=4)

        training_report = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "classes": list(label_encoder.classes_),
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "classification_report": report
        }

        with open(report_path, "w") as f:
            json.dump(training_report, f, indent=4)

        self.stdout.write(self.style.SUCCESS(f"Model saved to: {model_path}"))
        self.stdout.write(self.style.SUCCESS(f"Label map saved to: {label_map_path}"))
        self.stdout.write(self.style.SUCCESS(f"Report saved to: {report_path}"))