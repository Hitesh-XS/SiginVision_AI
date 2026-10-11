# SignVision AI 🤟

### AI-Powered Sign Language Recognition Using Deep Learning

SignVision AI is a web-based sign language recognition application that uses computer vision and deep learning to recognize static hand gestures and dynamic sign language gestures. It converts recognized signs into text and provides tools to build sentences, listen to recognized text through text-to-speech, and review saved sentence history.

The application combines **Django, TensorFlow, MediaPipe, OpenCV, NumPy, and JavaScript** to provide an interactive browser-based experience.

The core idea is to extract hand landmarks instead of training directly on raw images. MediaPipe identifies 21 landmarks on a hand, each containing three coordinates (`x`, `y`, and `z`). These coordinates become numerical features for the machine learning models.

SignVision AI uses two separate recognition pipelines:

- **Static gesture recognition:** A feed-forward neural network classifies a single hand pose, such as a sign representing an alphabet.
- **Dynamic gesture recognition:** A Bidirectional LSTM processes a sequence of hand poses to recognize gestures that depend on movement over time.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [System Architecture](#-system-architecture)
5. [How Sign Recognition Works](#-how-sign-recognition-works)
6. [Static Gesture Recognition](#-static-gesture-recognition)
7. [Dynamic Gesture Recognition](#-dynamic-gesture-recognition)
8. [Hand Landmark Extraction](#-hand-landmark-extraction)
9. [Landmark Normalization](#-landmark-normalization)
10. [Dataset Preparation](#-dataset-preparation)
11. [Model Training](#-model-training)
12. [Project Structure](#-project-structure)
13. [Installation and Setup](#-installation-and-setup)
14. [Running the Application](#-running-the-application)
15. [Application Workflow](#-application-workflow)
16. [Prediction Output](#-prediction-output)
17. [Sentence Builder and Text-to-Speech](#-sentence-builder-and-text-to-speech)
18. [Database and Persistence](#-database-and-persistence)
19. [Model Evaluation](#-model-evaluation)
20. [Configuration](#-configuration)
21. [Troubleshooting](#-troubleshooting)
22. [Limitations](#-limitations)
23. [Future Improvements](#-future-improvements)
24. [Contributing](#-contributing)
25. [License](#-license)

---

## 📌 Project Overview

Sign language is a visual language that communicates meaning through hand shapes, movement, orientation, and other visual cues. Recognizing signs automatically requires a system that can understand both the shape of a hand and how that shape changes over time.

SignVision AI addresses this problem using landmark-based gesture recognition.

Instead of sending every pixel of an image directly to the classifier, the application extracts a compact representation of the hand.

For each detected hand:

- 21 landmarks are extracted.
- Each landmark contains `x`, `y`, and `z` coordinates.
- The resulting feature vector contains 63 numerical values.

For static gestures, one feature vector represents one sample. For dynamic gestures, multiple feature vectors are arranged chronologically to represent a movement.

### Recognition overview

| Component | Static recognition | Dynamic recognition |
|---|---|---|
| Input | One image or frame | Sequence of frames |
| Features per frame | 63 | 63 |
| Frames per model input | 1 | 30 |
| Input shape | `(63,)` per sample | `(30, 63)` per sample |
| Model | Dense neural network (MLP) | Bidirectional LSTM |
| Main objective | Recognize hand pose | Recognize motion over time |
| Output | Predicted gesture and confidence | Predicted gesture and confidence |

The dynamic model expects a sequence of 30 frames. Each frame contains 63 landmark values, giving 1,890 input values per complete sequence before considering the batch dimension.

---

## ✨ Key Features

### 1. Static gesture recognition

Recognizes hand poses from individual images or webcam frames. This pipeline is intended for static signs, such as alphabet gestures.

### 2. Dynamic gesture recognition

Recognizes movement-based gestures from a sequence of frames using a recurrent neural network.

### 3. MediaPipe hand landmark extraction

Uses MediaPipe to detect the hand and extract its 21 landmarks, creating a compact representation of hand geometry.

### 4. Landmark normalization

Shifts landmark coordinates relative to the wrist and scales the values to reduce the influence of hand position and scale.

### 5. Confidence-based predictions

Returns the most likely gesture, its confidence score, the second-best prediction, and the confidence difference between the two leading classes.

### 6. Browser-based camera interaction

Provides a web interface for camera-based gesture recognition, with support for switching between static and dynamic modes.

### 7. Sentence builder

Allows recognized gestures to be assembled into text.

### 8. Text-to-speech

Provides audio playback of recognized text using the application's text-to-speech functionality.

### 9. Saved sentence history

Allows sentences to be saved and retrieved for later review.

### 10. Hand-removal duplicate protection

Uses hand-presence or hand-removal locking in the live recognition workflow to help prevent a continuously visible static sign from being appended repeatedly.

---

## 🛠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Django | Backend, request handling, views, and application integration |
| TensorFlow / Keras | Model creation, training, saving, and inference |
| MediaPipe | Hand detection and landmark extraction |
| OpenCV | Image decoding, color conversion, and image processing |
| NumPy | Numerical operations and model input preparation |
| Scikit-learn | Dataset splitting, label encoding, and evaluation metrics |
| SQLite | Local application database |
| HTML and CSS | Web interface |
| JavaScript | Webcam interaction, prediction requests, and frontend behavior |
| JSON | Label maps and training reports |

---

## 🏗 System Architecture

The system is divided into five main layers.

### Layer 1: User interface

The browser provides the camera interface, recognition mode selection, prediction display, sentence builder, text-to-speech controls, and saved history.

### Layer 2: Django application

Django handles HTTP requests, routes requests to the appropriate views, invokes prediction services, and returns prediction results to the frontend.

### Layer 3: Computer vision

OpenCV decodes images and converts their color representation. MediaPipe detects the hand and extracts its landmarks.

### Layer 4: Machine learning

The extracted landmarks are normalized and passed to one of two TensorFlow models:

- Static Dense neural network
- Dynamic Bidirectional LSTM

### Layer 5: Persistence

Django's database layer stores application data, including saved sentence history. The trained model files and label maps are stored separately in the `ml_models` directory.

### Overall architecture

    Browser / Webcam
          |
          v
    Django Views / API
          |
          v
    Prediction Service
          |
          v
    OpenCV + MediaPipe
          |
          v
    21 Hand Landmarks
          |
          v
    63 Values per Frame
          |
          +-----------------------+
          |                       |
          v                       v
    Static Pipeline          Dynamic Pipeline
    One Frame                30 Frames
          |                       |
          v                       v
    Normalize                 Normalize Sequence
          |                       |
          v                       v
    Dense Neural Network     Bidirectional LSTM
          |                       |
          +-----------+-----------+
                      |
                      v
             Gesture Prediction
                      |
                      v
             Django JSON Response
                      |
                      v
             Frontend / Sentence
                 / Text-to-Speech

---

## 🧠 How Sign Recognition Works

The recognition pipeline consists of several stages.

### Step 1: Capture an image or frame

The user presents a hand gesture to the webcam or supplies an image.

For static recognition, one frame is sufficient.

For dynamic recognition, the application collects multiple frames in chronological order.

### Step 2: Decode the image

OpenCV converts the uploaded image bytes into an image array.

For example:

```python
file_bytes = image_file.read()
np_arr = np.frombuffer(file_bytes, np.uint8)
image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
```

If decoding fails, the image cannot be processed.

### Step 3: Detect the hand

MediaPipe processes the image and identifies hand landmarks.

The current service is configured to process one hand:

```python
max_num_hands=1
```

The service also checks whether a hand was detected and whether its estimated bounding box is too small.

### Step 4: Extract landmarks

Each detected hand provides 21 landmarks. Each landmark has three coordinates.

    21 landmarks × 3 coordinates = 63 values

The flattened representation looks like:

```python
[
    x0, y0, z0,
    x1, y1, z1,
    # ...
    x20, y20, z20
]
```

The landmark coordinates are then normalized.

### Step 5: Prepare model input

The static model receives one normalized feature vector.

The dynamic model receives a normalized sequence of 30 feature vectors.

### Step 6: Predict the gesture

The selected TensorFlow model returns a probability-like score for each output class.

The highest-scoring class is selected as the predicted gesture.

### Step 7: Map the output to a label

The predicted class index is translated into a human-readable gesture using the corresponding JSON label map.

For example:

```json
{
  "0": "A",
  "1": "B",
  "2": "C"
}
```

This is an illustrative mapping; the actual labels depend on the classes used during training.

### Step 8: Return the result

The backend returns the prediction, confidence information, and other relevant metadata to the frontend.

---

## ✋ Static Gesture Recognition

Static recognition is designed for gestures whose meaning can be identified from a single hand pose.

### Input representation

One static training sample contains:

```text
1 image
   |
   v
21 landmarks
   |
   v
63 numerical features
```

For a dataset containing 1,500 valid images, the feature matrix would have shape:

```python
X.shape == (1500, 63)
y.shape == (1500,)
```

Here:

- `1500` is the number of samples.
- `63` is the number of input features.
- `y` contains the class label for each sample.

### Model architecture

The static classifier is a feed-forward neural network, also called a Multi-Layer Perceptron (MLP).

The training architecture uses Dense layers, Batch Normalization, and Dropout.

    Input: 63 features
            |
            v
       Dense(128)
          ReLU
            |
            v
    Batch Normalization
            |
            v
       Dropout(0.3)
            |
            v
        Dense(64)
          ReLU
            |
            v
    Batch Normalization
            |
            v
       Dropout(0.2)
            |
            v
        Dense(32)
          ReLU
            |
            v
    Dense(number of classes)
          Softmax
            |
            v
    Predicted static gesture

### Explanation of the layers

**Dense layers:** Learn relationships between the 63 input features and the gesture classes.

**ReLU activation:** Adds non-linearity, allowing the network to learn more complex decision boundaries.

**Batch Normalization:** Helps stabilize the distribution of activations during training.

**Dropout:** Randomly disables some activations during training to help reduce overfitting.

**Softmax:** Converts the output logits into a normalized set of class scores.

### Static model files

The trained model and associated metadata are stored as:

```text
ml_models/
├── static_gesture_model.keras
├── static_label_map.json
└── static_training_report.json
```

The `.keras` file contains the trained model. The label map connects output indexes to gesture names, and the training report records evaluation information.

---

## 🎬 Dynamic Gesture Recognition

Dynamic recognition handles gestures whose meaning depends on movement over time.

A single frame may show a hand pose, but it cannot fully represent the motion used to produce a dynamic sign.

### Input representation

Each frame contains 63 landmark values.

The current dynamic model uses 30 frames per sequence.

    Frame 1  -> 63 features
    Frame 2  -> 63 features
    ...
    Frame 30 -> 63 features

    Total = 30 × 63 = 1,890 values

The model input shape for one batch containing one sample is:

```python
(1, 30, 63)
```

The dimensions represent:

- `1`: batch size
- `30`: time steps or frames
- `63`: features per frame

### Dynamic preprocessing

The dynamic prediction service follows these steps:

1. Extract landmarks from every supplied image.
2. Discard frames without a valid 63-value landmark vector.
3. Require at least 15 valid frames.
4. Resample the sequence to 30 frames.
5. Normalize the entire sequence.
6. Convert the sequence to a NumPy array.
7. Run inference using the trained dynamic model.

### Why resample the sequence?

Recordings can contain different numbers of valid frames.

The model expects a fixed sequence length, so the service uses evenly spaced indexes to select 30 frames.

For example:

```text
20 valid frames -> 30 model frames
30 valid frames -> 30 model frames
50 valid frames -> 30 model frames
```

This produces a consistent input shape. When the source sequence is shorter than the target length, the current implementation may repeat selected frames; it does not synthesize genuinely new motion.

### Dynamic model architecture

The dynamic classifier uses a stacked Bidirectional LSTM architecture.

    Input: (30, 63)
            |
            v
         Masking
            |
            v
    Bidirectional LSTM(96)
       return_sequences=True
            |
            v
         Dropout
            |
            v
    Bidirectional LSTM(64)
            |
            v
         Dense(64)
            |
            v
         Dense(32)
            |
            v
    Dense(number of classes)
          Softmax
            |
            v
    Predicted dynamic gesture

### Why use an LSTM?

An LSTM (Long Short-Term Memory network) is a recurrent neural network designed to learn patterns in sequential data.

For a dynamic gesture, the model needs to learn how landmark positions change from one frame to another.

For example, two recordings may contain similar hand poses but opposite movement directions. A model that only sees individual frames may struggle to distinguish them.

The LSTM processes the sequence and learns temporal relationships between frames.

### Why Bidirectional LSTM?

A Bidirectional LSTM processes the supplied sequence in both directions:

```text
Forward:  Frame 1  -> Frame 30
Backward: Frame 30 -> Frame 1
```

This gives the model information from both earlier and later frames in the recorded sequence.

It is useful for offline classification, where the complete gesture sequence is available before prediction. It is not inherently a zero-latency streaming model because future frames are needed by the backward direction.

### Dynamic model files

```text
ml_models/
├── dynamic_gesture_model.keras
└── dynamic_label_map.json
```

The label map must correspond to the exact class ordering used to train the dynamic model.

---

## 📍 Hand Landmark Extraction

The `HandLandmarkService` centralizes hand detection and landmark extraction.

Its responsibilities include:

- Decoding uploaded images
- Converting BGR images to RGB
- Running MediaPipe hand detection
- Checking whether a hand was detected
- Estimating a bounding-box area
- Rejecting extremely small detections
- Extracting and flattening the 21 landmarks
- Returning quality information
- Drawing landmarks for visualization

### MediaPipe configuration

The service uses the following settings:

```python
self.hands = self.mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50
)
```

**`static_image_mode=True`:** Treats each frame as an independent image and runs hand detection on each one. This is suitable for individual images but may be less efficient for continuous video.

**`max_num_hands=1`:** Limits detection to one hand.

**`model_complexity=1`:** Selects the configured model complexity.

**`min_detection_confidence=0.50`:** Sets the minimum detection confidence.

**`min_tracking_confidence=0.50`:** Configures the tracking confidence threshold. Tracking is less relevant when static-image mode is enabled.

### Bounding-box quality check

The service calculates the width and height of the detected landmark bounding box:

```python
box_width = max(xs) - min(xs)
box_height = max(ys) - min(ys)

box_area = box_width * box_height
```

It rejects detections where:

```python
box_area < 0.003
```

This threshold is an application-specific heuristic for filtering extremely small detections, not a universal guarantee of landmark quality.

### Landmark output

A successful extraction returns data similar to:

```python
{
    "success": True,
    "reason": "valid hand",
    "landmarks": [...],  # 63 values
    "box_area": 0.12
}
```

The numerical `box_area` above is an illustrative value.

A failed extraction may return a reason such as:

- `invalid image`
- `empty frame`
- `no hand detected`
- `hand too small`

These results let the prediction service handle invalid input without attempting model inference.

---

## 📐 Landmark Normalization

Normalization transforms the raw landmark coordinates into a more consistent representation before model inference.

The project provides two functions:

```python
normalize_landmarks(landmarks)
normalize_landmark_sequence(sequence)
```

### Static normalization

Static normalization operates on one 63-value feature vector.

Its steps are:

1. Group the 63 values into 21 `(x, y, z)` points.
2. Use the wrist landmark (index 0) as the origin.
3. Subtract the wrist coordinates from every landmark.
4. Find the largest absolute coordinate after subtraction.
5. Divide every coordinate by that maximum value.

Mathematically, for landmark \(i\):

```text
x'i = xi - xw
y'i = yi - yw
z'i = zi - zw
```

where the subscript `w` denotes the wrist.

Let \(M\) be the maximum absolute value across all shifted coordinates. The final values are:

```text
normalized_value = shifted_value / M
```

If every shifted coordinate is zero, the function returns the zero-valued vector without dividing by zero.

This reduces sensitivity to the hand's absolute position and provides a common numerical scale. It does not guarantee invariance to every change in camera perspective, hand orientation, or physical size.

### Dynamic sequence normalization

Dynamic normalization operates on a sequence of frames.

Its steps are:

1. Select the wrist coordinates from the first frame.
2. Subtract those same coordinates from every landmark in every frame.
3. Collect the shifted coordinates from the complete sequence.
4. Calculate one maximum absolute value across all frames.
5. Divide every coordinate in every frame by that same maximum.

The important distinction is that **one reference wrist and one scale factor are used for the entire sequence**.

This prevents each frame from being scaled independently, which could otherwise remove some information about relative changes in hand configuration.

The function returns the same sequence dimensions:

```text
Before normalization: (30, 63)
After normalization:  (30, 63)
```

Normalization reduces absolute-position effects but can also remove some global movement information. If whole-hand translation is important for a particular sign, the preprocessing strategy should be evaluated to ensure it preserves the required information.

---

## 🗂 Dataset Preparation

SignVision AI uses a landmark-based dataset pipeline.

Rather than repeatedly running hand detection during every training epoch, hand landmarks can be extracted from source images and stored as processed training samples.

### Static dataset

A static dataset sample consists of:

- A gesture class
- One set of 63 landmark values

Conceptually:

```text
DatasetSample
├── gesture label
└── landmarks (63 values)
```

The static dataset import command uses `HandLandmarkService` to detect the hand and extract landmarks before saving valid samples to the database.

The import process can be configured to select classes, limit the number of images, and clear previous samples.

An illustrative source dataset layout is:

```text
dataset/
├── A/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── B/
│   ├── image_001.jpg
│   └── image_002.jpg
└── C/
    └── ...
```

Each subdirectory represents a gesture class. The exact supported classes depend on the dataset and the training configuration.

### Dynamic dataset

A dynamic dataset sample represents one complete gesture recording.

Conceptually:

```text
DynamicDatasetSample
├── gesture label
└── sequence of frames
    ├── frame 1: 63 values
    ├── frame 2: 63 values
    └── ...
        frame 30: 63 values
```

The dynamic training pipeline validates the number of frames and the number of features per frame before constructing the model input.

### Data quality recommendations

For reliable evaluation:

- Capture samples under different lighting conditions.
- Vary hand positions and distances from the camera.
- Include different valid executions of each gesture.
- Avoid mislabeled or incomplete recordings.
- Inspect failed landmark detections.
- Keep the training and evaluation data separate.
- Avoid allowing near-identical frames from the same recording to leak into both training and test sets.

---

## 🏋️ Model Training

The project uses separate training pipelines for static and dynamic gestures.

### Static training pipeline

```text
Stored DatasetSample records
          |
          v
Filter selected gesture classes
          |
          v
Validate 63-value landmarks
          |
          v
Normalize each sample
          |
          v
Construct X and y
          |
          v
Encode class labels
          |
          v
Train/test split
          |
          v
Train Dense neural network
          |
          v
Evaluate predictions
          |
          v
Save model and metadata
```

The feature matrix has shape:

```python
X.shape == (number_of_samples, 63)
```

The label vector has shape:

```python
y.shape == (number_of_samples,)
```

The training implementation uses label encoding, a train/test split, and neural-network training callbacks.

Relevant techniques include:

**LabelEncoder:** Converts class labels into integer class IDs.

**Stratified splitting:** Helps maintain class proportions across training and testing partitions.

**EarlyStopping:** Stops training when the monitored validation metric stops improving according to its configuration.

**ReduceLROnPlateau:** Reduces the learning rate when the monitored validation metric plateaus.

**Accuracy:** Measures the fraction of correct predictions.

**Weighted F1 score:** Combines precision and recall while weighting classes by their support.

### Dynamic training pipeline

```text
Stored DynamicDatasetSample records
             |
             v
Validate sequence length and features
             |
             v
Normalize each sequence
             |
             v
Construct X and y
             |
             v
Encode gesture labels
             |
             v
Train/test split
             |
             v
Train Bidirectional LSTM
             |
             v
Evaluate predictions
             |
             v
Save model and label map
```

The dynamic feature matrix has shape:

```python
X.shape == (number_of_samples, 30, 63)
```

The dynamic model learns from temporal sequences rather than isolated hand poses.

### Saved training artifacts

The static training pipeline saves:

```text
ml_models/
├── static_gesture_model.keras
├── static_label_map.json
└── static_training_report.json
```

The dynamic pipeline saves:

```text
ml_models/
├── dynamic_gesture_model.keras
└── dynamic_label_map.json
```

The models and their label maps must remain synchronized. Changing class ordering without retraining or updating the corresponding mapping can cause the application to display incorrect gesture names.

---

## 📁 Project Structure

The following is a logical representation of the main project components. Some supporting directories and filenames may differ in the local checkout.

```text
SignVision_AI/
│
├── manage.py
│
├── vision/
│   ├── management/
│   │   └── commands/
│   │       ├── static dataset import command
│   │       ├── static model training command
│   │       └── dynamic model training command
│   │
│   ├── services/
│   │   ├── hand_landmark_service.py
│   │   ├── landmark_utils.py
│   │   ├── static_prediction_service.py
│   │   └── dynamic_prediction_service.py
│   │
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── ...
│
├── ml_models/
│   ├── static_gesture_model.keras
│   ├── static_label_map.json
│   ├── static_training_report.json
│   ├── dynamic_gesture_model.keras
│   └── dynamic_label_map.json
│
├── templates/
│   └── ...
│
├── static/
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Important files

| File | Responsibility |
|---|---|
| `hand_landmark_service.py` | Hand detection, landmark extraction, and detection-quality checks |
| `landmark_utils.py` | Static and dynamic landmark normalization |
| `static_prediction_service.py` | Single-image gesture inference |
| `dynamic_prediction_service.py` | Multi-frame gesture inference and sequence resampling |
| `models.py` | Database models for stored samples and application data |
| `views.py` | Django request handling and application endpoints |
| `urls.py` | URL routing |
| `static_gesture_model.keras` | Trained static classifier |
| `dynamic_gesture_model.keras` | Trained dynamic classifier |
| `static_label_map.json` | Static class-index mapping |
| `dynamic_label_map.json` | Dynamic class-index mapping |

---

## 🚀 Installation and Setup

### Prerequisites

Install the following before setting up the application:

- Python compatible with the project's installed Django, TensorFlow, and MediaPipe versions
- Git
- A supported operating system
- A webcam for live recognition
- A browser that supports webcam access

A compatible Python environment is important because TensorFlow, MediaPipe, and OpenCV may have platform- and Python-version-specific requirements.

### Step 1: Clone the repository

```bash
git clone https://github.com/Hitesh-XS/SiginVision_AI.git
cd SiginVision_AI
```

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of its execution policy, follow the Python environment activation instructions appropriate for your system. Alternatively, use the virtual environment's Python executable directly.

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

If the repository includes a complete `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency file should include the project's required versions of Django, TensorFlow, MediaPipe, OpenCV, NumPy, and Scikit-learn, along with other dependencies used by the application.

If installation fails, check the Python version and package compatibility before changing individual dependency versions.

### Step 4: Configure environment variables

If the project uses environment variables for Django configuration, create a local environment file or set the variables in your shell according to the project's settings.

Typical settings include:

```text
SECRET_KEY
DEBUG
ALLOWED_HOSTS
DATABASE_URL
```

These are examples of common Django settings, not a claim that every variable is already implemented in the repository. Use the actual variable names supported by the project's settings module.

Do not commit real secret keys or production credentials.

### Step 5: Apply database migrations

```bash
python manage.py migrate
```

This creates or updates the local database schema for Django's registered applications.

### Step 6: Create an administrator (optional)

```bash
python manage.py createsuperuser
```

Follow the prompts if you want to manage supported database records through Django's admin interface.

### Step 7: Check the Django configuration

```bash
python manage.py check
```

This runs Django's system checks and reports configuration problems that can be detected before starting the application.

### Step 8: Verify the model files

Confirm that the model files required for inference exist:

```text
ml_models/
├── static_gesture_model.keras
├── static_label_map.json
├── dynamic_gesture_model.keras
└── dynamic_label_map.json
```

The static prediction service raises a `FileNotFoundError` if its model or label map is missing. The dynamic prediction service performs the same checks for its corresponding artifacts.

If these files are not included in the repository, train the models locally or obtain the project-approved artifacts before testing prediction endpoints.

---

## ▶️ Running the Application

Start the Django development server:

```bash
python manage.py runserver
```

By default, Django serves the application at:

```text
http://127.0.0.1:8000/
```

Open the application's configured route in a browser.

For a local development environment, grant camera permission when prompted. Webcam access typically requires a secure context, such as `localhost` or HTTPS.

The exact landing page and API routes depend on the project's current `urls.py` configuration.

### Verify the main components

1. Confirm that Django starts without configuration errors.
2. Open the application's interface.
3. Allow camera access.
4. Test static recognition with a supported hand pose.
5. Test dynamic recognition with a supported movement.
6. Check the confidence values and returned gesture.
7. Test sentence building, text-to-speech, and saved history.

---

## 🔄 Application Workflow

### Static prediction workflow

The browser captures an image or frame and sends it to the static prediction endpoint.

The backend then:

1. Decodes the image.
2. Extracts landmarks and quality information.
3. Rejects invalid or missing hand detections.
4. Normalizes the landmarks.
5. Creates a model input with shape `(1, 63)`.
6. Runs the static TensorFlow model.
7. Selects the highest-scoring class.
8. Maps the class index to a gesture label.
9. Returns the prediction result.

The frontend displays the recognized gesture and confidence.

### Dynamic prediction workflow

The browser records or collects a sequence of frames.

The backend then:

1. Extracts landmarks from each supplied frame.
2. Keeps valid 63-value frames.
3. Requires at least 15 valid frames.
4. Resamples the sequence to the configured frame count, normally 30.
5. Normalizes the sequence.
6. Creates a model input with shape `(1, 30, 63)`.
7. Runs the dynamic TensorFlow model.
8. Returns the predicted gesture and confidence information.

The dynamic prediction service itself accepts image files. Video recording and frame collection are handled by the calling application workflow.

### Live static recognition

The existing application workflow has used periodic requests to:

```text
/api/vision/predict-static/
```

with approximately 700 ms between requests.

The exact polling interval and route should be checked against the current frontend implementation.

Dynamic recognition uses a recording window rather than treating every isolated webcam frame as a complete gesture.

---

## 📊 Prediction Output

Both prediction services return structured dictionaries that can be serialized as JSON by Django.

### Successful static prediction

An illustrative result:

```json
{
  "success": true,
  "error": null,
  "gesture": "A",
  "confidence": 0.92,
  "confidence_percent": 92.0,
  "second_confidence": 0.05,
  "margin": 0.87,
  "margin_percent": 87.0,
  "landmark_count": 63,
  "top_predictions": [
    {
      "gesture": "A",
      "confidence": 0.92,
      "confidence_percent": 92.0
    },
    {
      "gesture": "B",
      "confidence": 0.05,
      "confidence_percent": 5.0
    },
    {
      "gesture": "C",
      "confidence": 0.02,
      "confidence_percent": 2.0
    }
  ]
}
```

The numbers above are illustrative and are not measured model results.

### Successful dynamic prediction

An illustrative result:

```json
{
  "success": true,
  "error": null,
  "gesture": "HELLO",
  "confidence": 0.89,
  "confidence_percent": 89.0,
  "second_confidence": 0.07,
  "margin": 0.82,
  "margin_percent": 82.0,
  "valid_frames": 24,
  "top_predictions": [
    {
      "gesture": "HELLO",
      "confidence": 0.89,
      "confidence_percent": 89.0
    },
    {
      "gesture": "THANK_YOU",
      "confidence": 0.07,
      "confidence_percent": 7.0
    },
    {
      "gesture": "PLEASE",
      "confidence": 0.02,
      "confidence_percent": 2.0
    }
  ]
}
```

The gesture names and values are examples only. The supported vocabulary depends on the trained dataset.

### Confidence

The `confidence` field contains the model's score for the predicted class. The percentage field multiplies this score by 100.

Although these values are commonly treated as confidence, a neural network's softmax output is not necessarily a calibrated probability of being correct.

### Margin

The margin is calculated as:

```text
margin = top-1 confidence - top-2 confidence
```

A larger margin means the leading class is more separated from the second-best class. A small margin can indicate ambiguity, but it is not a guarantee that a prediction is wrong.

The prediction service returns scores; any confidence threshold used to accept or reject a prediction should be configured and evaluated separately.

---

## 💬 Sentence Builder and Text-to-Speech

The application can combine recognized gestures into text rather than treating every prediction as an isolated result.

### Sentence building

The sentence builder maintains the sequence of recognized symbols or words.

For static recognition, a gesture may represent an alphabet character. Multiple recognized characters can be combined into text.

For dynamic recognition, a gesture may represent a complete word, depending on the model's vocabulary.

### Duplicate protection

Live static recognition can repeatedly detect the same sign while the user holds their hand in position.

Without duplicate protection, a single held sign could be appended multiple times.

The application uses hand-removal or hand-presence locking to help prevent this. The intended interaction is to recognize a sign, wait for the hand to be removed or the recognition state to reset, and then accept the next sign.

The exact reset conditions depend on the current frontend implementation.

### Text-to-speech

Recognized text can be passed to the application's speech functionality so users can listen to the resulting sentence.

Text-to-speech is a separate application feature from the TensorFlow classification models.

---

## 💾 Database and Persistence

Django manages the application's database interactions through its ORM.

The project includes database-backed storage for gesture samples and saved sentence history. The exact models and fields are defined in the application's model files and migrations.

### Dataset storage

The dataset import process stores valid extracted landmark features with their corresponding gesture labels.

This avoids repeating the original image-processing step for every training epoch.

### Sentence history

The sentence-history workflow allows recognized text to be saved and retrieved.

The application has used a route similar to:

```text
/api/vision/save-sentence/
```

Check the current URL configuration for the exact supported request method and payload.

### SQLite

SQLite is suitable for local development and small application deployments. A larger multi-user production deployment may require a database selected for its concurrency, backup, operational, and hosting requirements.

---

## 📈 Model Evaluation

Model quality should be measured using held-out data rather than inferred from a few successful predictions.

### Static model metrics

The static training workflow can evaluate:

- Accuracy
- Weighted F1 score
- Precision and recall by class
- Classification report

### Dynamic model metrics

The dynamic training workflow should use the same evaluation principles, with additional attention to class-specific errors and sequence-level generalization.

### Why accuracy alone is insufficient

Suppose a dataset contains one class with many more samples than the others. A model could achieve high overall accuracy while performing poorly on less frequent classes.

For this reason, macro F1, per-class recall, and a confusion matrix are useful additional metrics.

### Avoiding data leakage

For a stronger evaluation:

- Keep test samples out of all training and model-selection steps.
- Avoid placing near-identical frames from the same recording in both training and testing.
- For dynamic data, split by recording session or participant when appropriate.
- Use the same evaluation split when comparing alternative models.
- Tune thresholds on validation data, not the final test set.

### Reproducibility

Save the class mapping, training configuration, random seeds, data split details, and evaluation report alongside each trained model.

A result is meaningful only when the dataset and evaluation procedure are clearly defined.

---

## ⚙️ Configuration

The following settings are important to the current implementation.

| Setting | Purpose |
|---|---|
| `max_num_hands=1` | Limits MediaPipe to one detected hand |
| `model_complexity=1` | Configures the MediaPipe hand model |
| `min_detection_confidence=0.50` | Minimum hand detection confidence |
| `box_area < 0.003` | Rejects extremely small detected hand regions |
| Minimum 15 valid dynamic frames | Rejects sequences with insufficient valid detections |
| Dynamic target length of 30 frames | Sets the temporal input length expected by the model |
| Static input length of 63 | Number of features for one static sample |
| Dynamic input shape `(30, 63)` | Number of frames and features per frame |

These values should remain consistent with the model training configuration and should be adjusted only after evaluation.

---

## 🧪 Troubleshooting

### 1. Model file not found

**Possible cause:** The `.keras` file is missing from `ml_models/`.

**Solution:**

- Check the directory and filename.
- Confirm that the model was trained successfully.
- Verify that the model path matches `settings.BASE_DIR`.
- Make sure the correct model artifact is present before testing inference.

### 2. Label map file not found

**Possible cause:** The corresponding JSON file is missing.

**Solution:** Restore or regenerate the label map from the training process. Ensure its class ordering matches the saved model.

### 3. No hand detected

**Possible causes:**

- Poor lighting
- Hand outside the camera frame
- Motion blur
- Occluded fingers
- Camera permission problems
- Hand too small in the image

**Solution:** Improve lighting, place the hand inside the frame, and ensure that the hand is clearly visible.

### 4. Dynamic prediction reports insufficient frames

**Possible cause:** Fewer than 15 valid frames were extracted.

**Solution:**

- Record the gesture for longer.
- Keep the hand visible throughout the recording.
- Reduce motion blur.
- Check whether MediaPipe is consistently detecting the hand.

### 5. Incorrect gesture predictions

**Possible causes:**

- Insufficient training samples
- Similar-looking classes
- Inconsistent gesture execution
- Incorrect normalization
- Label-map mismatch
- Training/test data leakage
- Domain differences between training and live camera conditions

**Solution:** Inspect the confusion matrix and class-level metrics, verify the preprocessing pipeline, and collect additional diverse samples for classes with poor recall.

### 6. TensorFlow installation errors

**Possible cause:** An incompatible Python version or conflicting numerical-library dependencies.

**Solution:** Use a clean virtual environment and install versions compatible with the project's dependency requirements. Avoid upgrading every dependency independently without checking compatibility.

### 7. Webcam does not work

**Possible causes:**

- Browser permission denied
- Another application using the camera
- Insecure origin
- Unsupported browser configuration

**Solution:** Check browser permissions, use `localhost` during development, and close other applications that may be using the camera.

### 8. Duplicate letters in the sentence

**Possible cause:** Static recognition repeatedly appends the same prediction while a gesture remains visible.

**Solution:** Check the hand-removal locking logic and ensure the recognition state resets appropriately between signs.

---

## ⚠️ Limitations

- The system recognizes only gestures represented in its trained label maps.
- A hand-landmark model does not directly interpret all facial expressions, body posture, or other non-manual components of sign language.
- Landmark detection can fail under poor lighting, occlusion, unusual camera angles, or motion blur.
- Normalization reduces some position and scale variation but does not remove every source of variation.
- Resampling a dynamic sequence to 30 frames can discard temporal detail.
- The current MediaPipe configuration treats images independently, which can add overhead when processing many video frames.
- Softmax scores may be overconfident on unfamiliar gestures.
- Recognizing individual signs or letters does not by itself provide complete sign-language translation, which also requires linguistic context and grammar.

SignVision AI should be considered a gesture-recognition application rather than a complete replacement for a fluent sign-language interpreter.

---

## 🚀 Future Improvements

Potential improvements include:

- **More diverse datasets:** Collect more examples across different users, backgrounds, lighting conditions, and camera angles.
- **Model benchmarking:** Compare the current static MLP against SVM and gradient-boosted models using the same evaluation split.
- **Dynamic architecture experiments:** Benchmark BiLSTM against GRU and temporal CNN architectures.
- **Improved video processing:** Evaluate MediaPipe video-tracking mode for continuous camera streams.
- **Confidence calibration:** Introduce a validation-based rejection threshold for uncertain predictions.
- **Robust evaluation:** Use participant-separated or recording-session-separated splits to measure generalization.
- **Expanded vocabulary:** Add more supported static and dynamic gesture classes.
- **Better temporal preprocessing:** Investigate methods that preserve relevant whole-hand motion and timing.
- **Deployment optimization:** Evaluate model loading, inference latency, and resource consumption.
- **Accessibility improvements:** Improve keyboard navigation, status messages, visual feedback, and speech controls.
- **Testing:** Add automated tests for landmark normalization, sequence resampling, missing model files, and invalid input handling.

---

## 🤝 Contributing

Contributions are welcome.

A typical contribution workflow is:

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Test the affected functionality.
5. Commit your changes with a clear message.
6. Open a pull request describing the change and how it was tested.

For machine learning changes, include the evaluation setup and explain whether the change affects preprocessing, model architecture, labels, or the training dataset.

---

## 🔒 Privacy and Data Handling

Camera-based recognition should be used with appropriate user consent.

Before deploying the application publicly:

- Explain what camera data is processed.
- Document whether images or recordings are stored.
- Avoid exposing uploaded image data through logs.
- Protect Django secrets and deployment credentials.
- Restrict access to saved user data where appropriate.
- Configure HTTPS and production security settings.
- Review the retention and deletion behavior for stored samples and sentence history.

The actual data-retention behavior depends on the application's implemented views, models, and storage configuration.

---

## 📄 License

Add the license that applies to this repository.

If the project does not yet have a license file, choose an appropriate open-source license and add it to the repository before describing the project as distributed under that license.

---

## 👨‍💻 Author

**Hitesh-XS**

GitHub: https://github.com/Hitesh-XS

Project repository: https://github.com/Hitesh-XS/SiginVision_AI

---

## ⭐ Support the Project

If you find SignVision AI useful, consider starring the repository, reporting issues, suggesting improvements, or contributing code and dataset-quality improvements.

Thank you for checking out SignVision AI!
