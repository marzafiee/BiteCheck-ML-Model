# BiteCheck: AI-Powered Food Classification & Health Assessment

_Computer Vision Meets Nutritional Science for Healthier Food Choices_

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [Technical Architecture](#technical-architecture)
5. [Model Performance](#model-performance)
6. [Dataset Information](#dataset-information)
7. [Installation Guide](#installation-guide)
8. [Usage Examples](#usage-examples)
9. [REST API](#rest-api)
10. [Testing](#testing)
11. [Project Structure](#project-structure)
12. [Ethical Considerations](#ethical-considerations)
13. [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Project Overview

**BiteCheck** is a dual-stage AI system that addresses the challenge of making informed dietary choices, especially in environments like university campuses where nutritional information is often limited or absent. The system:

1. **Classifies food images** using a fine-tuned ResNet50 deep learning model
2. **Assesses nutritional value** through a rule-based mapping system based on WHO and other health guidelines
3. **Serves predictions through a REST API** so other apps can send a photo and get a result back

The project was developed to help Ashesi University students make better food decisions by providing immediate visual analysis of their meal options.

```json
{
  "food_class": "hamburger",
  "confidence": 0.9312,
  "health_rating": "unhealthy"
}
```

---

## Problem Statement

Access to nutritious food is essential for student well-being, academic success, and long-term health. However, students often struggle to make informed dietary choices, especially when campus food options lack clear nutritional labeling. At Ashesi University, while vendors offer a variety of meals, students lack the information needed to differentiate between healthy and unhealthy options.

BiteCheck solves this problem by providing an automated system that can classify food images as healthy or unhealthy based solely on visual characteristics, without requiring manual nutritional analysis or database lookups. This transforms what would be an impractical manual task into an accessible tool providing immediate feedback.

---

## Key Features

| Feature                  | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| **Two-Stage Pipeline**   | Combines CNN classification with health assessment mapping    |
| **Transfer Learning**    | Fine-tuned ResNet50 with ~91% accuracy                        |
| **Custom Augmentation**  | Robust image transformations for better generalization        |
| **Explainable Output**   | Confidence score + health rating for every prediction         |
| **Dictionary Mapping**   | WHO/PubMed/HealthLine-backed nutritional rules                |
| **Focus on Local Foods** | Trained on 15 categories most common at Ashesi University     |
| **REST API**             | FastAPI service with input validation and clear error codes   |
| **Automated Tests**      | 31 pytest tests that run without TensorFlow or the model file |

---

## Technical Architecture

### 1. Stage 1: Food Classification (ResNet50)

```
Input Image (224x224 RGB) -> ResNet50 Backbone -> Global Average Pooling -> Dense Layer (128, ReLU) -> Dropout (0.2) -> 15-class Output with Softmax
```

The model was compiled using SGD optimizer with a learning rate of 0.0001 and momentum of 0.9. Training was conducted over 30 epochs with a batch size of 16.

```python
# ResNet50 Model Setup
resnet50 = ResNet50(weights='imagenet', include_top=False)
x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(n_classes, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
model = Model(inputs=resnet50.input, outputs=predictions)
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
```

### 2. Stage 2: Health Assessment (Dictionary-Based)

```python
# Nutritional labeling dictionary
nutri_dict = {
    'chicken_wings': 'unhealthy',
    'chocolate_cake': 'unhealthy',
    'donuts': 'unhealthy',
    'french_fries': 'unhealthy',
    'french_toast': 'healthy',
    'fried_rice': 'healthy',
    'hamburger': 'unhealthy',
    'ice_cream': 'unhealthy',
    'omelette': 'healthy',
    'pancakes': 'healthy',
    'pizza': 'unhealthy',
    'pork_chop': 'healthy',
    'samosa': 'unhealthy',
    'spring_rolls': 'unhealthy',
    'waffles': 'unhealthy'
}
```

The dictionary classifier was chosen for its interpretability, implementation efficiency, flexibility, and lack of additional data requirements. It maps food classes to health categories based on nutritional guidelines from WHO and other credible health sources.

### 3. Stage 3: Serving (FastAPI)

```
Client uploads photo -> POST /predict -> validate type and size -> preprocess (RGB, 224x224, /255) -> ResNet50 -> top class -> nutri_dict -> JSON response
```

Preprocessing in the API mirrors training exactly (RGB conversion, 224x224 nearest-neighbour resize, rescale to [0, 1]). A mismatch here would silently give wrong predictions, so it is covered by tests.

---

## Model Performance

### Food Classification Model

| Metric                    | Value                         |
| ------------------------- | ----------------------------- |
| Final Validation Accuracy | ~91%                          |
| Training Accuracy         | >92%                          |
| Batch Size                | 16                            |
| Epochs                    | 30                            |
| Optimizer                 | SGD (lr=0.0001, momentum=0.9) |
| Regularization            | Dropout (0.2) + L2 (λ=0.005)  |

### End-to-End Pipeline

- **Stage 1 (Food Classification)**: ~91% accuracy
- **Stage 2 (Health Classification)**: Deterministic mapping
- **Overall System Performance**: ~91% accuracy

### Training Characteristics

- Steady increase in both training and validation accuracy
- Validation loss consistently lower than training loss, suggesting good generalization
- Minimal overfitting observed

---

## Dataset Information

### Data Source

- Modified version of the **Food-101** dataset from Kaggle
- Selected 15 food categories most common at Ashesi University: chicken_wings, chocolate_cake, donuts, french_fries, french_toast, fried_rice, hamburger, ice_cream, omelette, pancakes, pizza, pork_chop, samosa, spring_rolls, waffles
- 1,000 images per category

### Preprocessing Steps

1. **Directory Structuring and Splitting** (per class, using Food-101's official `train.txt` / `test.txt` lists):
   - Training (75%): 750 images per class, 11,250 total
   - Held-out (25%): 250 images per class, 3,750 total
   - The held-out set was used for validation during training (including choosing the best checkpoint), so the ~91% figure is a validation accuracy, not a fully independent test score. A separate test split would give a less biased estimate.

2. **Image Validation and Cleaning**:
   - Used Pillow library to identify and exclude corrupted files

3. **Image Standardization**:
   - All images resized to 224x224 pixels (ResNet50 input requirement)
   - Normalization by rescaling pixel values from [0, 255] to [0, 1]

4. **Data Augmentation** (Training set only):
   - Random shear transformations (shear_range=0.2)
   - Random zooming (zoom_range=0.2)
   - Horizontal flipping

```python
# Data augmentation setup
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
```

### Dataset Challenges

- Varied image quality, lighting, angles, and resolution
- Cluttered backgrounds in some images
- Occasional distortions during preprocessing

---

## Installation Guide

### Prerequisites

- Python 3.11 or 3.12
- NVIDIA GPU recommended for training (not needed for the API or tests)
- 8GB RAM minimum

### Steps

```bash
# Clone repository
git clone https://github.com/marzafiee/BiteCheck-ML-Model.git
cd BiteCheck-ML-Model

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# source .venv/Scripts/activate  # Windows (Git Bash)
# .venv\Scripts\activate         # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt       # training notebook
pip install -r requirements-api.txt   # API and tests
```

### Model file

The trained model (`best_model_class.keras`) is not stored in this repository because of its size. Train it with the notebook, or place a copy in the repository root. To load it from somewhere else, set an environment variable:

```bash
export BITECHECK_MODEL_PATH=/path/to/best_model_class.keras
```

## Usage Examples

### Python

```python
from api.predictor import Predictor

predictor = Predictor("best_model_class.keras")
with open("food_image.jpg", "rb") as image_file:
   print(predictor.predict(image_file.read()))
```

### REST API

```bash
uvicorn api.main:app --reload
# Interactive docs: http://127.0.0.1:8000/docs
```

| Endpoint        | Description                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------- |
| `GET /health`   | Service status and whether the model file was found                                           |
| `POST /predict` | Upload a food photo as form field `file`; returns `food_class`, `confidence`, `health_rating` |

```bash
curl -X POST -F "file=@food_image.jpg" http://127.0.0.1:8000/predict
```

### Error responses

| Status | When                                            |
| ------ | ----------------------------------------------- |
| 400    | Empty file, or the file is not a readable image |
| 413    | Image larger than 5 MB                          |
| 415    | Not a JPEG, PNG or WebP                         |
| 503    | Model file is missing                           |

### Design notes

- **Model loads on first request**, so the server starts quickly and tests never import TensorFlow. Tradeoff: the first prediction is slower. In production, load it at startup instead.
- **The endpoint is a regular (sync) function.** Model inference is CPU-heavy, blocking work, so FastAPI runs it in a thread pool instead of blocking the event loop.
- **Uploads are read up to the size limit only**, so a very large file cannot exhaust memory.
- **Paths are relative to the code**, not the folder you run it from, so it works on any machine.

---

## Testing

```bash
pytest -v
```

31 tests cover preprocessing (shape, value range, transparent/greyscale/palette images, corrupt files), the class-to-health-rating mapping for all 15 classes, and every API response code. A fake model replaces ResNet50 in tests, so they run in seconds without TensorFlow or the model file. They test the serving logic, not model accuracy.

---

## Project Structure

```
BiteCheck-ML-Model/
├── BiteCheck_FoodClassifier.ipynb   # data prep, training, evaluation
├── api/
│   ├── predictor.py                 # preprocessing, model loading, health mapping
│   └── main.py                      # FastAPI app: /health, /predict
├── tests/
│   └── test_api.py                  # pytest suite
├── requirements.txt                 # training dependencies
├── requirements-api.txt             # API and test dependencies
└── pytest.ini
```

---

## Ethical Considerations

While our project used a publicly available food dataset from Kaggle, we recognize several ethical considerations for real-world applications:

- **Bias and Fairness**: Models trained predominantly on Western cuisines may perform poorly with diverse cultural food items. A truly effective system demands representation across global food cultures to ensure inclusivity.

- **Cultural Sensitivity**: Food health perception varies across cultures, leading to potential cultural bias in health classifications.

- **Limitations of Binary Classification**: Health exists on a spectrum, not in binary categories, and varies per person.

- **Contextual Limitations**: The system doesn't account for portion size and preparation methods.

Throughout our work, we maintained proper attribution to the original Food-101 dataset creators, respecting intellectual property rights.

---

## Limitations and Future Improvements

### Current Limitations

1. Not all possible food items are included in the dictionary
2. Binary classification doesn't capture the spectrum of healthiness
3. No consideration for portion size and preparation methods
4. Potential cultural bias in health assessments
5. The model always picks one of 15 classes, even for a photo that is not food
6. No separate test set: the held-out split was also used to select the best checkpoint
7. The training notebook uses absolute local paths; set `dataset_path` to your own location before running it

### Proposed Improvements

1. Add more food items and regional cuisines to the dictionary
2. Implement a continuous health score instead of binary classification
3. Classify foods along multiple dimensions (multi-label approach)
4. Combine predictions from multiple models for improved accuracy
5. Return "unsure" below a confidence threshold instead of forcing a class
6. Run the test suite automatically on every push with GitHub Actions
