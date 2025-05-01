# BiteCheck: AI-Powered Food Classification & Health Assessment
*Computer Vision Meets Nutritional Science for Healthier Food Choices*

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
9. [Team Members](#team-members)
10. [Ethical Considerations](#ethical-considerations)
11. [License](#license)

---

## Project Overview

**BiteCheck** is a dual-stage AI system that addresses the challenge of making informed dietary choices, especially in environments like university campuses where nutritional information is often limited or absent. The system:

1. **Classifies food images** using a fine-tuned ResNet50 deep learning model
2. **Assesses nutritional value** through a rule-based mapping system based on WHO and other health guidelines

The project was developed to help Ashesi University students make better food decisions by providing immediate visual analysis of their meal options.

```python
# Example output
{
  "food_class": "hamburger",
  "health_rating": "Unhealthy",
  }
}
```

---

## Problem Statement

Access to nutritious food is essential for student well-being, academic success, and long-term health. However, students often struggle to make informed dietary choices, especially when campus food options lack clear nutritional labeling. At Ashesi University, while vendors offer a variety of meals, students lack the information needed to differentiate between healthy and unhealthy options.

BiteCheck solves this problem by providing an automated system that can classify food images as healthy or unhealthy based solely on visual characteristics, without requiring manual nutritional analysis or database lookups. This transforms what would be an impractical manual task into an accessible tool providing immediate feedback.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Two-Stage Pipeline** | Combines CNN classification with health assessment mapping |
| **Transfer Learning** | Fine-tuned ResNet50 with ~91% accuracy |
| **Custom Augmentation** | Robust image transformations for better generalization |
| **Explainable AI** | Confidence scores + nutritional reasoning |
| **Dictionary Mapping** | WHO/PubMed/HealthLine-backed nutritional rules |
| **Focus on Local Foods** | Trained on 15 categories most common at Ashesi University |
| **Image Preprocessing** | Handles varied image quality, lighting, and angles |

---

## Technical Architecture

### 1. Stage 1: Food Classification (ResNet50)
```
Input Image (224x224 RGB) → ResNet50 Backbone → Global Average Pooling → Dense Layer (128, ReLU) → Dropout (0.2) → 15-class Output with Softmax
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

---

## Model Performance

### Food Classification Model
| Metric | Value |
|--------|-------|
| Final Validation Accuracy | ~91% |
| Training Accuracy | >92% |
| Batch Size | 16 |
| Epochs | 30 |
| Optimizer | SGD (lr=0.0001, momentum=0.9) |
| Regularization | Dropout (0.2) + L2 (λ=0.005) |

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
1. **Directory Structuring and Splitting**:
   - Training (70%): 750 images per class
   - Validation (15%): 250 images per class
   - Testing (15%): 250 images per class

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
- Python 3.11
- NVIDIA GPU (Recommended)
- 8GB RAM minimum

### Steps
```bash
# Clone repository
git clone https://github.com/Marzafiee/BiteCheck-ML-Model.git

# Create virtual environment
python -m venv bitecheck_env
source bitecheck_env/bin/activate  # Linux/Mac
# .\bitecheck_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage Examples

### 1. Single Image Prediction
```python
from bitecheck import BiteCheckAnalyzer

analyzer = BiteCheckAnalyzer()
result = analyzer.predict("food_image.jpg")
print(result)
```

### 2. Batch Processing
```bash
python predict_batch.py --input_dir ./images --output results.csv
```

### 3. Prediction with Nutritional Labeling
```python
# Load the model
model = load_model('best_model_class.keras', compile=False)

# Function to predict class and nutritional label
def predict_with_nutrition(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    # Predict food class
    pred = model.predict(img_array)
    index = np.argmax(pred)
    food_class = list(class_map.keys())[list(class_map.values()).index(index)]
    
    # Get nutritional label
    health_rating = nutri_dict[food_class]
    
    return {
        "food_class": food_class,
        "confidence": float(pred[0][index]),
        "health_rating": health_rating
    }

# Example usage
result = predict_with_nutrition(model, "path/to/food/image.jpg")
print(result)
```

---

## Team Members

**Group 12 - CS254_B: Introduction to Artificial Intelligence**
- Eyra Inez Anne-Marie Agbenu (47152026)
- PraiseGod Ukwuoma Osiagor (63962026)
- Rose Carlene Mpawenayo (49262027)
- Kelly Kethia Gacuti (32742027)

**Instructor:** Dennis Asamoah Owusu

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

### Proposed Improvements
1. Add more food items and regional cuisines to the dictionary
2. Implement a continuous health score instead of binary classification
3. Classify foods along multiple dimensions (multi-label approach)
4. Combine predictions from multiple models for improved accuracy

---

## License

MIT License

```text
Copyright (c) 2025 Ashesi University CS254 Group 12

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
