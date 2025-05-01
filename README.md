# BiteCheck: AI-Powered Food Classification & Health Assessment  
*Computer Vision Meets Nutritional Science*  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technical Architecture](#technical-architecture)
4. [Performance Metrics](#performance-metrics)
5. [Installation Guide](#installation-guide)
6. [Usage Examples](#usage-examples)
7. [Dataset Preparation](#dataset-preparation)
8. [Model Training](#model-training)
9. [Repository Structure](#repository-structure)
10. [License](#license)

---

## Project Overview

**BiteCheck** is a dual-stage AI system that:

1. **Classifies food images** using deep learning (ResNet50 in particular)
2. **Assesses nutritional value** through rule-based mapping

```python
# Example output
{
  "food_class": "hamburger",
  "confidence": 0.92,
  "health_rating": "Unhealthy",
  "nutri_facts": {
    "calories": 354,
    "reason": "High in saturated fats"
  }
}
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Stage Pipeline** | Combines CNN classification with nutritional analysis |
| **High Accuracy** | 90.8% classification accuracy on Food-101 |
| **Custom Augmentation** | Advanced image transformations |
| **Explainable AI** | Confidence scores + top-3 predictions |
| **Health Mapping** | WHO/PubMed-backed nutritional rules |

---

## Technical Architecture

### 1. Classification Stage (ResNet50)
```
Input Image (224x224 RGB) → ResNet50 Backbone → Global Average Pooling → 512-neuron Dense Layer → 101-class Output
```

### 2. Health Assessment Stage
```python
health_rules = {
    "apple": {"rating": "Healthy", "criteria": "Low calorie, high fiber"},
    "pizza": {"rating": "Unhealthy", "criteria": "High saturated fat"}
}
```

---

## Performance Metrics

### Model Evaluation
| Metric | Value |
|--------|-------|
| Test Accuracy | 90.8% |
| Precision | 91.2% |
| Recall | 90.5% |
| AUC | 0.98 |

### Comparative Analysis
| Model | Top-1 Accuracy | Parameters |
|-------|---------------|------------|
| ResNet50 (Ours) | 90.8% | 25M |
| EfficientNetB3 | 92.1% | 12M |
| Baseline CNN | 78.3% | 5M |

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

### 3. Jupyter Notebook
```python
# See full workflow in notebooks/bitecheck-model.ipynb
```

---

## Dataset Preparation

### Food-101 Structure
```
food-101/
├── images/
│   ├── apple_pie/
│   ├── hamburger/
│   └── ... (101 classes)
└── meta/
    ├── classes.txt
    └── test.json
```

### Preprocessing Steps
1. Verify image integrity
2. Resize to 224x224
3. Split dataset (70/15/15)
4. Apply augmentations:
   ```python
   train_datagen = ImageDataGenerator(
       rotation_range=30,
       horizontal_flip=True,
       brightness_range=[0.8,1.2]
   )
   ```

---

## Model Training

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Initial LR | 0.0001 |
| Epochs | 50 |
| Early Stopping | Patience=10 |

---

## Repository Structure

```
BiteCheck/
├── README.md              # Main project documentation (this file)
├── requirements.txt        
├── bitecheck-model.ipynb   
├── data/
│   ├── original/          # Original dataset (untouched)
│   └── processed/         # Processed datasets (train/val/test splits)
├── reports/
│   ├── technical_report.pdf
│   ├── data_documentation.md
│   └── report.pdf         # Main report
└── .gitignore            
```

---

## License

MIT License  

```text
Copyright (c) 2025 Ashesi University CS254 Group 12

Permission is hereby granted... (see LICENSE for full terms)
```
