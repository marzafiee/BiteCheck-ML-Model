"""BiteCheck inference: image bytes in, food class + health rating out."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# Keras flow_from_directory numbers classes in alphabetical folder order.
# Confirmed against train_generator.class_indices in the notebook.
CLASS_NAMES = [
    "chicken_wings",
    "chocolate_cake",
    "donuts",
    "french_fries",
    "french_toast",
    "fried_rice",
    "hamburger",
    "ice_cream",
    "omelette",
    "pancakes",
    "pizza",
    "pork_chop",
    "samosa",
    "spring_rolls",
    "waffles",
]

NUTRI_DICT = {
    "chicken_wings": "unhealthy",
    "chocolate_cake": "unhealthy",
    "donuts": "unhealthy",
    "french_fries": "unhealthy",
    "french_toast": "healthy",
    "fried_rice": "healthy",
    "hamburger": "unhealthy",
    "ice_cream": "unhealthy",
    "omelette": "healthy",
    "pancakes": "healthy",
    "pizza": "unhealthy",
    "pork_chop": "healthy",
    "samosa": "unhealthy",
    "spring_rolls": "unhealthy",
    "waffles": "unhealthy",
}

IMG_SIZE = (224, 224)


class InvalidImageError(ValueError):
    """The uploaded bytes are not an image Pillow can read."""


class ModelUnavailableError(RuntimeError):
    """The model file is missing."""


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Turn raw bytes into the exact input shape the model was trained on."""
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Not a readable image") from exc

    img = img.convert("RGB")  # PNG transparency, greyscale, palette -> 3 channels
    img = img.resize(
        IMG_SIZE, Image.Resampling.NEAREST
    )  # matches Keras load_img default
    arr = np.asarray(img, dtype=np.float32) / 255.0  # same rescale as training
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3): a batch of one


def to_result(probs) -> dict:
    """Turn the model's 15 probabilities into a readable answer."""
    probs = np.asarray(probs).reshape(-1)
    if probs.shape[0] != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} scores, got {probs.shape[0]}")
    idx = int(np.argmax(probs))
    food = CLASS_NAMES[idx]
    return {
        "food_class": food,
        "confidence": round(float(probs[idx]), 4),
        "health_rating": NUTRI_DICT[food],
    }


class Predictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._model = None  # loaded on first use, not at import

    def _get_model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise ModelUnavailableError(
                    f"Model file not found: {self.model_path.name}"
                )
            from tensorflow.keras.models import (
                load_model,
            )  # imported here so tests never need TensorFlow

            self._model = load_model(self.model_path, compile=False)
        return self._model

    def predict(self, image_bytes: bytes) -> dict:
        x = preprocess(image_bytes)
        probs = self._get_model().predict(x, verbose=0)[0]
        return to_result(probs)
