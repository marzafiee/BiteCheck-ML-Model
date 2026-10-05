import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import MAX_UPLOAD_BYTES, app, get_predictor
from api.predictor import (
    CLASS_NAMES,
    NUTRI_DICT,
    InvalidImageError,
    ModelUnavailableError,
    Predictor,
    preprocess,
    to_result,
)


def image_bytes(mode="RGB", size=(300, 200), fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format=fmt)
    return buf.getvalue()


class FakeModel:
    """Stands in for the Keras model: no TensorFlow, no 100 MB file."""

    def __init__(self, top_class):
        self.top_index = CLASS_NAMES.index(top_class)
        self.last_input_shape = None

    def predict(self, x, verbose=0):
        self.last_input_shape = x.shape
        probs = np.full((1, len(CLASS_NAMES)), 0.01, dtype=np.float32)
        probs[0, self.top_index] = 0.86
        return probs


def fake_predictor(top_class="hamburger"):
    p = Predictor("not-used.keras")
    p._model = FakeModel(top_class)
    return p


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = lambda: fake_predictor("hamburger")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def upload(client, data, content_type="image/png", name="meal.png"):
    return client.post("/predict", files={"file": (name, data, content_type)})


# ---------- unit tests: predictor.py ----------


def test_every_class_has_a_health_rating():
    assert sorted(NUTRI_DICT) == CLASS_NAMES


def test_preprocess_shape_type_and_range():
    x = preprocess(image_bytes(size=(640, 480), fmt="JPEG"))
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_preprocess_converts_any_image_to_three_channels(mode):
    assert preprocess(image_bytes(mode=mode)).shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image_bytes():
    with pytest.raises(InvalidImageError):
        preprocess(b"definitely not an image")


@pytest.mark.parametrize("food", CLASS_NAMES)
def test_to_result_maps_each_class_to_its_rating(food):
    probs = np.zeros(len(CLASS_NAMES))
    probs[CLASS_NAMES.index(food)] = 1.0
    result = to_result(probs)
    assert result["food_class"] == food
    assert result["health_rating"] == NUTRI_DICT[food]
    assert result["confidence"] == 1.0


def test_to_result_rejects_wrong_number_of_scores():
    with pytest.raises(ValueError):
        to_result(np.ones(3))


def test_missing_model_file_raises():
    with pytest.raises(ModelUnavailableError):
        Predictor("does-not-exist.keras").predict(image_bytes())


# ---------- API tests: main.py ----------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_class_confidence_and_rating(client):
    r = upload(client, image_bytes())
    assert r.status_code == 200
    body = r.json()
    assert body["food_class"] == "hamburger"
    assert body["health_rating"] == "unhealthy"
    assert body["confidence"] == pytest.approx(0.86, abs=1e-4)


def test_model_receives_the_trained_input_shape():
    predictor = fake_predictor("pizza")
    app.dependency_overrides[get_predictor] = lambda: predictor
    with TestClient(app) as c:
        r = upload(c, image_bytes(size=(50, 80)))
    app.dependency_overrides.clear()
    assert r.json()["food_class"] == "pizza"
    assert predictor._model.last_input_shape == (1, 224, 224, 3)


def test_rejects_wrong_file_type(client):
    assert upload(client, b"hello", "text/plain", "notes.txt").status_code == 415


def test_rejects_empty_file(client):
    assert upload(client, b"").status_code == 400


def test_rejects_corrupt_image(client):
    assert upload(client, b"not really a png").status_code == 400


def test_rejects_file_over_size_limit(client):
    assert upload(client, b"0" * (MAX_UPLOAD_BYTES + 1)).status_code == 413


def test_returns_503_when_model_missing():
    app.dependency_overrides[get_predictor] = lambda: Predictor("missing.keras")
    with TestClient(app) as c:
        r = upload(c, image_bytes())
    app.dependency_overrides.clear()
    assert r.status_code == 503
