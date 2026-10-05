import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from api.predictor import InvalidImageError, ModelUnavailableError, Predictor

# Path relative to this file, so it works on any machine and from any folder
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv("BITECHECK_MODEL_PATH", REPO_ROOT / "best_model_class.keras")
)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="BiteCheck API", version="1.0.0")
_predictor = Predictor(MODEL_PATH)


def get_predictor() -> Predictor:
    return _predictor  # tests replace this with a fake


@app.get("/health")
def health():
    return {"status": "ok", "model_file_found": MODEL_PATH.exists()}


@app.post("/predict")
def predict(
    file: UploadFile = File(...), predictor: Predictor = Depends(get_predictor)
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WebP image")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)  # never read more than we allow
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB")

    try:
        return predictor.predict(data)
    except InvalidImageError:
        raise HTTPException(status_code=400, detail="The file is not a readable image")
    except ModelUnavailableError:
        raise HTTPException(status_code=503, detail="Model is not available")
