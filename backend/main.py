from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from torch import nn


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "webcam_asl_rgb.pt"
SUMMARY_PATH = ROOT / "evaluation" / "asl_hg_webcam" / "evaluation_summary.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WebcamRGBClassifier(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.35), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


app = FastAPI(title="GESTURA model API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

model = None
label_map: dict[str, int] = {}
index_to_letter: dict[int, str] = {}
model_error: str | None = None
summary: dict = {}


@app.on_event("startup")
def load_model() -> None:
    global model, label_map, index_to_letter, model_error, summary
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model checkpoint missing: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        label_map = {str(k): int(v) for k, v in checkpoint["label_map"].items()}
        index_to_letter = {v: k for k, v in label_map.items()}
        model = WebcamRGBClassifier(int(checkpoint["num_classes"])).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        if SUMMARY_PATH.is_file():
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        model = None
        model_error = str(exc)


class PredictionRequest(BaseModel):
    # Row-major RGB pixels for a 64x64 webcam crop, integer values in [0,255].
    pixels: list[int]


class QuizSubmission(BaseModel):
    total: int
    correct: int


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/api/model/status")
def model_status():
    return {
        "loaded": model is not None,
        "error": model_error,
        "task": "static ASL fingerspelling letters",
        "dataset": summary.get("dataset"),
        "classes": sorted(label_map.keys()),
        "input": {"width": 64, "height": 64, "channels": 3, "pixel_range": [0, 255], "layout": "row-major RGB"},
        "test_accuracy": summary.get("test_accuracy"),
        "test_macro_f1": summary.get("test_macro_f1"),
        "validation_accuracy": summary.get("best_validation_accuracy"),
        "test_rows": summary.get("test_rows"),
        "signer_independent_test": summary.get("signer_independent_split"),
        "hand_detection_coverage": summary.get("hand_detection_coverage"),
        "evaluation_note": summary.get("note"),
        "device": str(DEVICE),
    }


@app.get("/api/signs")
def signs():
    return {"language": "ASL", "scope": "static fingerspelling letters only", "classes": sorted(label_map.keys())}


@app.get("/api/learning/categories")
def learning_categories():
    return {"language": "ASL", "categories": [{"id": "fingerspelling", "name": "Fingerspelling alphabet", "scope": "Study A–Z; the still-image model supports only A–I and K–Y."}]}


@app.get("/api/learning/flashcards")
def learning_flashcards():
    return {
        "language": "ASL",
        "source": "https://www.lifeprint.com/asl101/fingerspelling/abc.htm",
        "image_source": "Wikimedia Commons, public-domain ASL alphabet chart",
        "image_source_url": "https://commons.wikimedia.org/wiki/File:Asl_alphabet_gallaudet.svg",
        "model_dataset": "Mendeley ASL-HG v1 (CC BY 4.0)",
        "model_dataset_url": "https://data.mendeley.com/datasets/j4y5w2c8w9/1",
        "cards": [{
            "letter": letter,
            "model_supported": letter in label_map,
            "motion_required": letter in {"J", "Z"},
            "image_url": f"/learning/asl/asl-letter-{letter}.svg?v=20261107-4",
            "reference_url": "https://www.lifeprint.com/asl101/fingerspelling/abc.htm",
        } for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    }


@app.post("/api/learning/quiz/submit")
def submit_learning_quiz(submission: QuizSubmission):
    if submission.total < 1 or submission.correct < 0 or submission.correct > submission.total:
        raise HTTPException(status_code=422, detail="correct must be between zero and total, and total must be positive")
    return {"correct": submission.correct, "total": submission.total, "score_percent": round(100 * submission.correct / submission.total, 1)}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {model_error or 'not loaded'}")
    if len(request.pixels) != 64 * 64 * 3:
        raise HTTPException(status_code=422, detail="pixels must contain exactly 12288 row-major RGB values for a 64x64 image")
    if any(value < 0 or value > 255 for value in request.pixels):
        raise HTTPException(status_code=422, detail="pixel values must be integers from 0 through 255")
    x = torch.tensor(request.pixels, dtype=torch.float32, device=DEVICE).reshape(1, 64, 64, 3).permute(0, 3, 1, 2) / 255.0
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probabilities, k=min(3, probabilities.numel()))
    prediction_index = int(top.indices[0].item())
    return {
        "sign": index_to_letter[prediction_index],
        "confidence": float(top.values[0].item()),
        "top_predictions": [
            {"sign": index_to_letter[int(idx.item())], "confidence": float(prob.item())}
            for idx, prob in zip(top.indices, top.values)
        ],
        "scope": "static ASL fingerspelling letter; not word or sentence translation",
    }


# In a production Docker build, serve the built Vite app and API from one HTTPS
# origin so webcam access works without cross-origin configuration.
FRONTEND_DIST = ROOT.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
