from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pathsnap.data import list_trips, load_trip
from pathsnap.inference import load_model, predict_trip

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", ROOT / "data" / "Student folder"))
TEST_PARTITION = os.environ.get("PATHSNAP_TEST_PARTITION", "student_test")
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "artifacts" / "model.joblib"))

app = FastAPI(title="PathSnap Map Matching API", version="1.0.0")
origins = os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])
if (ROOT / "ui").exists():
    app.mount("/ui", StaticFiles(directory=ROOT / "ui", html=True), name="ui")


@app.get("/")
def home():
    return FileResponse(ROOT / "ui" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists(), "evaluation_partition": TEST_PARTITION}


@app.get("/trips")
def trips():
    if not DATA_ROOT.exists():
        raise HTTPException(503, "Trip data folder is missing")
    return [{"trip_id": t.trip_id, "observations": int(len(t.gps))} for t in list_trips(DATA_ROOT, TEST_PARTITION)]


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Trained model is missing; run python train.py first")
    path = DATA_ROOT / TEST_PARTITION / trip_id
    if not path.is_dir() or path.resolve().parent != (DATA_ROOT / TEST_PARTITION).resolve():
        raise HTTPException(404, "Unknown trip")
    try:
        trip = load_trip(path, TEST_PARTITION)
        model = load_model(MODEL_PATH)
        result = predict_trip(model, trip)
        print(f"POST /predict/{trip_id}: scored {result['metrics']['scored_observations']}/{result['metrics']['observations']} observations", flush=True)
        return result
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc

