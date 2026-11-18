"""FastAPI boundary for PathSnap inference."""
from pathlib import Path
import os
import time
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import load_config
from src.data.loader import TripData, load_trip, validate_trip
from src.pipeline.inference import predict_trip

app = FastAPI(title="PathSnap API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
logger = logging.getLogger("pathsnap.api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

class PredictionRequest(BaseModel):
    trip_id: str

def _uploaded_trip(files: list[UploadFile]) -> TripData:
    """Parse one uploaded trip without persisting user-controlled files."""
    by_suffix: dict[str, UploadFile] = {}
    for upload in files:
        name = Path(upload.filename or "").name
        suffix = Path(name).suffix.lower()
        if name != (upload.filename or "") or suffix not in {".track", ".nodes", ".arcs"}:
            raise ValueError("Only .track, .nodes, and .arcs files with safe names are accepted")
        if suffix in by_suffix:
            raise ValueError(f"Exactly one {suffix} file is required")
        by_suffix[suffix] = upload
    if set(by_suffix) != {".track", ".nodes", ".arcs"}:
        raise ValueError("Exactly one .track, one .nodes, and one .arcs file are required")
    trip_id = Path(by_suffix[".track"].filename).stem
    if not trip_id or Path(trip_id).name != trip_id:
        raise ValueError("The .track filename must provide a valid trip identifier")
    try:
        track = pd.read_csv(by_suffix[".track"].file, sep=r"\s+", header=None,
                            names=["longitude", "latitude", "timestamp"])
        nodes = pd.read_csv(by_suffix[".nodes"].file, sep=r"\s+", header=None,
                            names=["longitude", "latitude"])
        arcs = pd.read_csv(by_suffix[".arcs"].file, sep=r"\s+", header=None,
                           names=["from_node", "to_node"], dtype=int)
    except (pd.errors.ParserError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not parse uploaded trip files: {exc}") from exc
    trip = TripData(trip_id, track, nodes, arcs, None)
    validate_trip(trip)
    if track.empty or nodes.empty or arcs.empty:
        raise ValueError("Uploaded trip files must not be empty")
    if not track[["longitude", "latitude", "timestamp"]].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError(".track contains non-numeric values")
    if not nodes[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError(".nodes contains non-numeric values")
    if (arcs[["from_node", "to_node"]] < 0).any().any() or arcs[["from_node", "to_node"]].to_numpy().max(initial=-1) >= len(nodes):
        raise ValueError(".arcs contains node indices outside .nodes")
    return trip

def _evaluator_root() -> Path:
    value = os.environ.get("PATHSNAP_EVALUATOR_DATASET_ROOT")
    if not value:
        raise HTTPException(status_code=503, detail="PATHSNAP_EVALUATOR_DATASET_ROOT is not configured")
    root = Path(value)
    return root.parent if root.name == "evaluator_test" else root

def _lcs_length(left: list[int], right: list[int]) -> int:
    row = [0] * (len(right) + 1)
    for value in left:
        previous = 0
        for index, target in enumerate(right, 1):
            old = row[index]
            row[index] = previous + 1 if value == target else max(row[index], row[index - 1])
            previous = old
    return row[-1]

def _evaluate_hidden_route(trip_id: str, predicted: list[int]) -> dict[str, float]:
    """Load the evaluator route only after prediction and return aggregates only."""
    route_trip = load_trip(_evaluator_root(), "evaluator_test", trip_id, include_route=True)
    reference = [int(value) for value in route_trip.route.tolist()]
    predicted_set, reference_set = set(map(int, predicted)), set(reference)
    overlap = len(predicted_set & reference_set)
    precision = overlap / max(1, len(predicted_set)); recall = overlap / max(1, len(reference_set))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"arc_set_precision": precision, "arc_set_recall": recall, "arc_set_f1": f1,
            "lcs_route_recall": _lcs_length(list(map(int, predicted)), reference) / max(1, len(reference)),
            "predicted_arc_count": len(predicted), "reference_arc_count": len(reference)}

@app.get("/health")
def health() -> dict[str, object]:
    """Report service readiness without claiming model readiness."""
    config = load_config()
    artifact_dir = Path(config["paths"]["artifacts_dir"])
    model_path = os.environ.get("PATHSNAP_MODEL_PATH")
    model_loaded = (Path(model_path).is_file() if model_path else (any(artifact_dir.glob("*.joblib")) if artifact_dir.is_dir() else False))
    return {"status": "ok", "model_loaded": model_loaded, "inference_ready": model_loaded}

@app.get("/trips")
def trips() -> dict[str, list[str]]:
    """List available trip directories without reading reference routes."""
    config = load_config()
    root = os.environ.get("PATHSNAP_DATASET_ROOT") or config["paths"].get("dataset_root")
    if not root:
        return {"trips": []}
    partition = config["dataset"].get("training_partition", "train")
    path = Path(root) / partition
    return {"trips": sorted(item.name for item in path.iterdir() if item.is_dir())} if path.is_dir() else {"trips": []}

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    """Run real prediction without loading a reference route."""
    config = load_config(); root = os.environ.get("PATHSNAP_DATASET_ROOT") or config["paths"].get("dataset_root")
    model_path = os.environ.get("PATHSNAP_MODEL_PATH")
    if not root or not model_path:
        raise HTTPException(status_code=503, detail="Set PATHSNAP_DATASET_ROOT and PATHSNAP_MODEL_PATH")
    try:
        started = time.perf_counter(); logger.info("POST /predict - Request received; trip_id=%s", request.trip_id)
        trip = load_trip(root, "train", request.trip_id, include_route=False)
        logger.info("Loading GPS observations; trip_id=%s observations=%d", request.trip_id, len(trip.track))
        result = predict_trip(trip, artifact_path=model_path)
        logger.info("Prediction completed; trip_id=%s candidates=%d predictions=%d duration=%.2fs", request.trip_id, result["diagnostics"]["candidate_rows"], len(result["predicted_arc_ids"]), time.perf_counter()-started)
        return result
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("POST /predict - 400; trip_id=%s error=%s", request.trip_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/upload/predict")
def upload_predict(files: list[UploadFile] = File(...)) -> dict[str, object]:
    """Predict an uploaded trip in memory and return its rendered map."""
    started = time.perf_counter()
    model_path = os.environ.get("PATHSNAP_MODEL_PATH")
    if not model_path:
        raise HTTPException(status_code=503, detail="PATHSNAP_MODEL_PATH is not configured")
    try:
        trip = _uploaded_trip(files)
        logger.info("POST /upload/predict - Request received; trip_id=%s observations=%d", trip.trip_id, len(trip.track))
        result = predict_trip(trip, artifact_path=model_path)
        from src.visualization.mapping import create_map
        result["map_html"] = create_map(trip, result["predicted_arc_ids"], classifier_arc_ids=result["classifier_arc_ids"]).get_root().render()
        result["processing_seconds"] = time.perf_counter() - started
        logger.info("POST /upload/predict - 200; trip_id=%s candidates=%d predictions=%d duration=%.2fs",
                    trip.trip_id, result["diagnostics"]["candidate_rows"], len(result["predicted_arc_ids"]), result["processing_seconds"])
        return result
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
        logger.error("POST /upload/predict - 400; error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/map/{trip_id}", response_class=HTMLResponse)
def development_map(trip_id: str) -> str:
    """Return a prediction-only development map for the UI."""
    from src.visualization.mapping import create_map
    config = load_config(); root = os.environ.get("PATHSNAP_DATASET_ROOT") or config["paths"].get("dataset_root"); model_path = os.environ.get("PATHSNAP_MODEL_PATH")
    if not root or not model_path: raise HTTPException(status_code=503, detail="Development dataset/model is not configured")
    try:
        trip = load_trip(root, "train", trip_id, include_route=False); result = predict_trip(trip, artifact_path=model_path)
        return create_map(trip, result["predicted_arc_ids"], classifier_arc_ids=result["classifier_arc_ids"]).get_root().render()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/evaluation/trips")
def evaluation_trips() -> dict[str, list[str]]:
    root = _evaluator_root(); path = root / "evaluator_test"
    if not path.is_dir(): return {"trips": []}
    return {"trips": sorted(p.name for p in path.iterdir() if p.is_dir())}

@app.post("/evaluation/predict")
def evaluation_predict(request: PredictionRequest) -> dict[str, object]:
    started = time.perf_counter(); root = _evaluator_root(); model_path = os.environ.get("PATHSNAP_MODEL_PATH")
    try:
        logger.info("POST /evaluation/predict - Request received; trip_id=%s", request.trip_id)
        available = set(evaluation_trips()["trips"])
        if request.trip_id not in available: raise HTTPException(status_code=404, detail="Unknown evaluator trip")
        if not model_path: raise HTTPException(status_code=503, detail="PATHSNAP_MODEL_PATH is not configured")
        trip = load_trip(root, "evaluator_test", request.trip_id, include_route=False)
        result = predict_trip(trip, artifact_path=model_path)
        metrics = _evaluate_hidden_route(request.trip_id, result["predicted_arc_ids"])
        logger.info("POST /evaluation/predict - 200; trip_id=%s predictions=%d duration=%.2fs", request.trip_id, len(result["predicted_arc_ids"]), time.perf_counter()-started)
        return {"trip_id": request.trip_id, "predicted_arc_ids": result["predicted_arc_ids"],
                "classifier_arc_ids": result["classifier_arc_ids"], "confidence": result["confidence"],
                "diagnostics": result["diagnostics"], "evaluation": metrics,
                "processing_seconds": time.perf_counter() - started}
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/evaluation/map/{trip_id}", response_class=HTMLResponse)
def evaluation_map(trip_id: str) -> str:
    """Return a prediction-only Folium map; no reference route is rendered."""
    from src.visualization.mapping import create_map
    result = evaluation_predict(PredictionRequest(trip_id=trip_id))
    trip = load_trip(_evaluator_root(), "evaluator_test", trip_id, include_route=False)
    return create_map(trip, result["predicted_arc_ids"]).get_root().render()
