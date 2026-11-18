"""Reusable raw-trip to classifier/HMM inference orchestration."""
from pathlib import Path
import json
import os
import logging
import numpy as np
from src.candidates.generator import generate_candidates
from src.features.engineering import build_features
from src.models.prediction import load_predictor, predict_proba
from src.hmm.decoder import decode
logger = logging.getLogger("pathsnap.inference")

def predict_trip(trip, artifact_path=None, feature_config="models/config/feature_columns.json"):
    """Predict a trip without accessing ``trip.route``."""
    artifact_path = artifact_path or os.environ.get("PATHSNAP_MODEL_PATH")
    if not artifact_path:
        raise FileNotFoundError("Set PATHSNAP_MODEL_PATH or pass artifact_path")
    candidates = generate_candidates(trip)
    logger.info("Candidate generation completed; trip_id=%s candidates=%d", trip.trip_id, len(candidates))
    features = build_features(trip, candidates)
    model, columns = load_predictor(artifact_path, feature_config)
    probabilities = predict_proba(features, model, columns) if len(features) else np.array([])
    logger.info("Classifier inference completed; trip_id=%s", trip.trip_id)
    if len(candidates):
        candidates = candidates.copy(); candidates["probability"] = probabilities
        candidates["classifier_prediction"] = candidates.groupby("point_idx").probability.transform("max").eq(candidates.probability)
    predicted = decode(trip, candidates, probabilities) if len(candidates) else []
    logger.info("HMM decoding completed; trip_id=%s predictions=%d", trip.trip_id, len(predicted))
    best = candidates.loc[candidates.groupby("point_idx").probability.idxmax()].sort_values("point_idx") if len(candidates) else candidates
    return {"trip_id": trip.trip_id, "predicted_arc_ids": predicted,
            "candidates": candidates.to_dict(orient="records"),
            "classifier_arc_ids": best.arc_id.astype(int).tolist() if len(best) else [],
            "confidence": {"mean_max_probability": float(best.probability.mean()) if len(best) else 0.0,
                           "min_max_probability": float(best.probability.min()) if len(best) else 0.0},
            "diagnostics": {"observations": int(len(trip.track)), "candidate_rows": int(len(candidates)),
                            "missing_candidate_observations": sorted(set(range(len(trip.track))) - set(candidates.point_idx.astype(int))) if len(candidates) else list(range(len(trip.track)))}}
