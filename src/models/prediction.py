"""Inference-time model prediction."""
import json
from pathlib import Path
import joblib

def load_predictor(artifact_path, feature_config):
    artifact = joblib.load(artifact_path)
    config = json.loads(Path(feature_config).read_text())
    model = artifact.get("model_pipeline", artifact) if isinstance(artifact, dict) else artifact
    columns = artifact.get("feature_columns", config["feature_columns"]) if isinstance(artifact, dict) else config["feature_columns"]
    return model, columns

def predict_proba(features, model, feature_columns):
    """Return positive-class probabilities preserving candidate row order."""
    missing = set(feature_columns) - set(features.columns)
    if missing: raise ValueError(f"missing model features: {sorted(missing)}")
    # The deployed estimator was fitted from arrays. Preserve the authoritative
    # configured column order, then pass an array to avoid sklearn's mixed
    # feature-name warning without changing any feature values.
    return model.predict_proba(features.loc[:, feature_columns].to_numpy())[:, 1]
