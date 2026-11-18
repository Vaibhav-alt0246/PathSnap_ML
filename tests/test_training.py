import json

import pandas as pd

from src.models.prediction import load_predictor, predict_proba
from src.models.training import train_models


def _table():
    rows = []
    for trip_id in ("a", "b", "c", "d", "e"):
        for point_idx in range(2):
            rows.extend([
                {"trip_id": trip_id, "point_idx": point_idx, "label": 1, "f": 1.0},
                {"trip_id": trip_id, "point_idx": point_idx, "label": 0, "f": 0.0},
            ])
    return pd.DataFrame(rows)


def test_grouped_training_returns_binary_model_and_fold_metrics():
    table = _table()
    result = train_models(table, table["trip_id"].to_numpy(), ["f"], folds=5)
    assert len(result["fold_metrics"]) == 5
    assert result["oof_probabilities"].shape == (len(table),)
    assert result["model"].predict_proba([[1.0]]).shape == (1, 2)


def test_feature_config_and_artifact_are_compatible(tmp_path):
    table = _table()
    result = train_models(table, table["trip_id"].to_numpy(), ["f"], folds=5)
    import joblib
    artifact = tmp_path / "model.joblib"
    joblib.dump({"model_pipeline": result["model"], "feature_columns": ["f"]}, artifact)
    config = tmp_path / "features.json"
    config.write_text(json.dumps({"feature_columns": ["f"]}))
    model, columns = load_predictor(artifact, config)
    assert columns == ["f"]
    assert predict_proba(pd.DataFrame({"f": [1.0]}), model, columns).shape == (1,)
