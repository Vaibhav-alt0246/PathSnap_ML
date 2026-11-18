import pytest
from fastapi import HTTPException

from api.main import PredictionRequest, health, predict, trips

def test_health_reports_model_state():
    result = health()
    assert result["status"] == "ok"
    assert isinstance(result["inference_ready"], bool)

def test_trips_lists_configured_dataset_without_routes():
    assert isinstance(trips()["trips"], list)

def test_predict_reports_missing_trip(monkeypatch):
    monkeypatch.setenv("PATHSNAP_DATASET_ROOT", "/Users/vaby/Downloads/Student folder")
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", "/tmp/pathsnap-phase2-artifacts/best_baseline_model.joblib")
    with pytest.raises(HTTPException) as error:
        predict(PredictionRequest(trip_id="does-not-exist"))
    assert error.value.status_code == 400
