"""Model training and prediction interfaces."""

from src.models.experiment import (
    FIVE_BASELINE_MODELS,
    compute_candidate_recall_diagnostics,
    run_baseline_experiment,
    validate_candidate_dataset,
)
from src.models.prediction import get_default_model, predict_proba, set_default_model
from src.models.registry import MODEL_REGISTRY, get_model, list_models
from src.models.training import compare_models, train_models

__all__ = [
    "train_models",
    "compare_models",
    "predict_proba",
    "set_default_model",
    "get_default_model",
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
    "FIVE_BASELINE_MODELS",
    "run_baseline_experiment",
    "validate_candidate_dataset",
    "compute_candidate_recall_diagnostics",
]
