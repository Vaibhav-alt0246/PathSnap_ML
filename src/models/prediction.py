"""Inference-time model prediction.

Provides the inference-time probability prediction interface for candidate road arcs.

Key Requirements:
- Returns one positive-class probability per candidate row.
- Preserves the original input candidate row order.
- Returned probabilities are strictly in the range [0.0, 1.0].
- Never accesses, loads, or depends on .route or ground-truth labels.
- Follows the shared interface contract.
"""

from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

from src.models.training import select_feature_columns

# Global reference for default inference model (can be set via set_default_model)
_DEFAULT_INFERENCE_MODEL: Optional[Any] = None


def set_default_model(model: Any) -> None:
    """Register the active trained model for inference-time predictions."""
    global _DEFAULT_INFERENCE_MODEL
    _DEFAULT_INFERENCE_MODEL = model


def get_default_model() -> Optional[Any]:
    """Retrieve the currently registered active inference model."""
    return _DEFAULT_INFERENCE_MODEL


def predict_proba(
    features: object,
    model: Optional[Any] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return probabilities preserving feature row order.

    Parameters
    ----------
    features : object
        Candidate feature DataFrame or 2D array of feature values.
        Must not contain ground-truth route data.
    model : object, optional
        Fitted scikit-learn model or pipeline with predict_proba().
        If None, uses the active default model registered via set_default_model().
    feature_cols : Sequence[str], optional
        Explicit feature column names to extract from features if it is a DataFrame.

    Returns
    -------
    np.ndarray
        1D numpy array of float positive-class probabilities, preserving input row order.
        All values are in the range [0.0, 1.0].
    """
    active_model = model if model is not None else _DEFAULT_INFERENCE_MODEL
    if active_model is None:
        raise RuntimeError(
            "No trained model provided or loaded for predict_proba. "
            "Pass model=<trained_pipeline> or register one using set_default_model()."
        )

    # If features is a pandas DataFrame, safely extract feature columns
    if isinstance(features, pd.DataFrame):
        # Explicitly verify .route is NOT present or used
        if "route" in features.columns:
            raise ValueError(".route is training-only and must never be present during inference.")

        if len(features) == 0:
            return np.empty(0, dtype=float)

        cols = select_feature_columns(features, feature_cols=feature_cols)
        X = features[cols].values
        n_rows = len(features)
    elif isinstance(features, np.ndarray):
        if len(features) == 0:
            return np.empty(0, dtype=float)
        X = features
        n_rows = len(features)
    else:
        raise TypeError(f"Expected DataFrame or ndarray for features, got {type(features)}")

    # Predict probabilities using model
    raw_probs = active_model.predict_proba(X)

    # Format to 1D positive-class probability array
    if raw_probs.ndim == 2:
        if raw_probs.shape[1] == 2:
            probs = raw_probs[:, 1]
        elif raw_probs.shape[1] == 1:
            probs = raw_probs[:, 0]
        else:
            raise ValueError(f"Unexpected probability output shape {raw_probs.shape}")
    elif raw_probs.ndim == 1:
        probs = raw_probs
    else:
        raise ValueError(f"Unexpected probability ndim={raw_probs.ndim}")

    probs = np.asarray(probs, dtype=float)

    # Verify length matches input rows
    if len(probs) != n_rows:
        raise RuntimeError(
            f"Output probability length ({len(probs)}) does not match input row count ({n_rows})."
        )

    # Verify probability range [0.0, 1.0]
    probs = np.clip(probs, 0.0, 1.0)

    return probs
