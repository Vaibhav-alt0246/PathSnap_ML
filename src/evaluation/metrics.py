"""Evaluation metrics for candidate classification, Top-1 accuracy, and prediction evaluation.

This module provides:
1. Candidate-level classification metrics on individual candidate arcs.
2. Observation-level Top-1 accuracy with explicit candidate miss tracking.
3. Prediction evaluation boundary (evaluate_predictions) comparing decoded arc sequences against route data.
"""

from typing import Any, Dict, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def evaluate_predictions(trip: object, predicted_arc_ids: object) -> dict[str, float]:
    """Evaluate predictions against route data in evaluation context.

    Parameters
    ----------
    trip : object
        Trip object with .route attribute containing ground-truth reference arc IDs.
    predicted_arc_ids : Sequence[int] or object
        Predicted sequence of road arc IDs.

    Returns
    -------
    dict[str, float]
        Evaluation metrics dictionary containing accuracy, total_arcs, and correct_arcs.
    """
    route = getattr(trip, "route", None)
    if route is None:
        raise ValueError("route is training/evaluation-only and required for evaluation")

    true_arcs = list(route)
    pred_arcs = list(predicted_arc_ids)

    if not true_arcs:
        return {"accuracy": 0.0, "total_arcs": 0.0, "correct_arcs": 0.0}

    # Sequence alignment accuracy
    correct = sum(1 for p, t in zip(pred_arcs, true_arcs) if p == t)
    total = len(true_arcs)
    acc = float(correct / total) if total > 0 else 0.0

    return {
        "accuracy": acc,
        "total_arcs": float(total),
        "correct_arcs": float(correct),
    }


def compute_candidate_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
) -> Dict[str, Any]:
    """Compute candidate-level classification metrics on individual candidate arcs.

    Parameters
    ----------
    y_true : np.ndarray or Sequence[int]
        True binary labels (0 = negative candidate, 1 = true candidate).
    y_pred : np.ndarray or Sequence[int]
        Predicted binary labels.
    y_prob : np.ndarray or Sequence[float]
        Predicted probabilities. Can be 1D (positive class probabilities) or
        2D of shape (N, 2).

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy: float
        - precision: float
        - recall: float
        - macro_f1: float
        - log_loss: float
        - confusion_matrix: dict with 'tn', 'fp', 'fn', 'tp' and 'matrix'
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    y_pr = np.asarray(y_prob, dtype=float)
    if y_pr.ndim == 1:
        p1 = np.clip(y_pr, 1e-15, 1.0 - 1e-15)
        p0 = 1.0 - p1
        prob_matrix = np.column_stack([p0, p1])
    elif y_pr.ndim == 2:
        if y_pr.shape[1] == 2:
            prob_matrix = np.clip(y_pr, 1e-15, 1.0 - 1e-15)
            prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Expected 2 columns for y_prob, got {y_pr.shape[1]}")
    else:
        raise ValueError(f"Expected 1D or 2D array for y_prob, got ndim={y_pr.ndim}")

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0.0))
    rec = float(recall_score(y_t, y_p, zero_division=0.0))
    f1_m = float(f1_score(y_t, y_p, average="macro", zero_division=0.0))

    try:
        loss = float(log_loss(y_t, prob_matrix, labels=[0, 1]))
    except Exception:
        loss = float("nan")

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1_m,
        "log_loss": loss,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
    }


def compute_top1_metrics(
    df: pd.DataFrame,
    y_prob: np.ndarray | Sequence[float],
    obs_id_cols: Union[str, Sequence[str]] = ("trip_id", "point_idx"),
    target_col: str = "label",
) -> Dict[str, Any]:
    """Compute observation-level Top-1 accuracy and candidate miss statistics.

    For each unique GPS observation:
    - Evaluates whether the candidate with the highest predicted positive-class probability
      is the true road arc (label == 1).
    - If an observation has no candidate with label == 1, it represents a candidate-generation miss
      (the true road arc was never captured in the candidate set).
    - Candidate misses are excluded from the Top-1 accuracy denominator to avoid inflating or
      distorting model classification evaluation.
    - Candidate misses are tracked and reported separately.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of candidate records. Must preserve index correspondence with y_prob.
    y_prob : np.ndarray or Sequence[float]
        Predicted positive-class probabilities for each candidate row in df.
    obs_id_cols : str or Sequence[str]
        Column name or list of column names uniquely identifying each GPS observation.
        Defaults to ('trip_id', 'point_idx').
    target_col : str
        Column name for ground-truth binary label. Defaults to 'label'.

    Returns
    -------
    dict
        Dictionary containing:
        - top1_accuracy: float (top1_correct / evaluable_observations)
        - top1_correct: int
        - evaluable_observations: int
        - candidate_misses: int (observations with no positive candidate)
        - candidate_miss_rate: float (candidate_misses / total_observations)
        - total_observations: int
    """
    if len(df) == 0:
        return {
            "top1_accuracy": 0.0,
            "top1_correct": 0,
            "evaluable_observations": 0,
            "candidate_misses": 0,
            "candidate_miss_rate": 0.0,
            "total_observations": 0,
        }

    y_pr = np.asarray(y_prob, dtype=float)
    if y_pr.ndim == 2:
        if y_pr.shape[1] == 2:
            p_pos = y_pr[:, 1]
        elif y_pr.shape[1] == 1:
            p_pos = y_pr[:, 0]
        else:
            raise ValueError(f"y_prob has unexpected 2D shape {y_pr.shape}")
    elif y_pr.ndim == 1:
        p_pos = y_pr
    else:
        raise ValueError(f"y_prob must be 1D or 2D, got ndim={y_pr.ndim}")

    if len(df) != len(p_pos):
        raise ValueError(
            f"Length mismatch: df has {len(df)} rows but y_prob has {len(p_pos)} items."
        )

    if isinstance(obs_id_cols, str):
        obs_cols = [obs_id_cols]
    else:
        obs_cols = list(obs_id_cols)

    for col in obs_cols:
        if col not in df.columns:
            raise KeyError(f"Observation ID column '{col}' not found in DataFrame.")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Create temporary evaluation frame preserving index
    eval_df = df[obs_cols].copy()
    eval_df["_target"] = df[target_col].values
    eval_df["_prob"] = p_pos

    # Group by observation
    grouped = eval_df.groupby(obs_cols, sort=False)

    top1_correct = 0
    evaluable_observations = 0
    candidate_misses = 0

    for _, group in grouped:
        y_true_obs = group["_target"].values
        probs_obs = group["_prob"].values

        has_positive = np.any(y_true_obs == 1)

        if not has_positive:
            # Candidate-generation miss: true arc is absent from candidate set
            candidate_misses += 1
            continue

        evaluable_observations += 1
        best_candidate_idx = int(np.argmax(probs_obs))

        if y_true_obs[best_candidate_idx] == 1:
            top1_correct += 1

    total_observations = evaluable_observations + candidate_misses
    top1_accuracy = (
        float(top1_correct / evaluable_observations)
        if evaluable_observations > 0
        else 0.0
    )
    candidate_miss_rate = (
        float(candidate_misses / total_observations)
        if total_observations > 0
        else 0.0
    )

    return {
        "top1_accuracy": top1_accuracy,
        "top1_correct": top1_correct,
        "evaluable_observations": evaluable_observations,
        "candidate_misses": candidate_misses,
        "candidate_miss_rate": candidate_miss_rate,
        "total_observations": total_observations,
    }
