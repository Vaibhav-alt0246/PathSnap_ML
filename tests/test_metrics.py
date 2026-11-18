"""Tests for candidate-level, observation-level Top-1 metrics, and evaluate_predictions."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_candidate_metrics,
    compute_top1_metrics,
    evaluate_predictions,
)


def test_candidate_metrics_perfect():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.85, 0.05])

    metrics = compute_candidate_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0
    assert metrics["log_loss"] > 0.0
    assert metrics["log_loss"] < 0.5


def test_candidate_metrics_imbalanced_zeros():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.05, 0.1, 0.02, 0.15])

    metrics = compute_candidate_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["tp"] == 0


def test_top1_metrics_perfect():
    df = pd.DataFrame(
        {
            "trip_id": [1, 1, 1, 1, 1, 1],
            "point_idx": [0, 0, 0, 1, 1, 1],
            "arc_id": [101, 102, 103, 201, 202, 203],
            "label": [0, 1, 0, 1, 0, 0],
        }
    )
    y_prob = np.array([0.1, 0.8, 0.1, 0.9, 0.05, 0.05])

    metrics = compute_top1_metrics(df, y_prob, obs_id_cols=["trip_id", "point_idx"])

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top1_correct"] == 2
    assert metrics["evaluable_observations"] == 2
    assert metrics["candidate_misses"] == 0
    assert metrics["candidate_miss_rate"] == 0.0
    assert metrics["total_observations"] == 2


def test_top1_metrics_with_candidate_misses():
    df = pd.DataFrame(
        {
            "trip_id": [1, 1, 1, 1, 1, 1],
            "point_idx": [0, 0, 1, 1, 2, 2],
            "arc_id": [10, 11, 20, 21, 30, 31],
            "label": [1, 0, 1, 0, 0, 0],  # obs 2 has no true arc
        }
    )
    y_prob = np.array([0.7, 0.3, 0.15, 0.85, 0.6, 0.4])

    metrics = compute_top1_metrics(df, y_prob, obs_id_cols=["trip_id", "point_idx"])

    assert metrics["total_observations"] == 3
    assert metrics["candidate_misses"] == 1
    assert metrics["candidate_miss_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["evaluable_observations"] == 2
    assert metrics["top1_correct"] == 1
    assert metrics["top1_accuracy"] == pytest.approx(0.5)


def test_top1_metrics_empty():
    df = pd.DataFrame(columns=["trip_id", "point_idx", "label"])
    metrics = compute_top1_metrics(df, np.array([]), obs_id_cols=["trip_id", "point_idx"])
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["total_observations"] == 0


def test_evaluate_predictions_requires_route():
    class DummyTripNoRoute:
        pass

    with pytest.raises(ValueError, match="route is training/evaluation-only"):
        evaluate_predictions(DummyTripNoRoute(), [10, 20])


def test_evaluate_predictions_with_route():
    class DummyTripWithRoute:
        route = [101, 102, 103, 104]

    trip = DummyTripWithRoute()
    # 3 of 4 match
    preds = [101, 102, 999, 104]
    result = evaluate_predictions(trip, preds)

    assert result["accuracy"] == 0.75
    assert result["total_arcs"] == 4.0
    assert result["correct_arcs"] == 3.0
