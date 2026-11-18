"""Tests for GroupKFold training framework, data leakage prevention, and model comparison."""

import numpy as np
import pandas as pd
import pytest

from src.models.training import (
    DEFAULT_FORBIDDEN_COLUMNS,
    compare_models,
    select_feature_columns,
    train_models,
)


@pytest.fixture
def synthetic_feature_table():
    """Generates a small synthetic candidate table across 10 trips.

    - 10 trips (trip_id: 1..10)
    - 4 observations per trip (point_idx: 0..3)
    - 6 candidate arcs per observation (40 observations * 6 = 240 candidate rows)
    - Exactly 1 true arc per observation (label=1, rest label=0)
    - Safe feature columns: distance_m, projection_fraction, arc_length_m
    - Metadata / forbidden columns: trip_id, point_idx, arc_id, projected_x, projected_y, label
    """
    np.random.seed(42)
    rows = []
    for trip in range(1, 11):
        for pt in range(4):
            for cand in range(6):
                is_true = 1 if cand == 0 else 0
                rows.append(
                    {
                        "trip_id": trip,
                        "point_idx": pt,
                        "arc_id": 1000 * trip + 10 * pt + cand,
                        "label": is_true,
                        "distance_m": float(np.random.uniform(1.0, 15.0) if is_true else np.random.uniform(20.0, 100.0)),
                        "projection_fraction": float(np.random.uniform(0.1, 0.9)),
                        "arc_length_m": float(np.random.uniform(50.0, 300.0)),
                        "projected_x": float(77.5 + np.random.uniform(-0.01, 0.01)),
                        "projected_y": float(12.9 + np.random.uniform(-0.01, 0.01)),
                    }
                )
    return pd.DataFrame(rows)


def test_select_feature_columns_auto(synthetic_feature_table):
    selected = select_feature_columns(synthetic_feature_table)
    assert "distance_m" in selected
    assert "projection_fraction" in selected
    assert "arc_length_m" in selected

    for forbidden in DEFAULT_FORBIDDEN_COLUMNS:
        assert forbidden not in selected, f"Forbidden column '{forbidden}' was auto-selected!"
    assert "projected_x" not in selected
    assert "projected_y" not in selected
    assert "trip_id" not in selected
    assert "point_idx" not in selected
    assert "arc_id" not in selected
    assert "label" not in selected


def test_select_feature_columns_rejects_forbidden(synthetic_feature_table):
    with pytest.raises(ValueError, match="forbidden"):
        select_feature_columns(synthetic_feature_table, feature_cols=["distance_m", "label"])

    with pytest.raises(ValueError, match="forbidden"):
        select_feature_columns(synthetic_feature_table, feature_cols=["distance_m", "trip_id"])

    with pytest.raises(ValueError, match="forbidden"):
        select_feature_columns(synthetic_feature_table, feature_cols=["distance_m", "projected_x"])


def test_train_models_groupkfold_isolation(synthetic_feature_table):
    """Verify that GroupKFold strictly isolates trip IDs across folds."""
    results = train_models(
        feature_table=synthetic_feature_table,
        groups="trip_id",
        n_splits=5,
        models_to_train=["logistic_regression"],
        random_seed=42,
    )

    lr_res = results["logistic_regression"]
    fold_metrics = lr_res["fold_metrics"]
    assert len(fold_metrics) == 5

    for f in fold_metrics:
        assert f["train_trips"] == 8
        assert f["val_trips"] == 2
        assert f["train_samples"] > 0
        assert f["val_samples"] > 0
        assert f["train_time_s"] > 0.0
        assert f["inference_time_s"] >= 0.0


def test_train_models_preserves_row_ordering(synthetic_feature_table):
    """Verify that out-of-fold predictions match input row ordering."""
    results = train_models(
        feature_table=synthetic_feature_table,
        groups="trip_id",
        n_splits=5,
        models_to_train=["logistic_regression"],
        random_seed=42,
    )

    lr_res = results["logistic_regression"]
    oof = lr_res["oof_predictions"]

    assert len(oof) == len(synthetic_feature_table)
    assert oof.shape == (len(synthetic_feature_table), 2)
    assert not np.all(oof == 0.0, axis=1).any()
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5)


def test_train_models_all_six_models(synthetic_feature_table):
    """Verify that all six models run through train_models without error."""
    results = train_models(
        feature_table=synthetic_feature_table,
        groups="trip_id",
        n_splits=3,
        random_seed=42,
    )

    assert len(results) == 6
    expected_models = [
        "logistic_regression",
        "gaussian_nb",
        "knn",
        "random_forest",
        "hist_gradient_boosting",
        "mlp",
    ]

    for model_name in expected_models:
        assert model_name in results
        res = results[model_name]
        agg = res["aggregate_metrics"]
        assert "top1_accuracy_mean" in agg
        assert 0.0 <= agg["top1_accuracy_mean"] <= 1.0
        assert "log_loss_mean" in agg
        assert "macro_f1_mean" in agg
        assert "mean_train_time_s" in agg
        assert "mean_inference_time_s" in agg


def test_compare_models_ranking():
    mock_results = {
        "model_a": {
            "model_name": "model_a",
            "display_name": "Model A",
            "family": "Family A",
            "aggregate_metrics": {
                "top1_accuracy_mean": 0.85,
                "top1_accuracy_std": 0.02,
                "log_loss_mean": 0.35,
                "macro_f1_mean": 0.70,
                "accuracy_mean": 0.90,
                "recall_mean": 0.80,
                "mean_inference_time_s": 0.005,
                "mean_train_time_s": 0.1,
            },
        },
        "model_b": {
            "model_name": "model_b",
            "display_name": "Model B",
            "family": "Family B",
            "aggregate_metrics": {
                "top1_accuracy_mean": 0.92,
                "top1_accuracy_std": 0.01,
                "log_loss_mean": 0.28,
                "macro_f1_mean": 0.78,
                "accuracy_mean": 0.93,
                "recall_mean": 0.85,
                "mean_inference_time_s": 0.010,
                "mean_train_time_s": 0.2,
            },
        },
        "model_c": {
            "model_name": "model_c",
            "display_name": "Model C",
            "family": "Family C",
            "aggregate_metrics": {
                "top1_accuracy_mean": 0.85,
                "top1_accuracy_std": 0.02,
                "log_loss_mean": 0.30,
                "macro_f1_mean": 0.71,
                "accuracy_mean": 0.90,
                "recall_mean": 0.80,
                "mean_inference_time_s": 0.004,
                "mean_train_time_s": 0.08,
            },
        },
    }

    comparison = compare_models(mock_results)

    assert comparison["best_model"] == "model_b"
    assert comparison["ranking"] == ["model_b", "model_c", "model_a"]

    table = comparison["comparison_table"]
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 3
    assert table.iloc[0]["model_name"] == "model_b"
    assert table.iloc[0]["rank"] == 1
    assert table.iloc[1]["model_name"] == "model_c"
    assert table.iloc[1]["rank"] == 2

    assert "Model B" in comparison["selection_rationale"]
