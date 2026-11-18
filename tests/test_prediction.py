"""Tests for inference prediction interface (predict_proba)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.prediction import (
    get_default_model,
    predict_proba,
    set_default_model,
)


def test_predict_proba_with_explicit_model():
    # Train dummy logistic regression
    np.random.seed(42)
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])
    y_train = np.array([0, 0, 1, 1])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Feature table
    df_features = pd.DataFrame(
        {
            "feat1": [1.5, 7.0],
            "feat2": [2.5, 8.0],
            "point_idx": [0, 1],  # forbidden metadata
            "arc_id": [10, 20],   # forbidden metadata
        }
    )

    probs = predict_proba(df_features, model=clf, feature_cols=["feat1", "feat2"])

    # Returns 1D array of positive probabilities
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Higher feature values should yield higher positive class probability
    assert probs[1] > probs[0]


def test_predict_proba_with_default_model():
    X_train = np.array([[1.0, 2.0], [8.0, 9.0]])
    y_train = np.array([0, 1])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    set_default_model(clf)
    assert get_default_model() is clf

    df_features = pd.DataFrame({"feat1": [2.0], "feat2": [3.0]})
    probs = predict_proba(df_features)

    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


def test_predict_proba_rejects_route_column():
    df_with_route = pd.DataFrame(
        {
            "distance_m": [10.0],
            "route": [101],  # Ground-truth route!
        }
    )
    with pytest.raises(ValueError, match="\\.route is training-only"):
        predict_proba(df_with_route, model=LogisticRegression())


def test_predict_proba_empty():
    df_empty = pd.DataFrame(columns=["distance_m"])
    probs = predict_proba(df_empty, model=LogisticRegression())
    assert len(probs) == 0
