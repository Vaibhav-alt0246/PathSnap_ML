"""Tests for ML model registry and individual model families."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.models.registry import MODEL_REGISTRY, get_model, list_models


EXPECTED_MODELS = [
    "logistic_regression",
    "gaussian_nb",
    "knn",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
]


def test_registry_contains_all_six_models():
    models = list_models()
    for name in EXPECTED_MODELS:
        assert name in models, f"Model '{name}' missing from registry."
        spec = models[name]
        assert spec.name == name
        assert len(spec.imbalance_strategy) > 0
        assert callable(spec.factory)


@pytest.mark.parametrize("model_name", EXPECTED_MODELS)
def test_model_pipeline_creation_and_predict_proba(model_name):
    pipeline = get_model(model_name, random_seed=42)
    assert isinstance(pipeline, Pipeline)

    # Synthetic imbalanced data (15 negative, 1 positive)
    np.random.seed(42)
    n_samples = 80
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.array([1 if i % 16 == 0 else 0 for i in range(n_samples)])

    spec = MODEL_REGISTRY[model_name]
    if spec.requires_sample_weights:
        sw = compute_sample_weight("balanced", y)
        pipeline.fit(X, y, classifier__sample_weight=sw)
    else:
        pipeline.fit(X, y)

    probs = pipeline.predict_proba(X)

    assert probs.shape == (n_samples, 2), f"Expected shape ({n_samples}, 2), got {probs.shape}"
    assert np.all(probs >= 0.0), f"{model_name} generated probabilities < 0"
    assert np.all(probs <= 1.0), f"{model_name} generated probabilities > 1"

    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"{model_name} row probabilities do not sum to 1"


def test_model_reproducibility():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.choice([0, 1], size=50, p=[0.9, 0.1])

    m1 = get_model("random_forest", random_seed=42)
    m2 = get_model("random_forest", random_seed=42)

    m1.fit(X, y)
    m2.fit(X, y)

    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)

    assert np.allclose(p1, p2), "Random forest predictions are not reproducible with identical seed."
