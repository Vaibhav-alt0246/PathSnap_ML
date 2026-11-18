"""Model registry for PathSnap ML candidate classification.

Centralizes definitions, hyperparameters, and class-imbalance / voting strategies
for all six candidate classification models:
1. Logistic Regression (balanced class weights in loss function)
2. Gaussian Naive Bayes (uniform priors [0.5, 0.5], intentionally differing from natural class prevalence)
3. K-Nearest Neighbors (distance-weighted voting based on feature proximity; not a class-imbalance correction)
4. Random Forest (balanced subsample tree weighting per bootstrap sample)
5. HistGradientBoosting (balanced sample weighting in gradient/hessian calculations)
6. Multi-Layer Perceptron (per-sample inverse frequency weighting during training)

Approximate Candidate Class Imbalance:
-------------------------------------
In GPS map-matching, each GPS point generates ~10-20 candidate road arcs, of which
only one is the true match (positive class). This creates an approximate 1:15
class imbalance.
To address this without artificially mutating, resampling, or altering the evaluation observations:
- Parametric and tree models adjust loss/sample weights on training observations.
- GaussianNB uses uniform priors [0.5, 0.5] so that conditional feature likelihood governs ranking.
- KNN uses inverse-distance voting so closest candidates dominate neighbor ballots.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    """Specification and metadata for an ML candidate classification model."""

    name: str
    display_name: str
    family: str
    requires_scaling: bool
    requires_sample_weights: bool
    imbalance_strategy: str
    description: str
    factory: Callable[[int], BaseEstimator]
    default_params: Dict[str, Any] = field(default_factory=dict)


def _build_logistic_regression(random_seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=random_seed,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _build_gaussian_nb(random_seed: int = 42) -> Pipeline:
    # Uniform priors [0.5, 0.5] intentionally differ from the natural ~1:15 prevalence
    # to eliminate prior probability bias and allow candidate feature likelihood to drive ranking.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                GaussianNB(
                    priors=[0.5, 0.5],
                ),
            ),
        ]
    )


def _build_knn(random_seed: int = 42) -> Pipeline:
    # weights='distance' is distance-weighted voting where closer neighbors have higher weight.
    # Note: this is distance-weighted voting based on spatial proximity, not an analytical class-imbalance correction.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                ),
            ),
        ]
    )


def _build_random_forest(random_seed: int = 42) -> Pipeline:
    # class_weight='balanced_subsample' computes balanced weights dynamically per bootstrap sample.
    return Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced_subsample",
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _build_hist_gradient_boosting(random_seed: int = 42) -> Pipeline:
    # class_weight='balanced' weights samples inversely proportional to class frequencies.
    return Pipeline(
        [
            (
                "classifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    random_state=random_seed,
                    max_iter=100,
                ),
            ),
        ]
    )


def _build_mlp(random_seed: int = 42) -> Pipeline:
    # Uses sample weights computed via compute_sample_weight('balanced', y_train) during fit.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=300,
                    random_state=random_seed,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        family="Linear",
        requires_scaling=True,
        requires_sample_weights=False,
        imbalance_strategy="class_weight='balanced' (inversely proportional class weighting in loss)",
        description="Standard linear classifier with L2 regularization and balanced loss weighting.",
        factory=_build_logistic_regression,
    ),
    "gaussian_nb": ModelSpec(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        family="Probabilistic / Naive Bayes",
        requires_scaling=True,
        requires_sample_weights=False,
        imbalance_strategy="priors=[0.5, 0.5] (uniform priors intentionally differing from natural class prevalence)",
        description="Generative Gaussian classifier using uniform priors so conditional feature likelihood drives ranking.",
        factory=_build_gaussian_nb,
    ),
    "knn": ModelSpec(
        name="knn",
        display_name="K-Nearest Neighbors",
        family="Instance-based",
        requires_scaling=True,
        requires_sample_weights=False,
        imbalance_strategy="weights='distance' (distance-weighted voting based on proximity; not a class-imbalance correction)",
        description="Instance-based classifier using distance-weighted voting and standard feature scaling.",
        factory=_build_knn,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        family="Ensemble / Bagging",
        requires_scaling=False,
        requires_sample_weights=False,
        imbalance_strategy="class_weight='balanced_subsample' (balanced weights calculated per bootstrap sample)",
        description="Bagging ensemble of decision trees with per-tree balanced bootstrap weighting.",
        factory=_build_random_forest,
    ),
    "hist_gradient_boosting": ModelSpec(
        name="hist_gradient_boosting",
        display_name="Histogram Gradient Boosting",
        family="Ensemble / Boosting",
        requires_scaling=False,
        requires_sample_weights=False,
        imbalance_strategy="class_weight='balanced' (inverse frequency sample weighting per boosting round)",
        description="Fast histogram-based gradient boosting classifier with native class imbalance weighting.",
        factory=_build_hist_gradient_boosting,
    ),
    "mlp": ModelSpec(
        name="mlp",
        display_name="Multi-Layer Perceptron",
        family="Neural Network",
        requires_scaling=True,
        requires_sample_weights=True,
        imbalance_strategy="compute_sample_weight('balanced', y_train) passed to fit()",
        description="Multi-layer feed-forward neural network with early stopping and sample-weighted loss.",
        factory=_build_mlp,
    ),
}


def get_model(name: str, random_seed: int = 42) -> Pipeline:
    """Instantiate a model pipeline by registry name.

    Parameters
    ----------
    name : str
        Model identifier (e.g., 'logistic_regression', 'random_forest').
    random_seed : int
        Random seed for reproducibility. Defaults to 42.

    Returns
    -------
    Pipeline
        Configured scikit-learn Pipeline.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model name '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name].factory(random_seed)


def list_models() -> Dict[str, ModelSpec]:
    """Return all registered model specifications."""
    return dict(MODEL_REGISTRY)
