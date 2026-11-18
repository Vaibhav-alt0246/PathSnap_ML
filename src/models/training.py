"""Model training boundaries and comparison framework for PathSnap ML.

Implements GroupKFold cross-validation grouped strictly by trip ID to prevent
temporal, spatial, or trajectory-level data leakage between training and validation sets.

Key Principles:
- GroupKFold splitting by trip ID (never random row-level splitting).
- Identical folds precomputed and applied across all models.
- Preprocessing and scalers fitted strictly within each fold on training observations.
- Automatic exclusion of target, trip identifiers, observation identifiers, candidate IDs,
  raw coordinates, and route-derived reference columns.
- Candidate row ordering preserved in out-of-fold predictions.
- Training and inference time recorded per fold.
- Multi-criteria model comparison and ranking based on validation Top-1 accuracy,
  log loss, macro F1, stability, and latency.

Feature Whitelist Note:
----------------------
Automatic numeric-column detection is a temporary fallback mechanism strictly for
software testing before Member A provides the confirmed feature schema.
Production training MUST provide an explicit feature whitelist from Member A.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation.metrics import compute_candidate_metrics, compute_top1_metrics
from src.models.registry import MODEL_REGISTRY

# Default set of forbidden columns that must never be used as model features
DEFAULT_FORBIDDEN_COLUMNS = {
    # Target columns
    "label",
    "target",
    "y",
    "is_match",
    "ground_truth",
    "matched",
    # Group & Trajectory identifiers
    "trip_id",
    "trip",
    "group",
    "groups",
    # Observation identifiers
    "point_idx",
    "point_id",
    "obs_id",
    "observation_id",
    # Candidate identifiers
    "arc_id",
    "candidate_id",
    "edge_id",
    # Raw geographic coordinates (leak location identity & trip path)
    "projected_x",
    "projected_y",
    "obs_x",
    "obs_y",
    "gps_x",
    "gps_y",
    "lat",
    "lon",
    "latitude",
    "longitude",
    "x",
    "y",
    # Route-derived reference columns (inference target leakage)
    "route_arc_id",
    "route_id",
    "aligned_arc",
    "reference_arc",
}


def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    forbidden_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """Identify and validate safe feature columns for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate feature table.
    feature_cols : Sequence[str], optional
        Explicit whitelist of feature columns to use (from Member A).
        If provided, checks against forbidden columns.
    forbidden_cols : Sequence[str], optional
        Additional columns to forbid. Defaults to DEFAULT_FORBIDDEN_COLUMNS.

    Returns
    -------
    List[str]
        Validated list of safe feature column names.
    """
    forbidden = set(DEFAULT_FORBIDDEN_COLUMNS)
    if forbidden_cols:
        forbidden.update(c.lower() for c in forbidden_cols)

    if feature_cols is not None:
        # Whitelist specified: verify none are forbidden
        selected = []
        for col in feature_cols:
            if col.lower() in forbidden:
                raise ValueError(
                    f"Column '{col}' is in the forbidden features list and cannot be used for training."
                )
            if col not in df.columns:
                raise KeyError(f"Feature column '{col}' not found in DataFrame.")
            selected.append(col)
        return selected

    # Temporary fallback: auto-detect numeric feature columns excluding forbidden names.
    # Note: Production experiments must specify the confirmed whitelist from Member A.
    selected = []
    for col in df.columns:
        if col.lower() in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            selected.append(col)

    if not selected:
        raise ValueError(
            "No valid feature columns found in table after excluding forbidden columns."
        )

    return selected


def train_models(
    feature_table: object,
    groups: object = "trip_id",
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = "label",
    obs_id_cols: Union[str, Sequence[str]] = ("trip_id", "point_idx"),
    n_splits: int = 5,
    random_seed: int = 42,
    models_to_train: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train candidate classifiers with trip-grouped validation.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Candidate-level feature table containing features, target, and group/observation identifiers.
    groups : array-like or str
        Group labels for observations (trip IDs), or string name of the group column in feature_table.
    feature_cols : Sequence[str], optional
        Explicit feature whitelist from Member A. If None, safe numeric columns are auto-detected (fallback).
    target_col : str
        Name of binary target column. Defaults to 'label'.
    obs_id_cols : str or Sequence[str]
        Column(s) identifying GPS observations for Top-1 evaluation. Defaults to ('trip_id', 'point_idx').
    n_splits : int
        Number of GroupKFold splits. Defaults to 5.
    random_seed : int
        Random seed for reproducibility. Defaults to 42.
    models_to_train : Sequence[str], optional
        List of model registry keys to train. If None, trains all registered models.

    Returns
    -------
    dict
        Dictionary mapping model_name -> model results containing:
        - model_name: str
        - display_name: str
        - feature_names: list of str
        - fold_metrics: list of per-fold metric dictionaries
        - aggregate_metrics: dict of mean and std for all metrics across folds
        - oof_predictions: np.ndarray of shape (N, 2), preserving candidate-row ordering
        - total_train_time_s: float
        - total_inference_time_s: float
    """
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")

    if target_col not in feature_table.columns:
        raise KeyError(f"Target column '{target_col}' not found in feature_table.")

    # Resolve groups
    if isinstance(groups, str):
        if groups not in feature_table.columns:
            raise KeyError(f"Group column '{groups}' not found in feature_table.")
        group_arr = feature_table[groups].values
    else:
        group_arr = np.asarray(groups)

    if len(group_arr) != len(feature_table):
        raise ValueError(
            f"Length mismatch: groups has {len(group_arr)} items but feature_table has {len(feature_table)} rows."
        )

    # Validate feature columns
    features = select_feature_columns(feature_table, feature_cols=feature_cols)

    # Determine unique groups and validate n_splits
    unique_groups = np.unique(group_arr)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Cannot perform {n_splits}-fold GroupKFold with only {len(unique_groups)} unique groups."
        )

    # Precompute identical GroupKFold splits for all models
    gkf = GroupKFold(n_splits=n_splits)
    X_full = feature_table[features].values
    y_full = feature_table[target_col].values.astype(int)

    # Generate and store fold splits
    splits = list(gkf.split(X_full, y_full, groups=group_arr))

    # Determine models to train
    if models_to_train is None:
        target_model_names = list(MODEL_REGISTRY.keys())
    else:
        target_model_names = list(models_to_train)

    overall_results: Dict[str, Any] = {}
    n_samples = len(feature_table)

    for model_name in target_model_names:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

        spec = MODEL_REGISTRY[model_name]
        fold_records: List[Dict[str, Any]] = []
        oof_probs = np.zeros((n_samples, 2), dtype=float)

        total_train_time = 0.0
        total_inf_time = 0.0

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Verify no group overlap (strict GroupKFold check)
            train_groups = set(group_arr[train_idx])
            val_groups = set(group_arr[val_idx])
            overlap = train_groups.intersection(val_groups)
            if overlap:
                raise RuntimeError(
                    f"Group leakage detected in fold {fold_idx}: {overlap} present in both train and val."
                )

            X_tr, y_tr = X_full[train_idx], y_full[train_idx]
            X_val, y_val = X_full[val_idx], y_full[val_idx]

            # Build fresh pipeline instance for this fold
            pipeline = spec.factory(random_seed)

            # Fit model and measure train time
            t0 = time.perf_counter()
            if spec.requires_sample_weights:
                sw = compute_sample_weight("balanced", y_tr)
                pipeline.fit(X_tr, y_tr, classifier__sample_weight=sw)
            else:
                pipeline.fit(X_tr, y_tr)
            t_train = time.perf_counter() - t0
            total_train_time += t_train

            # Predict and measure inference time
            t1 = time.perf_counter()
            probs_val = pipeline.predict_proba(X_val)
            t_inf = time.perf_counter() - t1
            total_inf_time += t_inf

            # Ensure probabilities have 2 columns
            if probs_val.shape[1] == 1:
                p1 = probs_val[:, 0]
                probs_val = np.column_stack([1.0 - p1, p1])

            # Preserve row ordering into out-of-fold predictions
            oof_probs[val_idx] = probs_val

            # Candidate-level predictions (threshold 0.5)
            preds_val = (probs_val[:, 1] >= 0.5).astype(int)

            # Compute candidate-level metrics
            cand_metrics = compute_candidate_metrics(y_val, preds_val, probs_val)

            # Compute observation-level Top-1 metrics
            df_val = feature_table.iloc[val_idx]
            top1_metrics = compute_top1_metrics(
                df=df_val,
                y_prob=probs_val[:, 1],
                obs_id_cols=obs_id_cols,
                target_col=target_col,
            )

            fold_record = {
                "fold": fold_idx,
                "train_trips": len(train_groups),
                "val_trips": len(val_groups),
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "train_time_s": t_train,
                "inference_time_s": t_inf,
                "candidate_metrics": cand_metrics,
                "top1_metrics": top1_metrics,
            }
            fold_records.append(fold_record)

        # Compute aggregate metrics across folds
        agg_metrics: Dict[str, float] = {}

        # Candidate-level aggregates
        for k in ["accuracy", "precision", "recall", "macro_f1", "log_loss"]:
            vals = [f["candidate_metrics"][k] for f in fold_records if not np.isnan(f["candidate_metrics"][k])]
            agg_metrics[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg_metrics[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")

        # Top-1 observation aggregates
        for k in ["top1_accuracy", "candidate_miss_rate"]:
            vals = [f["top1_metrics"][k] for f in fold_records]
            agg_metrics[f"{k}_mean"] = float(np.mean(vals)) if vals else 0.0
            agg_metrics[f"{k}_std"] = float(np.std(vals)) if vals else 0.0

        agg_metrics["mean_train_time_s"] = float(np.mean([f["train_time_s"] for f in fold_records]))
        agg_metrics["mean_inference_time_s"] = float(np.mean([f["inference_time_s"] for f in fold_records]))

        overall_results[model_name] = {
            "model_name": model_name,
            "display_name": spec.display_name,
            "family": spec.family,
            "imbalance_strategy": spec.imbalance_strategy,
            "feature_names": features,
            "fold_metrics": fold_records,
            "aggregate_metrics": agg_metrics,
            "oof_predictions": oof_probs,
            "total_train_time_s": total_train_time,
            "total_inference_time_s": total_inf_time,
        }

    return overall_results


def compare_models(results: object) -> Dict[str, Any]:
    """Compare model results.

    Ranking Hierarchy:
    1. Primary criterion: Mean GroupKFold Top-1 observation accuracy (higher is better).
    2. Diagnostic & Tiebreaker: Mean candidate-level Log Loss (lower is better).
    3. Secondary metric: Mean Macro F1 score (higher is better).
    4. Fold stability: Standard deviation of Top-1 accuracy across folds (lower is better).
    5. Efficiency: Mean inference time per fold (lower is better).

    Strictly ignores training accuracy during selection.

    Parameters
    ----------
    results : Dict[str, Any]
        Output dictionary from train_models(), mapping model_name -> result dict.

    Returns
    -------
    dict
        Dictionary containing:
        - comparison_table: pd.DataFrame summarizing all models sorted by rank
        - best_model: str, key of the top-ranked model
        - best_model_details: dict of metrics for the best model
        - ranking: List[str], model names in rank order
        - selection_rationale: str, detailed explanation of selection
    """
    if not isinstance(results, dict) or not results:
        raise ValueError("Cannot compare empty or non-dictionary results.")

    rows: List[Dict[str, Any]] = []

    for model_name, res in results.items():
        agg = res.get("aggregate_metrics", {})
        display_name = res.get("display_name", model_name)
        family = res.get("family", "Unknown")

        top1_mean = agg.get("top1_accuracy_mean", 0.0)
        top1_std = agg.get("top1_accuracy_std", 0.0)
        logloss_mean = agg.get("log_loss_mean", float("nan"))
        logloss_std = agg.get("log_loss_std", float("nan"))
        macro_f1_mean = agg.get("macro_f1_mean", 0.0)
        macro_f1_std = agg.get("macro_f1_std", 0.0)
        cand_acc_mean = agg.get("accuracy_mean", 0.0)
        cand_rec_mean = agg.get("recall_mean", 0.0)
        mean_inf_time = agg.get("mean_inference_time_s", 0.0)
        mean_trn_time = agg.get("mean_train_time_s", 0.0)

        rows.append(
            {
                "model_name": model_name,
                "display_name": display_name,
                "family": family,
                "top1_accuracy_mean": top1_mean,
                "top1_accuracy_std": top1_std,
                "log_loss_mean": logloss_mean,
                "log_loss_std": logloss_std,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_std": macro_f1_std,
                "candidate_accuracy_mean": cand_acc_mean,
                "candidate_recall_mean": cand_rec_mean,
                "mean_inference_time_s": mean_inf_time,
                "mean_train_time_s": mean_trn_time,
            }
        )

    df = pd.DataFrame(rows)

    def sort_key(row):
        top1 = row["top1_accuracy_mean"]
        loss = row["log_loss_mean"]
        if np.isnan(loss):
            loss = float("inf")
        f1 = row["macro_f1_mean"]
        stability = row["top1_accuracy_std"]
        inf_time = row["mean_inference_time_s"]
        return (-top1, loss, -f1, stability, inf_time)

    sorted_indices = sorted(range(len(df)), key=lambda i: sort_key(df.iloc[i]))
    df = df.iloc[sorted_indices].reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    best_row = df.iloc[0]
    best_model_name = str(best_row["model_name"])
    ranking = df["model_name"].tolist()

    best_top1 = best_row["top1_accuracy_mean"]
    best_std = best_row["top1_accuracy_std"]
    best_loss = best_row["log_loss_mean"]
    best_f1 = best_row["macro_f1_mean"]
    best_inf = best_row["mean_inference_time_s"]

    rationale = (
        f"Selected '{best_row['display_name']}' ({best_model_name}) as rank 1.\n"
        f"- Primary Criterion (Mean GroupKFold Top-1 Accuracy): {best_top1:.4f} (std: {best_std:.4f})\n"
        f"- Diagnostic Tiebreaker (Mean Log Loss): {best_loss:.4f}\n"
        f"- Macro F1 Score: {best_f1:.4f}\n"
        f"- Mean Fold Inference Latency: {best_inf * 1000:.2f} ms\n"
        f"Evaluation strictly uses out-of-fold validation sets grouped by trip ID; "
        f"training accuracy was not considered in selection."
    )

    return {
        "comparison_table": df,
        "best_model": best_model_name,
        "best_model_details": best_row.to_dict(),
        "ranking": ranking,
        "selection_rationale": rationale,
    }
