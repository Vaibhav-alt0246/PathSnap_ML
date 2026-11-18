"""Executable Five-Model Baseline GroupKFold Experiment Pipeline for PathSnap ML.

Prepares and executes the 5-model baseline comparison across:
1. Logistic Regression (balanced class weights)
2. Gaussian Naive Bayes (balanced uniform priors)
3. K-Nearest Neighbors (distance-weighted spatial voting)
4. Random Forest (balanced subsample tree weighting)
5. HistGradientBoosting (balanced class weighting)

Features:
- Validates candidate table schema, types, class distribution, and data integrity.
- Strictly excludes all forbidden metadata, coordinates, and route leakage.
- Enforces GroupKFold(n_splits=5) trip-level isolation with zero train/val overlap.
- Computes per-trip candidate recall diagnostics, isolating outliers like trip 00000054.
- Generates out-of-fold predictions, fold metrics, and overall comparison rankings.
- Exports structured results (CSV, JSON metadata) and trained deployment model (.joblib).
- Supports CLI execution and YAML/JSON configuration without modifying source code.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import yaml

from src.evaluation.metrics import compute_top1_metrics
from src.models.registry import MODEL_REGISTRY
from src.models.training import (
    DEFAULT_FORBIDDEN_COLUMNS,
    compare_models,
    select_feature_columns,
    train_models,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pathsnap.experiment")

FIVE_BASELINE_MODELS = [
    "logistic_regression",
    "gaussian_nb",
    "knn",
    "random_forest",
    "hist_gradient_boosting",
]


def load_candidate_table(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load candidate feature table from CSV or Parquet file.

    Parameters
    ----------
    data_path : str or Path
        Path to input dataset file (.csv or .parquet).

    Returns
    -------
    pd.DataFrame
        Loaded candidate table.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        logger.info(f"Loading Parquet candidate table from {path}...")
        return pd.read_parquet(path)
    elif suffix in [".csv", ".txt"]:
        logger.info(f"Loading CSV candidate table from {path}...")
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported formats are .parquet and .csv."
        )


def validate_candidate_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "label",
    trip_id_col: str = "trip_id",
    obs_id_cols: Sequence[str] = ("trip_id", "point_idx"),
    min_trips: int = 5,
) -> Dict[str, Any]:
    """Validate candidate feature table integrity before model training.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded candidate feature table.
    feature_cols : Sequence[str]
        Explicit whitelist of feature column names provided by Member A.
    target_col : str
        Ground-truth binary label column.
    trip_id_col : str
        Trip grouping identifier column.
    obs_id_cols : Sequence[str]
        Columns identifying unique GPS observations.
    min_trips : int
        Minimum number of distinct trips required for GroupKFold.

    Returns
    -------
    dict
        Validation summary report containing integrity statistics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    if len(df) == 0:
        raise ValueError("Candidate feature table is empty.")

    # 1. Verify required identifier & target columns
    required_cols = [target_col, trip_id_col] + list(obs_id_cols)
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise KeyError(f"Missing required columns in candidate table: {missing_required}")

    # 2. Verify explicit feature columns are supplied
    if not feature_cols:
        raise ValueError("feature_cols must be explicitly supplied (cannot be empty).")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise KeyError(f"Specified feature columns not found in candidate table: {missing_features}")

    # 3. Verify feature columns are strictly numeric and contain no forbidden leakage columns
    forbidden_set = {c.lower() for c in DEFAULT_FORBIDDEN_COLUMNS}
    for col in feature_cols:
        if col.lower() in forbidden_set:
            raise ValueError(
                f"Feature '{col}' is in the forbidden leakage list and must never be used as a model feature."
            )
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Feature column '{col}' is not numeric (dtype: {df[col].dtype}). Features must be numeric."
            )

    # 4. Verify target contains both classes
    unique_labels = np.unique(df[target_col].dropna())
    if not np.array_equal(np.sort(unique_labels), np.array([0, 1])):
        raise ValueError(
            f"Target column '{target_col}' must contain binary classes {{0, 1}}, got {unique_labels.tolist()}."
        )

    # 5. Check distinct trips
    unique_trips = np.unique(df[trip_id_col])
    if len(unique_trips) < min_trips:
        raise ValueError(
            f"Candidate table has only {len(unique_trips)} distinct trips; "
            f"at least {min_trips} trips required for {min_trips}-fold GroupKFold."
        )

    # 6. Check missing and infinite values
    feature_nans = int(df[list(feature_cols)].isna().sum().sum())
    feature_infs = int(np.isinf(df[list(feature_cols)].select_dtypes(include=[np.number])).sum().sum())
    target_nans = int(df[target_col].isna().sum())
    if target_nans > 0:
        raise ValueError(f"Target column '{target_col}' contains {target_nans} missing values.")

    # 7. Check candidate row duplicates
    duplicate_candidates = 0
    if "arc_id" in df.columns:
        dup_subset = list(obs_id_cols) + ["arc_id"]
        duplicate_candidates = int(df.duplicated(subset=dup_subset).sum())

    total_rows = len(df)
    total_obs = df[list(obs_id_cols)].drop_duplicates().shape[0]
    pos_count = int((df[target_col] == 1).sum())
    neg_count = int((df[target_col] == 0).sum())

    report = {
        "total_rows": total_rows,
        "total_trips": len(unique_trips),
        "total_observations": total_obs,
        "positive_labels": pos_count,
        "negative_labels": neg_count,
        "class_ratio": f"1 : {neg_count / max(pos_count, 1):.2f}",
        "feature_count": len(feature_cols),
        "feature_columns": list(feature_cols),
        "feature_missing_values": feature_nans,
        "feature_infinite_values": feature_infs,
        "duplicate_candidate_arcs": duplicate_candidates,
        "distinct_trip_ids": [str(t) for t in sorted(unique_trips)],
    }

    logger.info(
        f"Validation PASSED: {total_rows} candidate rows across {len(unique_trips)} trips, "
        f"{total_obs} observations. Imbalance {report['class_ratio']}."
    )
    if feature_nans > 0 or feature_infs > 0:
        logger.warning(
            f"Feature matrix contains {feature_nans} NaNs and {feature_infs} Inf values. "
            "Imputation inside training folds will be applied."
        )

    return report


def compute_candidate_recall_diagnostics(
    df: pd.DataFrame,
    trip_id_col: str = "trip_id",
    obs_id_cols: Sequence[str] = ("trip_id", "point_idx"),
    target_col: str = "label",
) -> pd.DataFrame:
    """Compute per-trip candidate recall statistics to isolate outliers like trip 00000054.

    For each trip:
    - Counts total candidate rows and total GPS observations.
    - Counts observations where NO candidate has label == 1 (candidate-generation miss).
    - Computes candidate recall (fraction of observations where true arc is recalled).
    - Computes candidate miss rate.
    - Flags trips with unusually low recall (< 0.70) or explicitly named '00000054'.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate feature table.
    trip_id_col : str
        Trip grouping column.
    obs_id_cols : Sequence[str]
        Observation identifier columns.
    target_col : str
        Ground truth binary label column.

    Returns
    -------
    pd.DataFrame
        Per-trip diagnostic table sorted by candidate miss rate (descending).
    """
    records = []
    obs_cols = list(obs_id_cols)

    for trip_id, trip_df in df.groupby(trip_id_col, sort=False):
        n_candidates = len(trip_df)
        obs_groups = trip_df.groupby(obs_cols, sort=False)
        n_obs = len(obs_groups)

        misses = 0
        recalled = 0
        for _, obs_df in obs_groups:
            has_pos = np.any(obs_df[target_col].values == 1)
            if has_pos:
                recalled += 1
            else:
                misses += 1

        miss_rate = float(misses / n_obs) if n_obs > 0 else 0.0
        recall = float(recalled / n_obs) if n_obs > 0 else 0.0
        avg_cands = float(n_candidates / n_obs) if n_obs > 0 else 0.0

        # Outlier flag: specifically inspect trip 00000054 or severe miss rate (> 30%)
        trip_str = str(trip_id).strip()
        is_trip_54 = trip_str in ["00000054", "54", "trip_00000054"]
        is_outlier = is_trip_54 or (miss_rate > 0.30)

        notes = []
        if is_trip_54:
            notes.append("Target Outlier (trip 00000054)")
        if miss_rate > 0.30:
            notes.append("High Candidate Miss Rate (>30%)")
        if not notes:
            notes.append("Nominal")

        records.append(
            {
                "trip_id": str(trip_id),
                "num_candidates": n_candidates,
                "num_observations": n_obs,
                "evaluable_observations": recalled,
                "candidate_misses": misses,
                "candidate_recall": round(recall, 4),
                "candidate_miss_rate": round(miss_rate, 4),
                "avg_candidates_per_obs": round(avg_cands, 2),
                "is_outlier": is_outlier,
                "diagnostic_notes": "; ".join(notes),
            }
        )

    diag_df = pd.DataFrame(records)
    # Sort by candidate miss rate descending (worst recall first)
    diag_df = diag_df.sort_values(by="candidate_miss_rate", ascending=False).reset_index(drop=True)
    return diag_df


def run_baseline_experiment(
    data_path: Union[str, Path],
    feature_cols: Sequence[str],
    target_col: str = "label",
    trip_id_col: str = "trip_id",
    obs_id_cols: Sequence[str] = ("trip_id", "point_idx"),
    output_dir: Union[str, Path] = "results",
    artifacts_dir: Union[str, Path] = "artifacts",
    n_splits: int = 5,
    random_seed: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Execute the full 5-model baseline GroupKFold cross-validation comparison.

    Parameters
    ----------
    data_path : str or Path
        Path to candidate table (.csv or .parquet).
    feature_cols : Sequence[str]
        Explicit whitelist of feature columns confirmed by Member A.
    target_col : str
        Binary target column name. Defaults to 'label'.
    trip_id_col : str
        Trip grouping column name. Defaults to 'trip_id'.
    obs_id_cols : Sequence[str]
        Observation identifier columns. Defaults to ('trip_id', 'point_idx').
    output_dir : str or Path
        Directory where comparison CSVs and metadata will be written. Defaults to 'results'.
    artifacts_dir : str or Path
        Directory where winning trained model will be exported. Defaults to 'artifacts'.
    n_splits : int
        Number of GroupKFold cross-validation splits. Defaults to 5.
    random_seed : int
        Random seed for reproducibility. Defaults to 42.
    save_artifacts : bool
        Whether to save CSV/joblib outputs to disk. Defaults to True.

    Returns
    -------
    dict
        Comprehensive experiment results including comparison table, fold metrics,
        diagnostics, and best model name.
    """
    out_path = Path(output_dir)
    art_path = Path(artifacts_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    art_path.mkdir(parents=True, exist_ok=True)

    # 1. Load candidate table
    df = load_candidate_table(data_path)

    # 2. Validate candidate dataset
    val_report = validate_candidate_dataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        trip_id_col=trip_id_col,
        obs_id_cols=obs_id_cols,
        min_trips=n_splits,
    )

    # 3. Compute candidate recall diagnostics (specifically tracking trip 00000054)
    logger.info("Computing per-trip candidate recall diagnostics...")
    diag_df = compute_candidate_recall_diagnostics(
        df=df,
        trip_id_col=trip_id_col,
        obs_id_cols=obs_id_cols,
        target_col=target_col,
    )

    # 4. Precompute and verify GroupKFold splits
    logger.info(f"Generating {n_splits}-fold GroupKFold splits grouped by '{trip_id_col}'...")
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[trip_id_col].values
    X_dummy = np.zeros(len(df))
    y = df[target_col].values.astype(int)

    fold_assignments = np.full(len(df), -1, dtype=int)
    splits = list(gkf.split(X_dummy, y, groups=groups))

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        tr_trips = set(groups[tr_idx])
        val_trips = set(groups[val_idx])

        # Strict GroupKFold isolation verification
        overlap = tr_trips.intersection(val_trips)
        if overlap:
            raise RuntimeError(
                f"Group leakage detected in fold {fold_idx}: trips {overlap} present in both train and validation!"
            )
        if not tr_trips.isdisjoint(val_trips):
            raise RuntimeError(f"Fold {fold_idx} failed train_trip_ids.isdisjoint(validation_trip_ids) check.")

        fold_assignments[val_idx] = fold_idx

    # Ensure every single row received an assigned fold
    if np.any(fold_assignments == -1):
        raise RuntimeError("Some candidate rows did not receive a validation fold assignment.")

    # Build fold assignments DataFrame
    assign_cols = [c for c in [trip_id_col, "point_idx", "arc_id"] if c in df.columns]
    assignments_df = df[assign_cols].copy()
    assignments_df.insert(0, "row_index", df.index)
    assignments_df["validation_fold"] = fold_assignments

    # 5. Execute 5-model training across identical folds
    logger.info("Training and evaluating 5 baseline models...")
    t0_all = time.perf_counter()
    train_results = train_models(
        feature_table=df,
        groups=trip_id_col,
        feature_cols=feature_cols,
        target_col=target_col,
        obs_id_cols=obs_id_cols,
        n_splits=n_splits,
        random_seed=random_seed,
        models_to_train=FIVE_BASELINE_MODELS,
    )
    total_experiment_time = time.perf_counter() - t0_all
    logger.info(f"Cross-validation completed in {total_experiment_time:.2f}s.")

    # 6. Compare and rank models
    comp_results = compare_models(train_results)
    comparison_table = comp_results["comparison_table"]
    best_model_name = comp_results["best_model"]
    best_model_details = comp_results["best_model_details"]
    rationale = comp_results["selection_rationale"]

    logger.info(f"Model Comparison Completed. Rank 1: '{best_model_name}'.")

    # 7. Collect per-fold metrics across all models
    fold_metric_rows = []
    for model_name, m_res in train_results.items():
        for f in m_res["fold_metrics"]:
            cm = f["candidate_metrics"]
            top1 = f["top1_metrics"]
            fold_metric_rows.append(
                {
                    "model_name": model_name,
                    "display_name": m_res["display_name"],
                    "fold": f["fold"],
                    "train_trips": f["train_trips"],
                    "val_trips": f["val_trips"],
                    "train_samples": f["train_samples"],
                    "val_samples": f["val_samples"],
                    "top1_accuracy": round(top1["top1_accuracy"], 4),
                    "evaluable_observations": top1["evaluable_observations"],
                    "candidate_misses": top1["candidate_misses"],
                    "candidate_miss_rate": round(top1["candidate_miss_rate"], 4),
                    "candidate_accuracy": round(cm["accuracy"], 4),
                    "precision": round(cm["precision"], 4),
                    "recall": round(cm["recall"], 4),
                    "macro_f1": round(cm["macro_f1"], 4),
                    "log_loss": round(cm["log_loss"], 4),
                    "train_time_s": round(f["train_time_s"], 4),
                    "inference_time_s": round(f["inference_time_s"], 4),
                }
            )
    fold_metrics_df = pd.DataFrame(fold_metric_rows)

    # 8. Retrain the winning baseline model on all training data for export
    # Note: Keep fold-trained models distinct from this final full-dataset artifact
    logger.info(f"Retraining winning model '{best_model_name}' on all training data for deployment...")
    winning_spec = MODEL_REGISTRY[best_model_name]
    deployment_pipeline = winning_spec.factory(random_seed)
    X_full = df[list(feature_cols)].values
    y_full = df[target_col].values.astype(int)

    t_fit_start = time.perf_counter()
    if winning_spec.requires_sample_weights:
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y_full)
        deployment_pipeline.fit(X_full, y_full, classifier__sample_weight=sw)
    else:
        deployment_pipeline.fit(X_full, y_full)
    fit_duration = time.perf_counter() - t_fit_start

    # Build comprehensive metadata
    metadata = {
        "dataset_path": str(data_path),
        "dataset_validation": val_report,
        "feature_columns": list(feature_cols),
        "target_column": target_col,
        "trip_id_column": trip_id_col,
        "observation_columns": list(obs_id_cols),
        "random_seed": random_seed,
        "cross_validation": {
            "strategy": "GroupKFold",
            "n_splits": n_splits,
            "groups": trip_id_col,
            "verified_disjoint_trips": True,
        },
        "models_evaluated": FIVE_BASELINE_MODELS,
        "best_model": {
            "name": best_model_name,
            "display_name": winning_spec.display_name,
            "family": winning_spec.family,
            "imbalance_strategy": winning_spec.imbalance_strategy,
            "mean_top1_accuracy": round(best_model_details["top1_accuracy_mean"], 4),
            "top1_std": round(best_model_details["top1_accuracy_std"], 4),
            "mean_log_loss": round(best_model_details["log_loss_mean"], 4),
            "mean_macro_f1": round(best_model_details["macro_f1_mean"], 4),
            "selection_rationale": rationale,
        },
        "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_experiment_time_s": round(total_experiment_time, 2),
    }

    # 9. Save outputs to disk if requested
    if save_artifacts:
        comp_csv = out_path / "baseline_model_comparison.csv"
        comparison_table.to_csv(comp_csv, index=False)
        logger.info(f"Saved comparison table to {comp_csv}")

        fold_csv = out_path / "baseline_fold_metrics.csv"
        fold_metrics_df.to_csv(fold_csv, index=False)
        logger.info(f"Saved fold metrics to {fold_csv}")

        assign_csv = out_path / "baseline_fold_assignments.csv"
        assignments_df.to_csv(assign_csv, index=False)
        logger.info(f"Saved fold assignments to {assign_csv}")

        diag_csv = out_path / "candidate_recall_diagnostics.csv"
        diag_df.to_csv(diag_csv, index=False)
        logger.info(f"Saved candidate recall diagnostics to {diag_csv}")

        meta_json = out_path / "baseline_metadata.json"
        with meta_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved experiment metadata to {meta_json}")

        # Save trained artifact (pipeline + metadata wrapper)
        model_artifact_file = art_path / "best_baseline_model.joblib"
        artifact_bundle = {
            "model_pipeline": deployment_pipeline,
            "model_name": best_model_name,
            "display_name": winning_spec.display_name,
            "feature_columns": list(feature_cols),
            "random_seed": random_seed,
            "training_date": metadata["experiment_timestamp"],
            "training_samples": len(df),
            "training_trips": len(np.unique(groups)),
            "full_fit_time_s": round(fit_duration, 4),
            "cv_top1_mean": round(best_model_details["top1_accuracy_mean"], 4),
            "cv_log_loss_mean": round(best_model_details["log_loss_mean"], 4),
        }
        joblib.dump(artifact_bundle, model_artifact_file)
        logger.info(f"Saved best baseline model artifact to {model_artifact_file}")

    return {
        "comparison_table": comparison_table,
        "fold_metrics": fold_metrics_df,
        "fold_assignments": assignments_df,
        "recall_diagnostics": diag_df,
        "metadata": metadata,
        "best_model_name": best_model_name,
        "best_model_artifact": deployment_pipeline,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="PathSnap Member B — Five-Model Baseline GroupKFold Experiment Runner"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to candidate feature table file (.csv or .parquet).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of confirmed feature column names.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="label",
        help="Target label column name (default: label).",
    )
    parser.add_argument(
        "--trip-col",
        type=str,
        default="trip_id",
        help="Trip ID grouping column name (default: trip_id).",
    )
    parser.add_argument(
        "--obs-cols",
        type=str,
        default="trip_id,point_idx",
        help="Comma-separated observation grouping columns (default: trip_id,point_idx).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save result CSVs and metadata (default: results).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to save trained model artifact (default: artifacts).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits (default: 5).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML experiment configuration file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load from YAML config if supplied
    data_path = args.data_path
    features = [f.strip() for f in args.features.split(",")] if args.features else []
    target = args.target
    trip_col = args.trip_col
    obs_cols = [c.strip() for c in args.obs_cols.split(",")]
    out_dir = args.output_dir
    art_dir = args.artifacts_dir
    n_splits = args.n_splits
    seed = args.random_seed

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        data_path = cfg.get("data_path", data_path)
        features = cfg.get("features", features)
        target = cfg.get("target_column", target)
        trip_col = cfg.get("trip_column", trip_col)
        obs_cols = cfg.get("observation_columns", obs_cols)
        out_dir = cfg.get("output_dir", out_dir)
        art_dir = cfg.get("artifacts_dir", art_dir)
        n_splits = cfg.get("n_splits", n_splits)
        seed = cfg.get("random_seed", seed)

    if not data_path:
        raise ValueError(
            "Missing --data-path (or data_path in --config). "
            "Please provide the path to Member A's verified candidate table."
        )
    if not features:
        raise ValueError(
            "Missing --features (or features list in --config). "
            "Please provide Member A's confirmed feature column list."
        )

    logger.info("=" * 70)
    logger.info("PATHSNAP — MEMBER B FIVE-MODEL BASELINE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Features ({len(features)}): {features}")
    logger.info(f"Target: {target} | Trip Col: {trip_col} | Obs Cols: {obs_cols}")
    logger.info(f"GroupKFold splits: {n_splits} | Seed: {seed}")
    logger.info("=" * 70)

    run_baseline_experiment(
        data_path=data_path,
        feature_cols=features,
        target_col=target,
        trip_id_col=trip_col,
        obs_id_cols=obs_cols,
        output_dir=out_dir,
        artifacts_dir=art_dir,
        n_splits=n_splits,
        random_seed=seed,
        save_artifacts=True,
    )
    logger.info("Experiment pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
