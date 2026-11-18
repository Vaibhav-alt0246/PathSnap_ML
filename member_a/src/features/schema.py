"""Feature schema definitions and documentation for PathSnap candidate classification.

Defines all engineered machine learning features, identifier columns, and target labels.
Ensures strict leakage prevention: no absolute coordinates, identifiers, or target
labels are included in the feature set.
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd

# Identifier columns preserved for index mapping and handoff (NOT to be fed into ML models)
IDENTIFIER_COLUMNS: List[str] = [
    "point_idx",
    "arc_id",
]

# Coordinate columns for geometric reference/handoff (excluded from ML feature matrix)
METRIC_COORDINATE_COLUMNS: List[str] = [
    "projected_x",
    "projected_y",
]

# Binary target label column (training only)
TARGET_COLUMN: str = "label"

# Engineered ML Features (pure numerical/boolean features for Member B models)
GEOMETRIC_FEATURES: List[str] = [
    "distance_m",
    "distance_to_closest_cand",
    "distance_ratio_closest",
    "dist_rank",
    "cand_count_for_point",
    "projection_fraction",
    "dist_to_segment_start_m",
    "dist_to_segment_end_m",
    "arc_length_m",
]

DIRECTIONAL_FEATURES: List[str] = [
    "gps_heading_deg",
    "road_bearing_deg",
    "heading_bearing_diff_deg",
    "heading_bearing_cos_sim",
    "is_low_speed",
]

KINEMATIC_FEATURES: List[str] = [
    "gps_delta_t_s",
    "gps_step_distance_m",
    "gps_speed_mps",
    "gps_accel_mps2",
    "speed_ratio_to_mean",
]

TOPOLOGICAL_FEATURES: List[str] = [
    "u_in_degree",
    "u_out_degree",
    "v_in_degree",
    "v_out_degree",
    "has_reverse_twin",
]

FEATURE_COLUMNS: List[str] = (
    GEOMETRIC_FEATURES
    + DIRECTIONAL_FEATURES
    + KINEMATIC_FEATURES
    + TOPOLOGICAL_FEATURES
)


def get_feature_names() -> List[str]:
    """Return the ordered list of feature column names."""
    return list(FEATURE_COLUMNS)


def split_features_and_identifiers(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """Split a processed DataFrame into features (X), identifiers (meta), and optional target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame produced by `build_features` or label alignment.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]
        - X: DataFrame containing only FEATURE_COLUMNS for ML training / inference.
        - meta: DataFrame containing IDENTIFIER_COLUMNS and coordinate columns.
        - y: Series containing binary TARGET_COLUMN if present, else None.
    """
    missing_feats = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_feats:
        raise ValueError(f"DataFrame is missing required features: {missing_feats}")

    X = df[FEATURE_COLUMNS].copy()
    meta_cols = [c for c in IDENTIFIER_COLUMNS + METRIC_COORDINATE_COLUMNS if c in df.columns]
    meta = df[meta_cols].copy()

    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, meta, y


def load_training_dataset(
    path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load the precomputed full training dataset for Member B model training.

    Parameters
    ----------
    path : Optional[Union[str, Path]]
        Path to training_candidates_features.parquet. Defaults to results directory.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]
        - X: Feature matrix containing only the 24 FEATURE_COLUMNS.
        - y: Binary classification target labels (1 = matching arc, 0 = non-matching).
        - groups: Trip IDs for GroupKFold cross-validation.
        - meta: Candidate metadata (trip_id, point_idx, arc_id, projected_x, projected_y, distance_m).
    """
    if path is None:
        target_path = Path(__file__).resolve().parent.parent.parent / "results" / "training_candidates_features.parquet"
    else:
        target_path = Path(path)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {target_path}. "
            f"Run `python member_a/scripts/build_full_training_dataset.py` to generate it."
        )

    df = pd.read_parquet(target_path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    groups = df["trip_id"].copy()
    meta_cols = [c for c in ["trip_id", "point_idx", "arc_id", "projected_x", "projected_y", "distance_m"] if c in df.columns]
    meta = df[meta_cols].copy()
    return X, y, groups, meta
