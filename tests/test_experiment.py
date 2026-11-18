"""Tests for the 5-model baseline experiment pipeline, validation, and diagnostics.

Note: Uses temporary synthetic fixtures strictly for software verification.
Never writes to project results/ or artifacts/ directories.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.models.experiment import (
    FIVE_BASELINE_MODELS,
    compute_candidate_recall_diagnostics,
    load_candidate_table,
    run_baseline_experiment,
    validate_candidate_dataset,
)


@pytest.fixture
def sample_candidate_df():
    """Generates a small synthetic candidate DataFrame across 6 trips.

    Includes trip '00000054' with an artificially induced candidate miss.
    """
    np.random.seed(42)
    rows = []
    trips = ["00000001", "00000002", "00000003", "00000004", "00000005", "00000054"]

    for trip in trips:
        for pt in range(4):
            for cand in range(5):
                # For trip 00000054 point 2, simulate candidate miss (all label 0)
                if trip == "00000054" and pt == 2:
                    is_true = 0
                else:
                    is_true = 1 if cand == 0 else 0

                rows.append(
                    {
                        "trip_id": trip,
                        "point_idx": pt,
                        "arc_id": 1000 + cand,
                        "label": is_true,
                        "distance_m": float(np.random.uniform(1.0, 15.0) if is_true else np.random.uniform(20.0, 80.0)),
                        "projection_fraction": float(np.random.uniform(0.1, 0.9)),
                        "arc_length_m": float(np.random.uniform(50.0, 200.0)),
                        "projected_x": float(77.5 + np.random.uniform(-0.01, 0.01)),
                        "projected_y": float(12.9 + np.random.uniform(-0.01, 0.01)),
                    }
                )
    return pd.DataFrame(rows)


def test_validate_candidate_dataset_success(sample_candidate_df):
    report = validate_candidate_dataset(
        df=sample_candidate_df,
        feature_cols=["distance_m", "projection_fraction", "arc_length_m"],
        target_col="label",
        trip_id_col="trip_id",
        obs_id_cols=["trip_id", "point_idx"],
        min_trips=5,
    )
    assert report["total_trips"] == 6
    assert report["feature_count"] == 3
    assert report["feature_missing_values"] == 0
    assert report["positive_labels"] > 0
    assert report["negative_labels"] > 0


def test_validate_candidate_dataset_rejects_forbidden_feature(sample_candidate_df):
    with pytest.raises(ValueError, match="forbidden"):
        validate_candidate_dataset(
            df=sample_candidate_df,
            feature_cols=["distance_m", "projected_x"],
            target_col="label",
        )

    with pytest.raises(ValueError, match="forbidden"):
        validate_candidate_dataset(
            df=sample_candidate_df,
            feature_cols=["distance_m", "trip_id"],
            target_col="label",
        )

    with pytest.raises(ValueError, match="forbidden"):
        validate_candidate_dataset(
            df=sample_candidate_df,
            feature_cols=["distance_m", "label"],
            target_col="label",
        )


def test_validate_candidate_dataset_rejects_insufficient_trips(sample_candidate_df):
    # Filter to only 3 trips
    df_small = sample_candidate_df[sample_candidate_df["trip_id"].isin(["00000001", "00000002", "00000003"])]
    with pytest.raises(ValueError, match="distinct trips"):
        validate_candidate_dataset(
            df=df_small,
            feature_cols=["distance_m", "projection_fraction"],
            min_trips=5,
        )


def test_validate_candidate_dataset_rejects_monoclass(sample_candidate_df):
    df_all_neg = sample_candidate_df.copy()
    df_all_neg["label"] = 0
    with pytest.raises(ValueError, match="binary classes"):
        validate_candidate_dataset(
            df=df_all_neg,
            feature_cols=["distance_m"],
        )


def test_compute_candidate_recall_diagnostics_isolates_trip_54(sample_candidate_df):
    diag_df = compute_candidate_recall_diagnostics(
        df=sample_candidate_df,
        trip_id_col="trip_id",
        obs_id_cols=["trip_id", "point_idx"],
        target_col="label",
    )

    assert isinstance(diag_df, pd.DataFrame)
    assert len(diag_df) == 6

    # Verify trip 00000054 is flagged
    trip_54_row = diag_df[diag_df["trip_id"] == "00000054"]
    assert len(trip_54_row) == 1
    assert bool(trip_54_row.iloc[0]["is_outlier"]) is True
    assert trip_54_row.iloc[0]["candidate_misses"] == 1
    assert "00000054" in trip_54_row.iloc[0]["diagnostic_notes"]


def test_run_baseline_experiment_pipeline(sample_candidate_df, tmp_path):
    # Save sample DataFrame to temp CSV
    csv_file = tmp_path / "synthetic_candidates.csv"
    sample_candidate_df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "results"
    art_dir = tmp_path / "artifacts"

    res = run_baseline_experiment(
        data_path=csv_file,
        feature_cols=["distance_m", "projection_fraction", "arc_length_m"],
        target_col="label",
        trip_id_col="trip_id",
        obs_id_cols=["trip_id", "point_idx"],
        output_dir=out_dir,
        artifacts_dir=art_dir,
        n_splits=5,
        random_seed=42,
        save_artifacts=True,
    )

    # Verify return dictionary structure
    assert res["best_model_name"] in FIVE_BASELINE_MODELS
    assert len(res["comparison_table"]) == 5
    assert len(res["fold_assignments"]) == len(sample_candidate_df)

    # Verify files created in tmp_path (NOT in project directory!)
    assert (out_dir / "baseline_model_comparison.csv").exists()
    assert (out_dir / "baseline_fold_metrics.csv").exists()
    assert (out_dir / "baseline_fold_assignments.csv").exists()
    assert (out_dir / "baseline_metadata.json").exists()
    assert (out_dir / "candidate_recall_diagnostics.csv").exists()
    assert (art_dir / "best_baseline_model.joblib").exists()

    # Verify fold assignments disjointness
    assign_df = pd.read_csv(out_dir / "baseline_fold_assignments.csv")
    for fold in range(5):
        val_trips = set(assign_df[assign_df["validation_fold"] == fold]["trip_id"])
        train_trips = set(assign_df[assign_df["validation_fold"] != fold]["trip_id"])
        assert train_trips.isdisjoint(val_trips), f"Leakage detected in fold {fold}!"


def test_load_candidate_table_parquet(sample_candidate_df, tmp_path):
    parquet_file = tmp_path / "candidates.parquet"
    sample_candidate_df.to_parquet(parquet_file)

    loaded = load_candidate_table(parquet_file)
    assert len(loaded) == len(sample_candidate_df)
    assert list(loaded.columns) == list(sample_candidate_df.columns)
