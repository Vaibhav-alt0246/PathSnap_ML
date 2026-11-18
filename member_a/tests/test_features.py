"""Unit tests for the feature engineering module."""

import pytest
import numpy as np
import pandas as pd
from member_a.src.candidates.generator import generate_candidates
from member_a.src.labels.alignment import align_route_labels
from member_a.src.features.engineering import build_features
from member_a.src.features.schema import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    split_features_and_identifiers,
)
from member_a.src.data.loader import Trip, load_trip


@pytest.fixture
def sample_candidates_and_trip():
    trip = load_trip(partition="train", trip_id="00000001", include_route=True)
    trip_subset = Trip(
        trip_id=trip.trip_id,
        partition=trip.partition,
        track=trip.track.iloc[:35].copy(),
        nodes=trip.nodes,
        arcs=trip.arcs,
        route=trip.route,
    )
    candidates = generate_candidates(trip_subset, search_radius_m=50.0, max_candidates=10)
    return trip_subset, candidates


def test_build_features_output_schema(sample_candidates_and_trip):
    trip, candidates = sample_candidates_and_trip
    feat_df = build_features(trip, candidates)

    assert isinstance(feat_df, pd.DataFrame)
    assert len(feat_df) == len(candidates)

    # Check all feature columns exist
    for feat in FEATURE_COLUMNS:
        assert feat in feat_df.columns, f"Missing feature: {feat}"

    # Check that candidate base columns are preserved
    for col in candidates.columns:
        assert col in feat_df.columns


def test_no_nan_or_inf_in_features(sample_candidates_and_trip):
    trip, candidates = sample_candidates_and_trip
    feat_df = build_features(trip, candidates)

    for col in FEATURE_COLUMNS:
        vals = feat_df[col].to_numpy()
        assert not np.isnan(vals).any(), f"Feature {col} contains NaN values."
        assert not np.isinf(vals).any(), f"Feature {col} contains infinite values."


def test_features_work_without_labels_or_route():
    trip = load_trip(partition="student_test", trip_id="00000000", include_route=False)
    trip_subset = Trip(
        trip_id=trip.trip_id,
        partition=trip.partition,
        track=trip.track.iloc[:20].copy(),
        nodes=trip.nodes,
        arcs=trip.arcs,
        route=None,
    )
    candidates = generate_candidates(trip_subset, search_radius_m=50.0, max_candidates=10)
    assert TARGET_COLUMN not in candidates.columns

    feat_df = build_features(trip_subset, candidates)
    assert len(feat_df) == len(candidates)
    assert TARGET_COLUMN not in feat_df.columns
    for feat in FEATURE_COLUMNS:
        assert feat in feat_df.columns


def test_preserves_candidate_row_order(sample_candidates_and_trip):
    trip, candidates = sample_candidates_and_trip
    feat_df = build_features(trip, candidates)

    # Row order check
    np.testing.assert_array_equal(feat_df["point_idx"].values, candidates["point_idx"].values)
    np.testing.assert_array_equal(feat_df["arc_id"].values, candidates["arc_id"].values)
    np.testing.assert_allclose(feat_df["distance_m"].values, candidates["distance_m"].values)


def test_split_features_and_identifiers_prevents_leakage(sample_candidates_and_trip):
    trip, candidates = sample_candidates_and_trip
    labeled = align_route_labels(trip, candidates)
    feat_df = build_features(trip, labeled)

    X, meta, y = split_features_and_identifiers(feat_df)

    # X must contain ONLY FEATURE_COLUMNS
    assert list(X.columns) == FEATURE_COLUMNS
    assert "point_idx" not in X.columns
    assert "arc_id" not in X.columns
    assert "label" not in X.columns
    assert "lat" not in X.columns
    assert "lon" not in X.columns
    assert "trip_id" not in X.columns

    # Meta must contain identifiers
    assert "point_idx" in meta.columns
    assert "arc_id" in meta.columns

    # y must be binary label series
    assert y is not None
    assert set(y.unique()).issubset({0, 1})


def test_feature_value_bounds(sample_candidates_and_trip):
    trip, candidates = sample_candidates_and_trip
    feat_df = build_features(trip, candidates)

    assert (feat_df["heading_bearing_diff_deg"] >= 0.0).all()
    assert (feat_df["heading_bearing_diff_deg"] <= 180.0).all()
    assert (feat_df["heading_bearing_cos_sim"] >= -1.0 - 1e-6).all()
    assert (feat_df["heading_bearing_cos_sim"] <= 1.0 + 1e-6).all()
    assert (feat_df["projection_fraction"] >= 0.0).all()
    assert (feat_df["projection_fraction"] <= 1.0).all()
    assert set(feat_df["is_low_speed"].unique()).issubset({0, 1})
    assert set(feat_df["has_reverse_twin"].unique()).issubset({0, 1})
    assert (feat_df["dist_rank"] >= 0).all()
