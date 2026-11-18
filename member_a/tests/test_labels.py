"""Unit tests for the training label alignment module."""

import pytest
import numpy as np
import pandas as pd
from member_a.src.labels.alignment import align_route_labels, compute_monotonic_route_alignment
from member_a.src.candidates.generator import generate_candidates
from member_a.src.data.loader import Trip, load_trip
from member_a.config.settings import TARGET_COLUMN


def test_align_route_labels_real_trip():
    trip = load_trip(partition="train", trip_id="00000001", include_route=True)
    trip_subset = Trip(
        trip_id=trip.trip_id,
        partition=trip.partition,
        track=trip.track.iloc[:30].copy(),
        nodes=trip.nodes,
        arcs=trip.arcs,
        route=trip.route,
    )

    candidates = generate_candidates(trip_subset, search_radius_m=50.0, max_candidates=10)
    labeled = align_route_labels(trip_subset, candidates)

    assert TARGET_COLUMN in labeled.columns
    assert len(labeled) == len(candidates)
    # Check binary values
    assert set(labeled[TARGET_COLUMN].unique()).issubset({0, 1})

    # Preserves candidate row order
    pd.testing.assert_frame_equal(
        candidates.reset_index(drop=True),
        labeled.drop(columns=[TARGET_COLUMN]).reset_index(drop=True),
    )

    # Positive count check: each covered GPS observation has at most 1 positive label
    positives = labeled[labeled[TARGET_COLUMN] == 1]
    assert len(positives) > 0
    assert not positives.duplicated(subset=["point_idx"]).any()


def test_student_test_label_alignment_blocked():
    """Verify that calling align_route_labels on student_test raises PermissionError."""
    trip = load_trip(partition="student_test", trip_id="00000000", include_route=False)
    candidates = pd.DataFrame({"point_idx": [0], "arc_id": [1]})
    with pytest.raises(PermissionError) as exc_info:
        align_route_labels(trip, candidates)
    assert "Access Denied" in str(exc_info.value)


def test_missing_route_raises_value_error():
    trip = load_trip(partition="train", trip_id="00000001", include_route=False)
    candidates = pd.DataFrame({"point_idx": [0], "arc_id": [1]})
    with pytest.raises(ValueError):
        align_route_labels(trip, candidates)
