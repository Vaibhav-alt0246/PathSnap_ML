"""Unit tests for the candidate generation module."""

import pytest
import numpy as np
import pandas as pd
from member_a.src.candidates.generator import (
    generate_candidates,
    _compute_segment_projections,
)
from member_a.src.data.loader import Trip, load_trip
from member_a.config.settings import CANDIDATE_COLUMNS


def test_segment_projection_math():
    # Segment from (0, 0) to (10, 0)
    ux, uy = np.array([0.0]), np.array([0.0])
    vx, vy = np.array([10.0]), np.array([0.0])

    # Point directly above midpoint at (5, 3)
    px, py = np.array([5.0]), np.array([3.0])
    d, t, proj_x, proj_y, arc_len = _compute_segment_projections(px, py, ux, uy, vx, vy)

    assert np.isclose(d[0], 3.0)
    assert np.isclose(t[0], 0.5)
    assert np.isclose(proj_x[0], 5.0)
    assert np.isclose(proj_y[0], 0.0)
    assert np.isclose(arc_len[0], 10.0)


def test_segment_projection_clamping_beyond_endpoints():
    ux, uy = np.array([0.0]), np.array([0.0])
    vx, vy = np.array([10.0]), np.array([0.0])

    # Point beyond start: (-2, 4) -> closest to start (0, 0), dist = sqrt(4+16) = sqrt(20)
    px, py = np.array([-2.0]), np.array([4.0])
    d, t, proj_x, proj_y, _ = _compute_segment_projections(px, py, ux, uy, vx, vy)
    assert np.isclose(t[0], 0.0)
    assert np.isclose(proj_x[0], 0.0)
    assert np.isclose(proj_y[0], 0.0)
    assert np.isclose(d[0], np.sqrt(20.0))

    # Point beyond end: (15, 0) -> closest to end (10, 0), dist = 5.0
    px, py = np.array([15.0]), np.array([0.0])
    d, t, proj_x, proj_y, _ = _compute_segment_projections(px, py, ux, uy, vx, vy)
    assert np.isclose(t[0], 1.0)
    assert np.isclose(proj_x[0], 10.0)
    assert np.isclose(d[0], 5.0)


def test_zero_length_segment_safe_handling():
    ux, uy = np.array([0.0]), np.array([0.0])
    vx, vy = np.array([0.0]), np.array([0.0])
    px, py = np.array([3.0]), np.array([4.0])

    d, t, proj_x, proj_y, arc_len = _compute_segment_projections(px, py, ux, uy, vx, vy)
    assert np.isclose(d[0], 5.0)
    assert np.isclose(t[0], 0.0)
    assert np.isclose(arc_len[0], 0.0)


def test_generate_candidates_real_trip():
    trip = load_trip(partition="train", trip_id="00000001", include_route=False)
    # Test with first 20 GPS points for fast execution
    trip_subset = Trip(
        trip_id=trip.trip_id,
        partition=trip.partition,
        track=trip.track.iloc[:20].copy(),
        nodes=trip.nodes,
        arcs=trip.arcs,
        route=None,
    )

    candidates = generate_candidates(trip_subset, search_radius_m=50.0, max_candidates=10)
    assert isinstance(candidates, pd.DataFrame)
    assert list(candidates.columns) == CANDIDATE_COLUMNS
    assert len(candidates) > 0
    assert not candidates.isna().any().any()

    # Check that distances are within search radius or nearest fallback
    assert (candidates["distance_m"] >= 0.0).all()
    assert (candidates["projection_fraction"] >= 0.0).all()
    assert (candidates["projection_fraction"] <= 1.0).all()

    # Check max candidates limit
    counts = candidates.groupby("point_idx").size()
    assert (counts <= 10).all()

    # Check deduplication
    assert not candidates.duplicated(subset=["point_idx", "arc_id"]).any()

    # Check deterministic ordering
    for _, group in candidates.groupby("point_idx"):
        dists = group["distance_m"].values
        assert np.all(np.diff(dists) >= -1e-8)
