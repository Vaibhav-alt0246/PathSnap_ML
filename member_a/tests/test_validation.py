"""Unit tests for the PathSnap validation module."""

import pytest
import pandas as pd
import numpy as np
from member_a.src.data.validation import validate_trip_data, ValidationError


@pytest.fixture
def sample_valid_data():
    track_df = pd.DataFrame({
        "lon": [8.595, 8.596, 8.597],
        "lat": [49.844, 49.845, 49.846],
        "timestamp": [1.0, 2.0, 3.0],
    })
    nodes_df = pd.DataFrame({
        "lon": [8.590, 8.595, 8.600],
        "lat": [49.840, 49.845, 49.850],
    })
    arcs_df = pd.DataFrame({
        "u": [0, 1],
        "v": [1, 2],
    })
    route_df = pd.DataFrame({
        "arc_id": [0, 1],
    })
    return track_df, nodes_df, arcs_df, route_df


def test_valid_data_passes(sample_valid_data):
    track, nodes, arcs, route = sample_valid_data
    result = validate_trip_data(track, nodes, arcs, route, trip_id="test")
    assert result["is_valid"] is True
    assert result["num_gps"] == 3
    assert result["num_nodes"] == 3
    assert result["num_arcs"] == 2
    assert result["num_route_arcs"] == 2


def test_empty_track_raises(sample_valid_data):
    _, nodes, arcs, route = sample_valid_data
    empty_track = pd.DataFrame(columns=["lon", "lat", "timestamp"])
    with pytest.raises(ValidationError):
        validate_trip_data(empty_track, nodes, arcs, route)


def test_nan_values_raise(sample_valid_data):
    track, nodes, arcs, route = sample_valid_data
    track_nan = track.copy()
    track_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValidationError):
        validate_trip_data(track_nan, nodes, arcs, route)


def test_invalid_node_index_in_arcs_raises(sample_valid_data):
    track, nodes, arcs, route = sample_valid_data
    arcs_invalid = arcs.copy()
    arcs_invalid.iloc[0, 0] = 999  # Node 999 does not exist (num_nodes=3)
    with pytest.raises(ValidationError):
        validate_trip_data(track, nodes, arcs_invalid, route)


def test_non_monotonic_timestamp_raises(sample_valid_data):
    track, nodes, arcs, route = sample_valid_data
    track_non_mono = track.copy()
    track_non_mono["timestamp"] = [3.0, 2.0, 1.0]
    with pytest.raises(ValidationError):
        validate_trip_data(track_non_mono, nodes, arcs, route)


def test_invalid_route_arc_raises(sample_valid_data):
    track, nodes, arcs, route = sample_valid_data
    route_invalid = route.copy()
    route_invalid.iloc[0, 0] = 50  # Arc 50 does not exist (num_arcs=2)
    with pytest.raises(ValidationError):
        validate_trip_data(track, nodes, arcs, route_invalid)
