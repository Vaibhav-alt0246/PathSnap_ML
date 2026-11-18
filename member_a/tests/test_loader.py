"""Unit tests for the PathSnap data loader."""

import pytest
import pandas as pd
from member_a.src.data.loader import load_trip, Trip
from member_a.config.settings import DEFAULT_DATA_ROOT


def test_load_train_trip_without_route():
    trip = load_trip(data_root=DEFAULT_DATA_ROOT, partition="train", trip_id="00000001", include_route=False)
    assert isinstance(trip, Trip)
    assert trip.trip_id == "00000001"
    assert trip.partition == "train"
    assert trip.num_gps > 0
    assert trip.num_nodes > 0
    assert trip.num_arcs > 0
    assert trip.route is None
    assert not trip.has_route
    assert list(trip.track.columns) == ["lon", "lat", "timestamp"]
    assert list(trip.nodes.columns) == ["lon", "lat"]
    assert list(trip.arcs.columns) == ["u", "v"]


def test_load_train_trip_with_route():
    trip = load_trip(data_root=DEFAULT_DATA_ROOT, partition="train", trip_id="00000001", include_route=True)
    assert trip.has_route
    assert trip.route is not None
    assert len(trip.route) > 0
    assert list(trip.route.columns) == ["arc_id"]


def test_load_student_test_trip_without_route():
    trip = load_trip(data_root=DEFAULT_DATA_ROOT, partition="student_test", trip_id="00000000", include_route=False)
    assert isinstance(trip, Trip)
    assert trip.partition == "student_test"
    assert trip.num_gps > 0
    assert trip.route is None


def test_student_test_route_protection():
    """Verify that attempting to load student_test route raises PermissionError."""
    with pytest.raises(PermissionError) as exc_info:
        load_trip(data_root=DEFAULT_DATA_ROOT, partition="student_test", trip_id="00000000", include_route=True)
    assert "Access Denied" in str(exc_info.value)


def test_nonexistent_trip_raises_error():
    with pytest.raises(FileNotFoundError):
        load_trip(data_root=DEFAULT_DATA_ROOT, partition="train", trip_id="99999999")


def test_invalid_partition_raises_error():
    with pytest.raises(ValueError):
        load_trip(data_root=DEFAULT_DATA_ROOT, partition="invalid_partition", trip_id="00000001")


def test_zero_based_indexing_preserved():
    trip = load_trip(data_root=DEFAULT_DATA_ROOT, partition="train", trip_id="00000001")
    assert trip.nodes.index.start == 0
    assert trip.nodes.index[0] == 0
    assert trip.arcs.index.start == 0
    assert trip.arcs.index[0] == 0
    assert trip.arcs["u"].min() >= 0
    assert trip.arcs["u"].max() < len(trip.nodes)
    assert trip.arcs["v"].min() >= 0
    assert trip.arcs["v"].max() < len(trip.nodes)
