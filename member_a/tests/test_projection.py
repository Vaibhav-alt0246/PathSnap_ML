"""Unit tests for the coordinate projection module."""

import pytest
import numpy as np
import pandas as pd
from member_a.src.projection.projection import (
    get_utm_epsg_for_coords,
    CoordinateProjector,
    project_coordinates,
    project_trip,
)
from member_a.src.data.loader import load_trip
from member_a.src.data.preprocessing import haversine_distance


def test_utm_zone_calculation():
    # Germany (Darmstadt): lon 8.59, lat 49.84 -> Zone 32N -> EPSG 32632
    assert get_utm_epsg_for_coords(8.59, 49.84) == 32632

    # New York: lon -74.0, lat 40.71 -> Zone 18N -> EPSG 32618
    assert get_utm_epsg_for_coords(-74.0, 40.71) == 32618

    # Sydney, Australia: lon 151.2, lat -33.86 -> Zone 56S -> EPSG 32756
    assert get_utm_epsg_for_coords(151.2, -33.86) == 32756


def test_project_coordinates_forward_and_inverse():
    epsg = 32632
    projector = CoordinateProjector(epsg)

    orig_lons = np.array([8.595, 8.596, 8.597])
    orig_lats = np.array([49.844, 49.845, 49.846])

    x, y = projector.forward(orig_lons, orig_lats)
    assert len(x) == 3
    assert len(y) == 3
    assert not np.isnan(x).any()
    assert not np.isnan(y).any()

    # Inverse transformation
    recov_lons, recov_lats = projector.inverse(x, y)
    np.testing.assert_allclose(recov_lons, orig_lons, atol=1e-6)
    np.testing.assert_allclose(recov_lats, orig_lats, atol=1e-6)


def test_distance_metric_consistency():
    """Verify that projected Euclidean distance corresponds to physical distance in metres."""
    epsg = 32632
    projector = CoordinateProjector(epsg)

    lon1, lat1 = 8.595, 49.844
    lon2, lat2 = 8.596, 49.845

    x1, y1 = projector.forward(lon1, lat1)
    x2, y2 = projector.forward(lon2, lat2)

    euclidean_m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    haversine_m = haversine_distance(np.array([lon1]), np.array([lat1]), np.array([lon2]), np.array([lat2]))[0]

    # UTM local conformal distortion is typically < 0.1%
    rel_error = abs(euclidean_m - haversine_m) / haversine_m
    assert rel_error < 0.002, f"Relative error {rel_error} exceeds 0.2%"


def test_project_trip_real_trip():
    trip = load_trip(partition="train", trip_id="00000001", include_route=False)
    proj_track, proj_nodes, epsg = project_trip(trip)

    assert epsg == 32632
    assert "x_m" in proj_track.columns
    assert "y_m" in proj_track.columns
    assert "x_m" in proj_nodes.columns
    assert "y_m" in proj_nodes.columns

    assert len(proj_track) == trip.num_gps
    assert len(proj_nodes) == trip.num_nodes
    assert not proj_track[["x_m", "y_m"]].isna().any().any()
    assert not proj_nodes[["x_m", "y_m"]].isna().any().any()
