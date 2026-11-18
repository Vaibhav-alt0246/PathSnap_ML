"""Coordinate projection module for PathSnap."""

from member_a.src.projection.projection import (
    get_utm_epsg_for_coords,
    project_coordinates,
    project_trip,
    CoordinateProjector,
)

__all__ = [
    "get_utm_epsg_for_coords",
    "project_coordinates",
    "project_trip",
    "CoordinateProjector",
]
