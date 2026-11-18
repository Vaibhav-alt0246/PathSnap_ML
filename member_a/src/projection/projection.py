"""Metric coordinate projection system for PathSnap.

Converts angular (longitude, latitude) coordinates (WGS 84 / EPSG:4326) into metric
Universal Transverse Mercator (UTM) coordinates where Euclidean distances represent
true physical distances in metres.
"""

from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd
import pyproj

from member_a.src.data.loader import Trip


def get_utm_epsg_for_coords(lon: float, lat: float) -> int:
    """Calculate the optimal UTM EPSG code for a given longitude and latitude.

    Parameters
    ----------
    lon : float
        Longitude in decimal degrees [-180, 180].
    lat : float
        Latitude in decimal degrees [-90, 90].

    Returns
    -------
    int
        EPSG code (e.g., 32632 for UTM Zone 32N, 32756 for UTM Zone 56S).
    """
    # Normalize longitude to [-180, 180)
    lon_norm = ((lon + 180.0) % 360.0) - 180.0
    zone = int(np.floor((lon_norm + 180.0) / 6.0)) + 1
    if zone > 60:
        zone = 60

    if lat >= 0.0:
        return 32600 + zone
    else:
        return 32700 + zone


class CoordinateProjector:
    """Manages forward and inverse transformations between WGS 84 and metric UTM."""

    def __init__(self, epsg_code: int):
        self.epsg_code = epsg_code
        self.source_crs = pyproj.CRS("EPSG:4326")
        self.target_crs = pyproj.CRS(f"EPSG:{epsg_code}")
        self.transformer = pyproj.Transformer.from_crs(
            self.source_crs, self.target_crs, always_xy=True
        )
        self.inv_transformer = pyproj.Transformer.from_crs(
            self.target_crs, self.source_crs, always_xy=True
        )

    def forward(
        self, lons: Union[float, np.ndarray, pd.Series], lats: Union[float, np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project longitude/latitude into metric easting (x) and northing (y) in metres."""
        lons_arr = np.asarray(lons, dtype=float)
        lats_arr = np.asarray(lats, dtype=float)
        x, y = self.transformer.transform(lons_arr, lats_arr)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def inverse(
        self, x: Union[float, np.ndarray, pd.Series], y: Union[float, np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert metric easting/northing in metres back to longitude and latitude."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        lon, lat = self.inv_transformer.transform(x_arr, y_arr)
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def project_coordinates(
    lons: Union[float, np.ndarray, pd.Series],
    lats: Union[float, np.ndarray, pd.Series],
    epsg_code: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience functional interface to project coordinates into a given EPSG CRS."""
    projector = CoordinateProjector(epsg_code)
    return projector.forward(lons, lats)


def project_trip(
    trip: Trip,
    override_epsg: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Project both GPS track points and road network nodes of a trip into metric coordinates.

    Automatically identifies the optimal UTM zone from the median geographic coordinates
    of the trip, ensuring that both the track and the road network use the exact same
    metric coordinate space in metres.

    Parameters
    ----------
    trip : Trip
        The loaded trip dataclass.
    override_epsg : Optional[int]
        Optional manual EPSG override code.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, int]
        - projected_track: DataFrame with original columns plus ['x_m', 'y_m'].
        - projected_nodes: DataFrame with original columns plus ['x_m', 'y_m'].
        - epsg_code: The applied metric UTM EPSG code.
    """
    if override_epsg is not None:
        epsg = override_epsg
    else:
        med_lon = float(trip.track["lon"].median())
        med_lat = float(trip.track["lat"].median())
        epsg = get_utm_epsg_for_coords(med_lon, med_lat)

    projector = CoordinateProjector(epsg)

    # Project GPS track
    track_df = trip.track.copy()
    x_track, y_track = projector.forward(track_df["lon"], track_df["lat"])
    track_df["x_m"] = x_track
    track_df["y_m"] = y_track

    # Project Road Nodes
    nodes_df = trip.nodes.copy()
    x_nodes, y_nodes = projector.forward(nodes_df["lon"], nodes_df["lat"])
    nodes_df["x_m"] = x_nodes
    nodes_df["y_m"] = y_nodes

    return track_df, nodes_df, epsg
