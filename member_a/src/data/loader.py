"""Data loader module for PathSnap.

Implements load_trip with strict partition isolation, path validation,
schema enforcement, and zero-based indexing preservation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any
import os
import pandas as pd

from member_a.config.settings import (
    DEFAULT_DATA_ROOT,
    TRAIN_PARTITION,
    STUDENT_TEST_PARTITION,
    PERMITTED_PARTITIONS,
    TRACK_COLUMNS,
    NODES_COLUMNS,
    ARCS_COLUMNS,
    ROUTE_COLUMNS,
)
from member_a.src.data.validation import validate_trip_data, ValidationError


@dataclass
class Trip:
    """Represents all components of a single PathSnap trip.

    Attributes
    ----------
    trip_id : str
        The unique 8-character trip identifier (e.g., '00000001').
    track : pd.DataFrame
        GPS observations with columns ['lon', 'lat', 'timestamp'].
    nodes : pd.DataFrame
        Road network nodes with columns ['lon', 'lat'], indexed by 0-based node_id.
    arcs : pd.DataFrame
        Directed network arcs with columns ['u', 'v'], indexed by 0-based arc_id.
    partition : str
        The partition this trip belongs to ('train' or 'student_test').
    route : Optional[pd.DataFrame]
        Chronological reference arc sequence with column ['arc_id'] (training only).
    """

    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    partition: str = "train"
    route: Optional[pd.DataFrame] = None

    def __post_init__(self):
        # Allow callers to use lon/lat or longitude/latitude seamlessly without mutating columns
        pass

    @property
    def num_gps(self) -> int:
        return len(self.track)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)

    @property
    def has_route(self) -> bool:
        return self.route is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Trip dataclass to dictionary representation."""
        return {
            "trip_id": self.trip_id,
            "partition": self.partition,
            "track": self.track,
            "nodes": self.nodes,
            "arcs": self.arcs,
            "route": self.route,
        }


# Alias TripData for shared architecture contract compatibility
TripData = Trip


def validate_trip(trip: object, require_route: bool = False) -> None:
    """Shared contract validation for a Trip/TripData object."""
    if trip is None:
        raise ValueError("trip cannot be None")
    track = getattr(trip, "track", None)
    nodes = getattr(trip, "nodes", None)
    arcs = getattr(trip, "arcs", None)
    route = getattr(trip, "route", None)

    if track is None or not isinstance(track, pd.DataFrame):
        raise ValueError("trip.track must be a DataFrame")
    if nodes is None or not isinstance(nodes, pd.DataFrame):
        raise ValueError("trip.nodes must be a DataFrame")
    if arcs is None or not isinstance(arcs, pd.DataFrame):
        raise ValueError("trip.arcs must be a DataFrame")

    has_long = ("longitude" in track.columns) or ("lon" in track.columns)
    has_lat = ("latitude" in track.columns) or ("lat" in track.columns)
    has_ts = "timestamp" in track.columns
    if not (has_long and has_lat and has_ts):
        raise ValueError("track must have longitude/lon, latitude/lat, and timestamp")

    has_node_long = ("longitude" in nodes.columns) or ("lon" in nodes.columns)
    has_node_lat = ("latitude" in nodes.columns) or ("lat" in nodes.columns)
    if not (has_node_long and has_node_lat):
        raise ValueError("nodes must have longitude/lon and latitude/lat")

    has_u = ("from_node" in arcs.columns) or ("u" in arcs.columns)
    has_v = ("to_node" in arcs.columns) or ("v" in arcs.columns)
    if not (has_u and has_v):
        raise ValueError("arcs must have from_node/u and to_node/v")

    if require_route:
        if route is None or not isinstance(route, pd.DataFrame) or len(route) == 0:
            raise ValueError("route is training-only and required for validation")


def load_trip(
    data_root: Optional[Union[str, Path]] = None,
    partition: str = TRAIN_PARTITION,
    trip_id: str = "00000001",
    include_route: bool = False,
    validate: bool = True,
) -> Trip:
    """Load a PathSnap trip with data integrity and leakage safeguards.

    Parameters
    ----------
    data_root : Optional[Union[str, Path]]
        Root path to dataset folder containing 'train' and 'student_test' directories.
        Defaults to DEFAULT_DATA_ROOT.
    partition : str
        Partition name ('train' or 'student_test').
    trip_id : str
        Trip identifier folder name (e.g., '00000001').
    include_route : bool
        Whether to load the reference route. MUST be False for student_test partition.
    validate : bool
        Whether to run structural and numeric validation after loading.

    Returns
    -------
    Trip
        Trip dataclass holding track, nodes, arcs, and optional route DataFrames.

    Raises
    ------
    PermissionError
        If include_route=True is requested for student_test partition (Leakage Safeguard).
    FileNotFoundError
        If the trip directory or any required file (.track, .nodes, .arcs) does not exist.
    ValueError
        If the partition is invalid or data fails validation.
    """
    if partition not in PERMITTED_PARTITIONS:
        raise ValueError(
            f"Invalid partition '{partition}'. Must be one of {PERMITTED_PARTITIONS}."
        )

    # CRITICAL DATA LEAKAGE SAFEGUARD:
    # Student test route files must NEVER be accessed.
    if include_route and partition == STUDENT_TEST_PARTITION:
        raise PermissionError(
            f"Access Denied: Attempted to load reference route for student_test trip '{trip_id}'. "
            f"Official rules strictly prohibit reading .route files in student_test."
        )

    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    trip_dir = root / partition / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip directory does not exist: {trip_dir.resolve()}"
        )

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    for req_file, name in [
        (track_file, "track"),
        (nodes_file, "nodes"),
        (arcs_file, "arcs"),
    ]:
        if not req_file.is_file():
            raise FileNotFoundError(
                f"Missing required {name} file for trip {trip_id}: {req_file.resolve()}"
            )

    # Load Track (lon, lat, timestamp)
    track_df = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=TRACK_COLUMNS,
        dtype=float,
        engine="c",
    )

    # Load Nodes (lon, lat)
    nodes_df = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=NODES_COLUMNS,
        dtype=float,
        engine="c",
    )

    # Load Arcs (u, v)
    arcs_df = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=ARCS_COLUMNS,
        dtype=int,
        engine="c",
    )

    # Load Route (optional, permitted for train only)
    route_df: Optional[pd.DataFrame] = None
    if include_route and partition == TRAIN_PARTITION:
        if not route_file.is_file():
            raise FileNotFoundError(
                f"Requested route file does not exist: {route_file.resolve()}"
            )
        route_df = pd.read_csv(
            route_file,
            sep=r"\s+",
            header=None,
            names=ROUTE_COLUMNS,
            dtype=int,
            engine="c",
        )

    if validate:
        validate_trip_data(
            track_df=track_df,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            route_df=route_df,
            trip_id=trip_id,
        )

    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_df,
    )
