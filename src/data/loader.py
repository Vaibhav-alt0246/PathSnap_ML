"""Trip data contract and loader boundary."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class TripData:
    """Trip-specific network; route is training/evaluation-only."""
    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.Series | None = None


def _rename_aliases(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Return a copy with supported contributor aliases mapped canonically."""
    rename = {source: target for source, target in aliases.items() if source in frame.columns and target not in frame.columns}
    return frame.rename(columns=rename)


def normalize_trip(trip: TripData) -> TripData:
    """Normalize Member A's lon/lat/u/v schema to the shared graph schema."""
    normalized = TripData(
        trip_id=trip.trip_id,
        track=_rename_aliases(trip.track, {"lon": "longitude", "lat": "latitude"}),
        nodes=_rename_aliases(trip.nodes, {"lon": "longitude", "lat": "latitude"}),
        arcs=_rename_aliases(trip.arcs, {"u": "from_node", "v": "to_node"}),
        route=trip.route,
    )
    return normalized

def validate_trip(trip: TripData, *, require_route: bool = False) -> None:
    """Validate required columns and optional route availability."""
    trip = normalize_trip(trip)
    required = {"track": {"longitude", "latitude", "timestamp"}, "nodes": {"longitude", "latitude"}, "arcs": {"from_node", "to_node"}}
    for name, columns in required.items():
        missing = columns - set(getattr(trip, name).columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if require_route and trip.route is None:
        raise ValueError("route is required for training labels/evaluation")

def load_trip(data_root: str | Path, partition: str, trip_id: str, include_route: bool = False) -> TripData:
    """Load one trip from the documented partitioned text-file layout.

    ``route`` is loaded only for non-student-test data when explicitly
    requested. No route file is accessed for student-test trips.
    """
    if partition not in {"train", "student_test", "evaluator_test"}:
        raise ValueError("partition must be 'train', 'student_test', or 'evaluator_test'")
    if include_route and partition == "student_test":
        raise PermissionError("student-test reference routes are inference-prohibited")
    trip_dir = Path(data_root) / partition / str(trip_id)
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    for path in (track_path, nodes_path, arcs_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])
    nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["longitude", "latitude"])
    arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node", "to_node"], dtype=int)
    route = None
    if include_route:
        route_path = trip_dir / f"{trip_id}.route"
        if not route_path.is_file():
            raise FileNotFoundError(route_path)
        route = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"], dtype=int)["arc_id"]
    trip = TripData(str(trip_id), track, nodes, arcs, route)
    validate_trip(trip, require_route=include_route)
    return trip
