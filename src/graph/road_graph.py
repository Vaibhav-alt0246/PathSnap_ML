"""Construction of trip-local directed road graphs."""
from __future__ import annotations
import math
from numbers import Integral
import re
import networkx as nx
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString
from src.data.loader import TripData, normalize_trip

_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")


def _parse_node_reference(value: object, *, arc_id: object, field: str) -> int:
    """Convert an endpoint only when it represents an exact finite integer."""
    if isinstance(value, bool):
        raise ValueError(f"arc {arc_id} has malformed {field} node reference")
    if isinstance(value, str):
        text = value.strip()
        if not _INTEGER_TEXT.fullmatch(text):
            raise ValueError(f"arc {arc_id} has malformed {field} node reference")
        return int(text)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"arc {arc_id} has malformed {field} node reference") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"arc {arc_id} has malformed {field} node reference")
    return int(numeric)

def _metric_transformer(nodes: pd.DataFrame) -> Transformer:
    """Create a local azimuthal-equidistant metric transformer."""
    if nodes.empty:
        raise ValueError("nodes must contain at least one row")
    lon = pd.to_numeric(nodes["longitude"], errors="coerce")
    lat = pd.to_numeric(nodes["latitude"], errors="coerce")
    if lon.isna().any() or lat.isna().any() or ((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)).any():
        raise ValueError("node coordinates must be finite longitude/latitude values")
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={float(lat.mean())} +lon_0={float(lon.mean())} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)

def build_road_graph(trip: TripData) -> nx.MultiDiGraph:
    """Build a directed graph preserving local node and arc IDs.

    A local azimuthal-equidistant projection provides metre-based lengths for
    one trip. The training/evaluation-only ``route`` field is ignored.
    """
    trip = normalize_trip(trip)
    if not {"longitude", "latitude"}.issubset(trip.nodes.columns):
        raise ValueError("nodes must contain longitude and latitude columns")
    if not {"from_node", "to_node"}.issubset(trip.arcs.columns):
        raise ValueError("arcs must contain from_node and to_node columns")
    transformer = _metric_transformer(trip.nodes)
    graph = nx.MultiDiGraph(trip_id=trip.trip_id)
    projected: dict[int, tuple[float, float]] = {}
    for node_id, row in trip.nodes.iterrows():
        if not isinstance(node_id, Integral):
            raise ValueError("node IDs must be integer row indices")
        lon, lat = float(row["longitude"]), float(row["latitude"])
        x, y = transformer.transform(lon, lat)
        node_id = int(node_id)
        projected[node_id] = (x, y)
        graph.add_node(node_id, longitude=lon, latitude=lat, projected_x=x, projected_y=y)
    for arc_id, row in trip.arcs.iterrows():
        if not isinstance(arc_id, Integral):
            raise ValueError("arc IDs must be integer row indices")
        source = _parse_node_reference(row["from_node"], arc_id=arc_id, field="from_node")
        target = _parse_node_reference(row["to_node"], arc_id=arc_id, field="to_node")
        if source not in projected or target not in projected:
            raise ValueError(f"arc {arc_id} references an unknown node: {source}->{target}")
        geometry = LineString([projected[source], projected[target]])
        graph.add_edge(source, target, key=int(arc_id), arc_id=int(arc_id), from_node=source, to_node=target, geometry=geometry, length_m=float(geometry.length))
    return graph
