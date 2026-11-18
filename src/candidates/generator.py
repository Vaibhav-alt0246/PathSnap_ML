"""Metric candidate generation for directed trip-local road arcs."""
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely import STRtree, linestrings, points
from src.data.loader import normalize_trip

CANONICAL_COLUMNS = ["point_idx", "arc_id", "distance_m", "projection_fraction", "projected_x", "projected_y", "arc_length_m"]

def _project(trip):
    t = normalize_trip(trip)
    lon = pd.concat([t.track["longitude"], t.nodes["longitude"]])
    lat = pd.concat([t.track["latitude"], t.nodes["latitude"]])
    zone = int(np.floor((float(lon.median()) + 180) / 6) + 1)
    epsg = (32600 if float(lat.median()) >= 0 else 32700) + min(zone, 60)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    tx, ty = tr.transform(t.track["longitude"].to_numpy(float), t.track["latitude"].to_numpy(float))
    nx, ny = tr.transform(t.nodes["longitude"].to_numpy(float), t.nodes["latitude"].to_numpy(float))
    return t, tx, ty, nx, ny


def generate_candidates(trip: object, search_radius_m: float = 50.0,
                        max_candidates: int = 15, fallback_nearest: bool = True) -> pd.DataFrame:
    """Generate deterministic point-to-directed-segment candidates in metres."""
    t, tx, ty, nx, ny = _project(trip)
    if len(tx) == 0 or len(t.arcs) == 0:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    au = t.arcs.from_node.to_numpy(int); av = t.arcs.to_node.to_numpy(int)
    if ((au < 0) | (av < 0) | (au >= len(nx)) | (av >= len(nx))).any():
        raise ValueError("arc references an unknown node")
    coords = np.stack([np.stack([nx[au], ny[au]], axis=1), np.stack([nx[av], ny[av]], axis=1)], axis=1)
    tree = STRtree(linestrings(coords)); gps = points(tx, ty)
    pairs = tree.query(gps, predicate="dwithin", distance=search_radius_m)
    pair_set = {(int(p), int(a)) for p, a in zip(pairs[0], pairs[1])}
    if fallback_nearest:
        nearest = tree.query_nearest(gps)
        for p, a in zip(nearest[0], nearest[1]): pair_set.add((int(p), int(a)))
    rows = []
    for point_idx, arc_idx in sorted(pair_set):
        x, y = tx[point_idx], ty[point_idx]; u, v = au[arc_idx], av[arc_idx]
        dx, dy = nx[v]-nx[u], ny[v]-ny[u]; length_sq = dx*dx + dy*dy
        fraction = 0.0 if length_sq == 0 else float(np.clip(((x-nx[u])*dx + (y-ny[u])*dy)/length_sq, 0, 1))
        px, py = nx[u] + fraction*dx, ny[u] + fraction*dy
        rows.append({"point_idx": point_idx, "arc_id": arc_idx, "distance_m": float(np.hypot(x-px, y-py)),
                     "projection_fraction": fraction, "projected_x": px, "projected_y": py,
                     "arc_length_m": float(np.sqrt(length_sq))})
    result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if result.empty: return result
    result = result.sort_values(["point_idx", "distance_m", "arc_id"])
    return result.groupby("point_idx", group_keys=False).head(max_candidates).reset_index(drop=True)

def validate_candidate_table(candidates: pd.DataFrame) -> None:
    """Validate required columns and candidate identity uniqueness."""
    missing = set(CANONICAL_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    if candidates[["point_idx", "arc_id"]].duplicated().any():
        raise ValueError("candidate identity must be unique")
