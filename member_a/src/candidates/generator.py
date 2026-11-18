"""Candidate road segment generation module for PathSnap.

Generates candidate directed road arcs for each GPS observation using metric UTM
projections and high-performance spatial indexing (Shapely STRtree).
Calculates exact perpendicular distances, along-segment projection fractions,
projected metric coordinates, and arc lengths.
"""

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from member_a.config.settings import (
    CANDIDATE_SEARCH_RADIUS_M,
    MAX_CANDIDATES_PER_POINT,
    CANDIDATE_COLUMNS,
)
from member_a.src.data.loader import Trip
from member_a.src.projection.projection import project_trip

CANONICAL_COLUMNS = list(CANDIDATE_COLUMNS)


def validate_candidate_table(candidates: pd.DataFrame) -> None:
    """Validate required columns and candidate identity uniqueness (shared contract)."""
    missing = set(CANONICAL_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    if candidates[["point_idx", "arc_id"]].duplicated().any():
        raise ValueError("candidate identity must be unique")


def _compute_segment_projections(
    px: np.ndarray,
    py: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of point-to-segment projections and perpendicular distances.

    Parameters
    ----------
    px, py : np.ndarray
        Observation point coordinates in metric metres.
    ux, uy : np.ndarray
        Arc start node coordinates in metric metres.
    vx, vy : np.ndarray
        Arc end node coordinates in metric metres.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (distance_m, projection_fraction, proj_x, proj_y, arc_length_m)
    """
    dx = vx - ux
    dy = vy - uy
    lens_sq = dx * dx + dy * dy
    arc_lengths = np.sqrt(lens_sq)

    # Handle zero-length segments safely
    is_zero = lens_sq < 1e-12
    safe_lens_sq = np.where(is_zero, 1.0, lens_sq)

    # Projection fraction along vector u -> v
    t = ((px - ux) * dx + (py - uy) * dy) / safe_lens_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    t_clamped = np.where(is_zero, 0.0, t_clamped)

    # Projected coordinates on segment
    proj_x = ux + t_clamped * dx
    proj_y = uy + t_clamped * dy

    # Perpendicular distance in metres
    dist_x = px - proj_x
    dist_y = py - proj_y
    distances = np.sqrt(dist_x * dist_x + dist_y * dist_y)

    return distances, t_clamped, proj_x, proj_y, arc_lengths


def generate_candidates(
    trip: Trip,
    search_radius_m: float = CANDIDATE_SEARCH_RADIUS_M,
    max_candidates: int = MAX_CANDIDATES_PER_POINT,
    fallback_nearest: bool = True,
    override_epsg: Optional[int] = None,
) -> pd.DataFrame:
    """Generate candidate road segments for each GPS observation in a trip.

    For each GPS observation, identifies nearby directed road arcs within the search radius
    using a metric spatial index. Computes perpendicular distances, projection fractions,
    projected coordinates, and segment lengths.

    Parameters
    ----------
    trip : Trip
        The loaded trip dataclass.
    search_radius_m : float
        Spatial candidate search radius in metres (default 50.0m).
    max_candidates : int
        Maximum number of candidates retained per GPS point (default 15).
    fallback_nearest : bool
        If True, ensures points with no candidates within search_radius_m receive
        their single closest road segment.
    override_epsg : Optional[int]
        Optional manual EPSG projection code override.

    Returns
    -------
    pd.DataFrame
        Candidate DataFrame containing columns:
        ['point_idx', 'arc_id', 'distance_m', 'projection_fraction',
         'projected_x', 'projected_y', 'arc_length_m']
        Deterministic ordering: point_idx asc, distance_m asc, arc_id asc.
    """
    # 1. Project GPS track and road nodes to metric UTM space
    proj_track, proj_nodes, epsg = project_trip(trip, override_epsg=override_epsg)

    num_gps = len(proj_track)
    num_arcs = len(trip.arcs)

    if num_gps == 0 or num_arcs == 0:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    # 2. Extract projected coordinates
    gps_x = proj_track["x_m"].to_numpy(dtype=float)
    gps_y = proj_track["y_m"].to_numpy(dtype=float)

    node_x = proj_nodes["x_m"].to_numpy(dtype=float)
    node_y = proj_nodes["y_m"].to_numpy(dtype=float)

    u_nodes = trip.arcs["u"].to_numpy(dtype=int)
    v_nodes = trip.arcs["v"].to_numpy(dtype=int)

    # 3. Build vectorized line segments and spatial index
    arc_coords = np.stack(
        [
            np.column_stack([node_x[u_nodes], node_y[u_nodes]]),
            np.column_stack([node_x[v_nodes], node_y[v_nodes]]),
        ],
        axis=1,
    )
    arc_lines = shapely.linestrings(arc_coords)
    spatial_tree = STRtree(arc_lines)

    # 4. Spatial query using Shapely STRtree (distance within search_radius_m)
    gps_points = shapely.points(gps_x, gps_y)
    pt_indices, arc_indices = spatial_tree.query(
        gps_points, predicate="dwithin", distance=search_radius_m
    )

    # Check for points with zero candidates within search radius
    if fallback_nearest and len(pt_indices) > 0:
        covered_pts = set(np.unique(pt_indices))
        missing_pts = [p for p in range(num_gps) if p not in covered_pts]
        if missing_pts:
            missing_geom = gps_points[missing_pts]
            nearest_pairs = spatial_tree.query_nearest(missing_geom)
            # nearest_pairs[0] is index in missing_geom, nearest_pairs[1] is arc_idx
            missing_pt_idx = np.array(missing_pts)[nearest_pairs[0]]
            missing_arc_idx = nearest_pairs[1]

            pt_indices = np.concatenate([pt_indices, missing_pt_idx])
            arc_indices = np.concatenate([arc_indices, missing_arc_idx])
    elif fallback_nearest and len(pt_indices) == 0:
        # Fallback for all points if no points found within radius
        nearest_pairs = spatial_tree.query_nearest(gps_points)
        pt_indices = nearest_pairs[0]
        arc_indices = nearest_pairs[1]

    if len(pt_indices) == 0:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    # 5. Compute exact geometric projection metrics
    pt_x = gps_x[pt_indices]
    pt_y = gps_y[pt_indices]

    arc_u = u_nodes[arc_indices]
    arc_v = v_nodes[arc_indices]

    ux = node_x[arc_u]
    uy = node_y[arc_u]
    vx = node_x[arc_v]
    vy = node_y[arc_v]

    distances, fractions, px, py, arc_lens = _compute_segment_projections(
        pt_x, pt_y, ux, uy, vx, vy
    )

    # 6. Assemble candidate DataFrame
    df_raw = pd.DataFrame(
        {
            "point_idx": pt_indices.astype(int),
            "arc_id": arc_indices.astype(int),
            "distance_m": distances.astype(float),
            "projection_fraction": fractions.astype(float),
            "projected_x": px.astype(float),
            "projected_y": py.astype(float),
            "arc_length_m": arc_lens.astype(float),
        }
    )

    # 7. Deduplicate identical (point_idx, arc_id) pairs (keep smallest distance)
    df_dedup = df_raw.sort_values(
        by=["point_idx", "arc_id", "distance_m"]
    ).drop_duplicates(subset=["point_idx", "arc_id"], keep="first")

    # 8. Sort deterministically: point_idx asc, distance_m asc, arc_id asc
    df_sorted = df_dedup.sort_values(
        by=["point_idx", "distance_m", "arc_id"], ascending=[True, True, True]
    )

    # 9. Apply max_candidates per GPS observation
    if max_candidates is not None and max_candidates > 0:
        df_final = (
            df_sorted.groupby("point_idx", as_index=False, group_keys=False)
            .head(max_candidates)
            .reset_index(drop=True)
        )
    else:
        df_final = df_sorted.reset_index(drop=True)

    return df_final[CANDIDATE_COLUMNS]
