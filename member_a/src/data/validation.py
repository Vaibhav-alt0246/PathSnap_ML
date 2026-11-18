"""Data validation module for PathSnap.

Provides thorough validation checks on GPS observations, road network nodes,
directed arcs, and reference routes to ensure strict schema adherence and
detect anomalies early.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when trip data violates structural, geometric, or topological constraints."""
    pass


def validate_trip_data(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame] = None,
    trip_id: str = "unknown",
) -> Dict[str, Any]:
    """Validate all components of a trip.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with columns ['lon', 'lat', 'timestamp'].
    nodes_df : pd.DataFrame
        DataFrame with columns ['lon', 'lat'].
    arcs_df : pd.DataFrame
        DataFrame with columns ['u', 'v'].
    route_df : Optional[pd.DataFrame]
        DataFrame with column ['arc_id'] (for training trips only).
    trip_id : str
        Trip identifier for reporting.

    Returns
    -------
    Dict[str, Any]
        Dictionary of validation findings, anomaly counts, and metrics.

    Raises
    ------
    ValidationError
        If critical integrity constraints are violated (empty files, missing values,
        invalid indices, non-monotonic timestamps).
    """
    anomalies: List[str] = []

    # 1. Validate Track
    if track_df is None or len(track_df) == 0:
        raise ValidationError(f"Trip {trip_id}: GPS track is empty.")
    if track_df.shape[1] < 3:
        raise ValidationError(f"Trip {trip_id}: GPS track must have at least 3 columns (lon, lat, timestamp).")
    if track_df.isna().any().any():
        raise ValidationError(f"Trip {trip_id}: GPS track contains NaN or missing values.")
    
    lons = track_df.iloc[:, 0].to_numpy(dtype=float)
    lats = track_df.iloc[:, 1].to_numpy(dtype=float)
    times = track_df.iloc[:, 2].to_numpy(dtype=float)

    if np.any(np.isinf(lons)) or np.any(np.isinf(lats)) or np.any(np.isinf(times)):
        raise ValidationError(f"Trip {trip_id}: GPS track contains infinite values.")
    if np.any((lons < -180.0) | (lons > 180.0)):
        raise ValidationError(f"Trip {trip_id}: GPS longitude out of valid range [-180, 180].")
    if np.any((lats < -90.0) | (lats > 90.0)):
        raise ValidationError(f"Trip {trip_id}: GPS latitude out of valid range [-90, 90].")
    
    time_diffs = np.diff(times)
    if len(time_diffs) > 0 and np.any(time_diffs < 0):
        raise ValidationError(f"Trip {trip_id}: GPS timestamps are not monotonically non-decreasing.")
    
    # 2. Validate Nodes
    if nodes_df is None or len(nodes_df) == 0:
        raise ValidationError(f"Trip {trip_id}: Road nodes table is empty.")
    if nodes_df.shape[1] < 2:
        raise ValidationError(f"Trip {trip_id}: Nodes must have 2 columns (lon, lat).")
    if nodes_df.isna().any().any():
        raise ValidationError(f"Trip {trip_id}: Nodes contains NaN or missing values.")

    node_lons = nodes_df.iloc[:, 0].to_numpy(dtype=float)
    node_lats = nodes_df.iloc[:, 1].to_numpy(dtype=float)
    if np.any((node_lons < -180.0) | (node_lons > 180.0)) or np.any((node_lats < -90.0) | (node_lats > 90.0)):
        raise ValidationError(f"Trip {trip_id}: Node coordinates out of valid [-180, 180], [-90, 90] bounds.")

    num_nodes = len(nodes_df)

    # 3. Validate Arcs
    if arcs_df is None or len(arcs_df) == 0:
        raise ValidationError(f"Trip {trip_id}: Road arcs table is empty.")
    if arcs_df.shape[1] < 2:
        raise ValidationError(f"Trip {trip_id}: Arcs must have 2 columns (u, v).")
    if arcs_df.isna().any().any():
        raise ValidationError(f"Trip {trip_id}: Arcs contains NaN or missing values.")

    u_nodes = arcs_df.iloc[:, 0].to_numpy(dtype=int)
    v_nodes = arcs_df.iloc[:, 1].to_numpy(dtype=int)

    if np.any((u_nodes < 0) | (u_nodes >= num_nodes)) or np.any((v_nodes < 0) | (v_nodes >= num_nodes)):
        raise ValidationError(
            f"Trip {trip_id}: Arcs reference invalid node IDs outside [0, {num_nodes - 1}]."
        )

    num_arcs = len(arcs_df)
    self_loops = int(np.sum(u_nodes == v_nodes))
    if self_loops > 0:
        anomalies.append(f"Network contains {self_loops} self-loop arcs (u == v).")

    # 4. Validate Route (if provided)
    if route_df is not None:
        if len(route_df) == 0:
            raise ValidationError(f"Trip {trip_id}: Route table is empty.")
        if route_df.isna().any().any():
            raise ValidationError(f"Trip {trip_id}: Route contains NaN or missing values.")
        
        route_arcs = route_df.iloc[:, 0].to_numpy(dtype=int)
        if np.any((route_arcs < 0) | (route_arcs >= num_arcs)):
            raise ValidationError(
                f"Trip {trip_id}: Route contains arc IDs outside [0, {num_arcs - 1}]."
            )

    return {
        "trip_id": trip_id,
        "is_valid": True,
        "num_gps": len(track_df),
        "num_nodes": num_nodes,
        "num_arcs": num_arcs,
        "num_route_arcs": len(route_df) if route_df is not None else None,
        "anomalies": anomalies,
    }
