"""Preprocessing utilities for GPS trajectories and road networks.

Computes time deltas, step displacements, speeds, and trajectory diagnostics
without mutating raw geographic inputs.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def haversine_distance(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Compute great-circle distance between two coordinate arrays in metres.

    Parameters
    ----------
    lon1, lat1 : np.ndarray
        Start coordinates in decimal degrees.
    lon2, lat2 : np.ndarray
        End coordinates in decimal degrees.

    Returns
    -------
    np.ndarray
        Distance in metres.
    """
    R = 6371000.0  # Earth radius in metres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return R * c


def preprocess_track(track_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich GPS track observations with basic kinematic context.

    Computes time differences (delta_t), step distances (step_distance_m),
    instantaneous speeds (speed_mps), and stationarity flags.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with columns ['lon', 'lat', 'timestamp'].

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with kinematic columns added.
    """
    df = track_df.copy()
    n = len(df)
    if n == 0:
        return df

    lons = df["lon"].to_numpy(dtype=float)
    lats = df["lat"].to_numpy(dtype=float)
    times = df["timestamp"].to_numpy(dtype=float)

    # Time delta
    dt = np.zeros(n, dtype=float)
    if n > 1:
        dt[1:] = np.diff(times)
        dt[0] = dt[1] if n > 1 else 1.0
    df["delta_t"] = dt

    # Step distance via haversine
    step_dist = np.zeros(n, dtype=float)
    if n > 1:
        step_dist[1:] = haversine_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])
        step_dist[0] = step_dist[1] if n > 1 else 0.0
    df["step_distance_m"] = step_dist

    # Instantaneous speed
    safe_dt = np.where(dt > 1e-4, dt, 1.0)
    speed_mps = step_dist / safe_dt
    df["speed_mps"] = speed_mps

    # Flag stationary points (speed < 0.2 m/s and step_dist < 0.5 m)
    df["is_stationary"] = (speed_mps < 0.2) & (step_dist < 0.5)

    return df
