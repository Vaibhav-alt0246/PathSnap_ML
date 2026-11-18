"""Leakage-free feature construction matching the approved model schema."""
import numpy as np
import pandas as pd
from src.data.loader import normalize_trip

FEATURE_COLUMNS = ["distance_m", "distance_to_closest_cand", "distance_ratio_closest", "dist_rank",
"cand_count_for_point", "projection_fraction", "dist_to_segment_start_m", "dist_to_segment_end_m",
"arc_length_m", "gps_heading_deg", "road_bearing_deg", "heading_bearing_diff_deg",
"heading_bearing_cos_sim", "is_low_speed", "gps_delta_t_s", "gps_step_distance_m", "gps_speed_mps",
"gps_accel_mps2", "speed_ratio_to_mean", "u_in_degree", "u_out_degree", "v_in_degree",
"v_out_degree", "has_reverse_twin"]

def _bearing(dx, dy): return (np.degrees(np.arctan2(dx, dy)) + 360) % 360
def _angle(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)

def build_features(trip: object, candidates: object) -> pd.DataFrame:
    """Build the same geometric, kinematic and topology features used in training."""
    t = normalize_trip(trip); out = candidates.copy()
    if out.empty: return out.assign(**{c: pd.Series(dtype=float) for c in FEATURE_COLUMNS})
    lon = t.track.longitude.to_numpy(float); lat = t.track.latitude.to_numpy(float)
    from src.candidates.generator import _project
    _, tx, ty, nx, ny = _project(t)
    times = pd.to_numeric(t.track.timestamp, errors="coerce").to_numpy(float)
    n = len(tx); dt = np.ones(n); step = np.zeros(n)
    if n > 1:
        rawdt = np.diff(times); dt[1:] = np.where(rawdt > 1e-4, rawdt, 1.0); dt[0] = dt[1]
        step[1:] = np.hypot(np.diff(tx), np.diff(ty)); step[0] = step[1]
    gap = dt > 60; step[gap] = 0; speed = step / dt; speed[gap] = 0
    accel = np.zeros(n)
    if n > 1: accel[1:] = np.where(gap[1:] | gap[:-1], 0, np.diff(speed) / dt[1:]); accel[0] = accel[1]
    heading = np.zeros(n)
    if n == 2: heading[:] = _bearing(np.array([tx[1]-tx[0]]), np.array([ty[1]-ty[0]]))[0]
    elif n > 2:
        dx = np.r_[tx[1]-tx[0], tx[2:]-tx[:-2], tx[-1]-tx[-2]]
        dy = np.r_[ty[1]-ty[0], ty[2:]-ty[:-2], ty[-1]-ty[-2]]; heading = _bearing(dx, dy)
    low = ((speed < .5) | (step < .5) | gap).astype(int)
    min_d = out.groupby("point_idx").distance_m.transform("min")
    out["distance_to_closest_cand"] = out.distance_m - min_d
    out["distance_ratio_closest"] = np.clip(out.distance_m / (min_d + 1e-4), 0, 1000)
    out["dist_rank"] = out.groupby("point_idx").distance_m.rank(method="first") - 1
    out["cand_count_for_point"] = out.groupby("point_idx").distance_m.transform("count")
    out["dist_to_segment_start_m"] = out.projection_fraction * out.arc_length_m
    out["dist_to_segment_end_m"] = (1 - out.projection_fraction) * out.arc_length_m
    pts = out.point_idx.to_numpy(int); arcs = out.arc_id.to_numpy(int)
    au = t.arcs.from_node.to_numpy(int); av = t.arcs.to_node.to_numpy(int)
    dx = nx[av[arcs]] - nx[au[arcs]]; dy = ny[av[arcs]] - ny[au[arcs]]
    rb = _bearing(dx, dy); gh = heading[pts]; diff = _angle(gh, rb); cos = np.cos(np.radians(diff));
    diff = np.where(low[pts], 90, diff); cos = np.where(low[pts], 0, cos); gh = np.where(low[pts], 0, gh)
    out["gps_heading_deg"] = gh; out["road_bearing_deg"] = rb; out["heading_bearing_diff_deg"] = diff; out["heading_bearing_cos_sim"] = cos
    out["is_low_speed"] = low[pts]; out["gps_delta_t_s"] = dt[pts]; out["gps_step_distance_m"] = step[pts]; out["gps_speed_mps"] = speed[pts]; out["gps_accel_mps2"] = accel[pts]
    out["speed_ratio_to_mean"] = speed[pts] / max(float(speed.mean()), 1e-3)
    indeg = np.bincount(av, minlength=len(t.nodes)); outdeg = np.bincount(au, minlength=len(t.nodes))
    out["u_in_degree"] = indeg[au[arcs]]; out["u_out_degree"] = outdeg[au[arcs]]; out["v_in_degree"] = indeg[av[arcs]]; out["v_out_degree"] = outdeg[av[arcs]]
    edge_set = set(zip(au, av)); out["has_reverse_twin"] = [int((av[a], au[a]) in edge_set) for a in arcs]
    return out.replace([np.inf, -np.inf], 0).fillna(0)
