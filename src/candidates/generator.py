"""Candidate table contract."""
import pandas as pd

CANONICAL_COLUMNS = ["point_idx", "arc_id", "distance_m", "projection_fraction", "projected_x", "projected_y", "arc_length_m"]

def generate_candidates(trip: object) -> pd.DataFrame:
    """Generate ordered candidates; implementation awaits confirmed data/CRS."""
    del trip
    raise NotImplementedError

def validate_candidate_table(candidates: pd.DataFrame) -> None:
    """Validate required columns and candidate identity uniqueness."""
    missing = set(CANONICAL_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    if candidates[["point_idx", "arc_id"]].duplicated().any():
        raise ValueError("candidate identity must be unique")
