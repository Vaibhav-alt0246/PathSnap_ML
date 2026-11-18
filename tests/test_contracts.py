import pandas as pd
import pytest

from src.candidates.generator import CANONICAL_COLUMNS, validate_candidate_table
from src.data.loader import TripData, validate_trip

def test_trip_contract_requires_core_columns():
    trip = TripData("t1", pd.DataFrame({"longitude": [1], "latitude": [2], "timestamp": [3]}), pd.DataFrame({"longitude": [1], "latitude": [2]}), pd.DataFrame({"from_node": [0], "to_node": [0]}))
    validate_trip(trip)

def test_candidate_schema_validation():
    validate_candidate_table(pd.DataFrame({column: [0] for column in CANONICAL_COLUMNS}))

def test_candidate_schema_rejects_missing_column():
    with pytest.raises(ValueError):
        validate_candidate_table(pd.DataFrame({"point_idx": [0]}))

def test_route_is_required_for_training_validation():
    trip = TripData("t1", pd.DataFrame(columns=["longitude", "latitude", "timestamp"]), pd.DataFrame(columns=["longitude", "latitude"]), pd.DataFrame(columns=["from_node", "to_node"]))
    with pytest.raises(ValueError, match="route"):
        validate_trip(trip, require_route=True)
