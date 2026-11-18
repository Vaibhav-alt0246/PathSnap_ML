import pandas as pd

from src.data.loader import TripData
from src.hmm.decoder import decode


def test_decode_selects_connected_high_probability_arcs():
    trip = TripData("t1", pd.DataFrame(columns=["longitude", "latitude", "timestamp"]), pd.DataFrame({"longitude": [0.0, 0.001, 0.002], "latitude": [0.0, 0.0, 0.0]}), pd.DataFrame({"from_node": [0, 1], "to_node": [1, 2]}))
    candidates = pd.DataFrame({"point_idx": [0, 1], "arc_id": [0, 1], "projection_fraction": [0.5, 0.5]})
    assert decode(trip, candidates, [0.9, 0.8]) == [0, 1]


def test_decode_empty_candidates_returns_empty():
    trip = TripData("t1", pd.DataFrame(columns=["longitude", "latitude", "timestamp"]), pd.DataFrame({"longitude": [0.0], "latitude": [0.0]}), pd.DataFrame({"from_node": [], "to_node": []}))
    candidates = pd.DataFrame(columns=["point_idx", "arc_id", "projection_fraction"])
    assert decode(trip, candidates, []) == []
