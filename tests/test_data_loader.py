import pandas as pd

from src.data.loader import TripData, normalize_trip


def test_member_a_aliases_normalize_to_shared_contract():
    trip = TripData(
        "t1",
        pd.DataFrame({"lon": [1.0], "lat": [2.0], "timestamp": [0.0]}),
        pd.DataFrame({"lon": [1.0], "lat": [2.0]}),
        pd.DataFrame({"u": [0], "v": [0]}),
    )
    normalized = normalize_trip(trip)
    assert list(normalized.nodes.columns) == ["longitude", "latitude"]
    assert list(normalized.arcs.columns) == ["from_node", "to_node"]
