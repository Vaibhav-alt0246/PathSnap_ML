from src.data.loader import TripData
from src.visualization.mapping import create_map
import pandas as pd


def _trip():
    return TripData(
        "synthetic",
        pd.DataFrame({"longitude": [77.0, 77.0001], "latitude": [12.0, 12.0001], "timestamp": [0, 1]}),
        pd.DataFrame({"longitude": [77.0, 77.0001, 77.0002], "latitude": [12.0, 12.0001, 12.0002]}),
        pd.DataFrame({"from_node": [0, 1], "to_node": [1, 2]}),
    )


def test_ui_map_call_accepts_classifier_route_and_contains_layers():
    html = create_map(_trip(), predicted_arc_ids=[1], classifier_arc_ids=[0]).get_root().render()
    assert "OpenStreetMap" in html
    assert "HMM predicted route" in html
    assert "Classifier selections" in html
    assert "GPS observations" in html
    assert "0057b8" in html
    assert "7b2cbf" in html
    assert "GPS observation" in html


def test_map_accepts_empty_classifier_predictions():
    html = create_map(_trip(), predicted_arc_ids=[], classifier_arc_ids=None).get_root().render()
    assert "HMM predicted route" in html
