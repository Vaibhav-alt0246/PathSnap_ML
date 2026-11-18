"""Prediction evaluation boundary."""
def evaluate_predictions(trip: object, predicted_arc_ids: object) -> dict[str, float]:
    """Evaluate predictions against route data in evaluation context."""
    del trip, predicted_arc_ids
    raise NotImplementedError
