"""Reference-route labels; never used by inference."""
def align_route_labels(trip: object):
    """Align route labels; requires a training route."""
    route = getattr(trip, "route", None)
    if route is None:
        raise ValueError("route is training-only and required for alignment")
    return route


def label_candidates(candidates, trip: object):
    """Add candidate labels from the trip-local reference route.

    Candidate arc IDs and route arc IDs are both local row indices in the
    trip's ``.arcs`` file. The route is consumed only by training/evaluation.
    """
    import pandas as pd
    route = align_route_labels(trip)
    route_ids = {int(value) for value in route.tolist()}
    result = candidates.copy()
    result["label"] = result["arc_id"].astype(int).isin(route_ids).astype(int)
    return result
