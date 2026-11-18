"""Reference-route labels; never used by inference."""
def align_route_labels(trip: object):
    """Align route labels; requires a training route."""
    if getattr(trip, "route", None) is None:
        raise ValueError("route is training-only and required for alignment")
    raise NotImplementedError
