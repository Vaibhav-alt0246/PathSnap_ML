# PathSnap data contracts

The canonical internal trip contract is `TripData(trip_id, track, nodes, arcs, route=None)`.

- `track`: `longitude`, `latitude`, `timestamp`
- `nodes`: `longitude`, `latitude`; row index is the trip-local node ID
- `arcs`: `from_node`, `to_node`; row index is the trip-local arc ID
- `route`: training/evaluation-only ordered arc IDs; never loaded for student-test inference

`src.data.loader.normalize_trip` accepts Member A aliases (`lon`, `lat`, `u`, `v`) without resetting indices or changing IDs. Graph edges retain `arc_id`, endpoints, metric geometry, and `length_m`.

Candidate rows and model probabilities must retain identical order. Model features must exclude identifiers, labels, route-derived fields, and absolute location identity columns.
