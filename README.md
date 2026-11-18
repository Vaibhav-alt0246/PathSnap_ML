# PathSnap

PathSnap is a Python machine-learning project for map matching: identifying travelled segments from noisy GPS observations and a directed, trip-specific road network.

This repository is a development foundation. Domain-specific ML, graph, HMM, API, and UI implementations are staged behind typed interfaces and are not claimed to be complete.

## Data contract

Each trip contains `.track` (longitude, latitude, timestamp), `.nodes` (longitude, latitude), `.arcs` (from_node, to_node), and `.route` (ordered reference arc IDs). Node and arc IDs are row indices and local to each trip. `.route` is training/evaluation-only and must never be loaded by inference. Metric projection is required for geometric calculations; raw longitude/latitude are not model features.

Candidate tables use `point_idx`, `arc_id`, `distance_m`, `projection_fraction`, `projected_x`, `projected_y`, and `arc_length_m`; training adds `label`. Identity and label columns must be excluded from model features, and probability rows retain candidate order.

## Architecture and ownership

- Member A: loading, projection, candidate generation, route alignment, and features.
- Member B: models, trip-grouped cross-validation, comparison, metrics, and optimization.
- Member C: graph/shortest paths, HMM/Viterbi, inference, FastAPI, Streamlit, and integration.

The package boundaries are under `src/`: `data`, `candidates`, `labels`, `features`, `models`, `tuning`, `graph`, `hmm`, `evaluation`, `pipeline`, and `visualization`.

## Shared interfaces

The agreed entry points are `load_trip`, `generate_candidates`, `align_route_labels`, `build_features`, `train_models`, `compare_models`, `optimize_best_model`, `predict_proba`, `build_road_graph`, `decode`, `evaluate_predictions`, and `predict_trip`. Their typed signatures and contracts are documented in module docstrings. Unimplemented boundaries raise `NotImplementedError` rather than pretending to work.

## Setup and configuration

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

`config/settings.yaml` is the single configuration source. Dataset and metric CRS paths are intentionally `null` until confirmed; do not commit private data or credentials.

## Testing

```bash
python -m compileall -q api src ui tests
pytest
```

Tests cover the trip and candidate contracts and training-route requirements without requiring a dataset.

## Development workflow

`main` is the shared integration branch. Use `member-a-data`, `member-b-ml`, or `member-c-integration` for owned work and submit pull requests into `main`; do not merge another member's work automatically or rewrite history. See `docs/DEVELOPMENT_WORKFLOW.md`.

The intended two-laptop deployment is FastAPI on Laptop 1 and Streamlit/Folium on Laptop 2. Configure the UI API base URL for the actual local network environment; no address is assumed here.

## Current status

Completed: repository/package foundation, configuration boundary, data/candidate contracts, API/UI placeholders, and dataset-independent tests. Planned: confirm the dataset parser/CRS, then implement Member A/B/C modules behind these interfaces.
