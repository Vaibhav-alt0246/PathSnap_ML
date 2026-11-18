# PathSnap ML

PathSnap is a Python machine-learning project for map matching: identifying travelled segments from noisy GPS observations and a directed, trip-specific road network.

The repository contains an assignment-scale inference path: raw trip loading, metric candidate generation, 24-feature extraction, classifier probabilities, directed HMM/Viterbi decoding, Folium maps, FastAPI, and Streamlit integration.

PathSnap addresses GPS map matching: noisy longitude/latitude observations must be assigned to plausible directed road arcs in a trip-specific network. Candidate generation searches within 50 m and keeps at most 15 candidates per observation. HistGradientBoosting scores the approved 24 engineered features; directed HMM/Viterbi decoding then uses road connectivity and emissions to produce an ordered route. The HMM is a sequence decoder, so it must not be described as improving every accuracy metric.

## Data contract

Each trip contains `.track` (longitude, latitude, timestamp), `.nodes` (longitude, latitude), `.arcs` (from_node, to_node), and `.route` (ordered reference arc IDs). Node and arc IDs are row indices and local to each trip. `.route` is training/evaluation-only and must never be loaded by inference. Metric projection is required for geometric calculations; raw longitude/latitude are not model features.

Candidate tables use `point_idx`, `arc_id`, `distance_m`, `projection_fraction`, `projected_x`, `projected_y`, and `arc_length_m`; training adds `label`. Identity and label columns must be excluded from model features, and probability rows retain candidate order.

## Architecture and ownership

- Member A: loading, projection, candidate generation, route alignment, and features.
- Member B: models, trip-grouped cross-validation, comparison, metrics, and optimization.
- Member C: graph/shortest paths, HMM/Viterbi, inference, FastAPI, Streamlit, and integration.

The package boundaries are under `src/`: `data`, `candidates`, `labels`, `features`, `models`, `tuning`, `graph`, `hmm`, `evaluation`, `pipeline`, and `visualization`.

## Shared interfaces

The agreed entry points are `load_trip`, `generate_candidates`, `align_route_labels`, `build_features`, `train_models`, `compare_models`, `optimize_best_model`, `predict_proba`, `build_road_graph`, `decode`, `evaluate_predictions`, and `predict_trip`. Prediction never loads `.route`; routes are only used separately for evaluation.

## Setup and configuration

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

`config/settings.yaml` is the single configuration source. Dataset and metric CRS paths are intentionally `null` until confirmed; do not commit private data or credentials.

### Model and data files

The repository includes the feature-order contract at `models/config/feature_columns.json` and the verified reconstructed baseline at `models/artifacts/hist_gradient_boosting_reconstructed_baseline.joblib`. This is a new model trained on the local 80-trip training partition; it is not the missing historical `/tmp/pathsnap-phase2-artifacts/best_baseline_model.joblib`. Its provenance, dataset fingerprint, parameters, grouped metrics, and skipped-trip report are in `models/artifacts/`. The model was trained with the 24 configured columns. The reproducible rebuild command is:

```bash
python scripts/train_baseline.py --dataset-root /absolute/path/to/dataset --output-dir models/artifacts
```

The dataset is not committed. Set `PATHSNAP_DATASET_ROOT` to a directory with this layout:

```text
DATASET_ROOT/{train,student_test,evaluator_test}/TRIP_ID/
  TRIP_ID.track   # longitude latitude timestamp
  TRIP_ID.nodes   # longitude latitude
  TRIP_ID.arcs    # from_node to_node
  TRIP_ID.route   # training/evaluation only; never required for inference
```

`PATHSNAP_EVALUATOR_DATASET_ROOT` may point to the evaluator root (or its parent) when evaluator endpoints are used. Student-test and evaluator data are local/private inputs and must not be committed.

### Running the services

Start the backend first:

```bash
export PATHSNAP_DATASET_ROOT=/absolute/path/to/dataset
export PATHSNAP_MODEL_PATH="$PWD/models/artifacts/hist_gradient_boosting_reconstructed_baseline.joblib"
./scripts/start_local.sh
```

In another shell, activate the same environment and start the Streamlit frontend. Set `PATHSNAP_API_BASE_URL` when the API is on another host:

```bash
source .venv/bin/activate
export PATHSNAP_API_BASE_URL=http://127.0.0.1:8000
streamlit run ui/app.py
```

Development-trip, evaluator, and uploaded-trip workflows are API-backed. Uploaded `.track`, `.nodes`, and `.arcs` files are validated and processed in memory by `POST /upload/predict`; they are not permanently stored. The API owns candidate generation, feature engineering, model inference, HMM decoding, and map rendering.

## Testing

```bash
python -m compileall -q api src ui tests
pytest
```

The available tests are contract/unit tests and API boundary tests. The current reconstructed model reproduced trip `00000033` with 5,844 observations, 43,219 candidates, and 5,844 predictions. Streamlit browser interaction was not directly exercised; backend responses and generated map HTML were verified programmatically.

Tests cover the trip, graph, HMM, API contracts, training-route requirements, grouped training, and artifact compatibility. A real smoke run was executed on training trip `00000033`.

## Evaluation status

Results are kept separate by provenance:

1. **Historical experiments:** reports under `docs/experiments/` and `report/` preserve earlier results and are not results from the reconstructed artifact.
2. **Reconstructed baseline grouped validation:** 80 training trips, 1,514,702 candidate rows, zero skipped trips; mean Top-1 approximately `0.844`, mean log loss approximately `0.436`, and mean macro F1 approximately `0.777`.
3. **Reconstructed baseline student-test:** all 10 student-test trips completed; mean arc membership `0.5474`.
4. **Reconstructed baseline evaluator-test:** the supplied 10-trip archive was extracted outside the repository and was not committed. All 10 trips completed successfully.

| Metric | Reconstructed evaluator result |
|---|---:|
| Mean arc-set precision | 0.5396 |
| Mean arc-set recall | 0.9318 |
| Mean arc-set F1 | 0.6748 |
| Mean LCS route recall | 0.9261 |
| Mean processing time | 3.54 seconds |

These reconstructed results must not be presented as recovery of the missing historical model, and no claim is made that HMM decoding improves every accuracy metric.

## Development workflow

`main` is the shared integration branch. Use `member-a-data`, `member-b-ml`, or `member-c-integration` for owned work and submit pull requests into `main`; do not merge another member's work automatically or rewrite history. See `docs/DEVELOPMENT_WORKFLOW.md`.

The intended two-laptop deployment is FastAPI on Laptop 1 and Streamlit/Folium on Laptop 2. Configure the UI API base URL for the actual local network environment; no address is assumed here.

## Current status

Completed: repository foundation, real candidate/features pipeline, approved model inference, directed HMM, API prediction, API-backed Streamlit/Folium integration, grouped model comparison, and 20-trial Optuna tuning. Reports distinguish classifier results, HMM diagnostics, student-test results, and evaluator results. The complete 80-trip OOF HMM aggregate remains a performance limitation documented in `docs/experiments/FINAL_EVALUATION.md`; no universal HMM accuracy improvement is claimed.
