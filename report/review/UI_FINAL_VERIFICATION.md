# PathSnap UI and end-to-end verification

## Overall assessment

**READY FOR PROFESSOR REVIEW WITH MANUAL BROWSER VERIFICATION.** The backend and complete inference path were executed on two real training trips. Streamlit startup and HTTP rendering passed. Interactive browser selection and button-click verification were not available in this environment.

## Configuration and launch

Machine-specific paths were removed from application fallbacks. Configure them explicitly:

```bash
export PATHSNAP_DATASET_ROOT="/path/to/Student folder"
export PATHSNAP_MODEL_PATH="/path/to/best_baseline_model.joblib"
```

Launch API:

```uvicorn api.main:app --host 127.0.0.1 --port 8001```

API docs: `http://127.0.0.1:8001/docs`

Launch UI:

```streamlit run ui/app.py --server.headless true --server.port 8502```

UI: `http://localhost:8502`

## Execution flow

Streamlit discovers training trip directories, loads the selected trip without `.route`, generates metric candidates, builds the approved 24 features, loads the configured model, computes probabilities, runs directed HMM decoding, displays diagnostics, and renders a Folium map. The API follows the same path for `POST /predict`.

## Changes made

- Removed hardcoded dataset/model fallback paths from API, UI, and inference.
- Added explicit `PATHSNAP_DATASET_ROOT` and `PATHSNAP_MODEL_PATH` configuration.
- Made `/trips` list the same training partition used by `/predict`.
- Added readable Streamlit inference error handling.
- No model, dataset, ground-truth file, or existing prediction output was intentionally changed.

## Verification

| Check | Status | Evidence |
|---|---|---|
| Configuration required | PASS | Missing configuration produces clear UI/API errors |
| Real trip 00000033 | PASS | 5,844 observations; 43,219 candidates; 5,844 outputs; zero missing |
| Real trip 00000034 | PASS | 3,614 observations; 27,658 candidates; 3,614 outputs; zero missing |
| Prediction uses .route | PASS | `include_route=False`; no reference route in inference |
| API /health | PASS | model_loaded=true, inference_ready=true |
| API /trips | PASS | Training trip IDs returned |
| API valid /predict | PASS | Trip 00000034 returned 3,614 predictions |
| API invalid trip | PASS | HTTP 400 with missing-file detail |
| Streamlit startup | PASS | HTTP 200 at localhost:8502 |
| Folium maps | PASS | Two real HTML maps generated with GPS and predicted-route layers |
| Browser interaction | NOT VERIFIED | No browser automation available |
| Switching trips in browser | NOT VERIFIED | Manual browser check required |
| Full tests | PASS | 28 passed |
| Compilation | PASS | compileall completed |

Generated maps:

- `report/review/00000033_e2e_map.html`
- `report/review/00000034_e2e_map.html`

## Manual demonstration

1. Set the two environment variables.
2. Start the API on port 8001.
3. Start Streamlit on port 8502.
4. Open `http://localhost:8502`.
5. Select trip `00000033`, click **Predict route**, and confirm diagnostics and map.
6. Select `00000034), click **Predict route**, and confirm the observation count changes.
7. Open `http://127.0.0.1:8001/docs` and test `/health`, `/trips`, and `/predict`.
8. Confirm no reference route is shown in prediction mode.

## Remaining limitations

The application does not have browser-level automated verification in this environment. The professor-facing review should include the manual interaction sequence above. Existing model warnings about feature names are non-fatal; predictions and tests completed successfully.

