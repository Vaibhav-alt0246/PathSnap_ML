# Final submission audit

| Requirement | Evidence | Status | Action |
|---|---|---|---|
| Data exploration and preprocessing | `src/data/loader.py`, review dataset summary | PASS | None |
| Feature engineering | `src/features/engineering.py`, 24-feature config | PASS | None |
| Five models | `/tmp/pathsnap-phase2-results/baseline_model_comparison.csv` | PASS | Keep external artifact available for review |
| CV metrics | Existing model comparison and OOF files | PASS | None |
| Student-test metrics for every model | Existing comparison does not contain five-model student-test metrics | MISSING | Run authorized student-test comparison separately |
| Bayesian optimization | `experiments/optuna_tuning.py`, Optuna outputs | PASS | None |
| HMM/Viterbi | `src/hmm/decoder.py` | PASS | None |
| OOF classifier/HMM | `report/evaluation/oof_per_trip.csv`, 80 trips | PASS | Interpret HMM diagnostics honestly |
| Student-test prediction/evaluation | `report/evaluation/student_test_predictions`, corrected CSV | PASS | Not used for tuning |
| Evaluator mode | `/evaluation/trips`, `/evaluation/predict`, `/evaluation/map` | IMPLEMENTED | Requires evaluator archive extracted on Laptop 1 |
| Ground-truth separation | Prediction uses `include_route=False`; metrics loaded afterward | PASS | None |
| Two-laptop configuration | `PATHSNAP_API_BASE_URL`, dataset/model/evaluator variables | PASS | Configure environment on both laptops |
| Folium maps | `src/visualization/mapping.py`, report maps | PASS | None |
| FastAPI/UI | `api/main.py`, `ui/app.py` | PASS | Manual browser test remains |
| Requirements and README | `requirements.txt`, `README.md` | PASS | None |
| Consolidated technical report | Existing PDF is concise one-page report | PARTIAL | Expand with full tables/graphs/maps if required by official rubric |

## Evaluator deployment

Laptop 1 must set:

```bash
export PATHSNAP_DATASET_ROOT="/path/to/Student folder"
export PATHSNAP_MODEL_PATH="/path/to/best_baseline_model.joblib"
export PATHSNAP_EVALUATOR_DATASET_ROOT="/path/to/evaluator_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 must set:

```bash
export PATHSNAP_API_BASE_URL="http://LAPTOP1_LAN_IP:8000"
streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

The UI evaluator mode never opens evaluator files or receives reference routes. The server returns aggregate metrics only.

## Remaining evidence gaps

The five-model comparison currently documents grouped validation metrics but not a complete five-model student-test metric table. The final PDF is an overview rather than a full consolidated technical report with every requested graph and table. These are evidence gaps, not fabricated claims.

