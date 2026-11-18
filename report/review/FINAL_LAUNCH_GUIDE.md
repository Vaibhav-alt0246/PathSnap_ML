# Final launch guide

## Laptop 1 — API and evaluator data

    cd /Users/vaby/Desktop/PathSnap_ML
    export PATHSNAP_DATASET_ROOT="/Users/vaby/Downloads/Student folder"
    export PATHSNAP_MODEL_PATH="/tmp/pathsnap-phase2-artifacts/best_baseline_model.joblib"
    export PATHSNAP_EVALUATOR_DATASET_ROOT="$HOME/Downloads/PathSnap_Evaluation/evaluator_test"
    .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Verify with curl http://LAPTOP1_IP:8000/health.

## Laptop 2 — Streamlit

    cd /Users/vaby/Desktop/PathSnap_ML
    export PATHSNAP_API_BASE_URL="http://LAPTOP1_IP:8000"
    .venv/bin/python -m streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501

Open http://LAPTOP2_IP:8501. Evaluator route files stay on Laptop 1.

