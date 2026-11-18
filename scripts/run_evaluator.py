"""Run the authorized server-side evaluator and save aggregate metrics."""
import json, os, time
from pathlib import Path
import pandas as pd
import requests

OUT = Path("report/evaluation/evaluator_test_results.csv")
os.environ.setdefault("PATHSNAP_EVALUATOR_DATASET_ROOT", "/Users/vaby/Downloads/PathSnap_Evaluation/evaluator_test")
base = os.environ.get("PATHSNAP_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
trip_ids = requests.get(f"{base}/evaluation/trips", timeout=30).json()["trips"]
rows=[]
for trip_id in trip_ids:
    started=time.perf_counter(); response=requests.post(f"{base}/evaluation/predict", json={"trip_id":trip_id}, timeout=600)
    if response.status_code != 200:
        rows.append({"trip_id":trip_id,"status":response.status_code,"error":response.text}); continue
    payload=response.json(); metrics=payload["evaluation"]
    rows.append({"trip_id":trip_id,"status":200,"processing_seconds":payload["processing_seconds"],**metrics})
result = pd.DataFrame(rows)
result.to_csv(OUT,index=False)
result[result.status == 200].mean(numeric_only=True).to_frame("mean").to_csv(OUT.with_name("evaluator_test_aggregate.csv"))
print(json.dumps({"trip_count":len(trip_ids),"successful":sum(r["status"]==200 for r in rows),"output":str(OUT)},indent=2))
