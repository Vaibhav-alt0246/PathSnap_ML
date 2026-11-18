# PathSnap model demonstration

This demonstration loads the recovered training candidate table and the
serialized HistGradientBoosting model. It scores a small deterministic sample
of real candidate rows and writes:

- `outputs/candidate_predictions.csv`
- `outputs/confusion_matrix.png`

The reported metrics are sample candidate-classification metrics only. They are
not independent test-set results and do not represent complete GPS-to-road
map matching or HMM route decoding.

Run from the repository root:

```bash
python demo/demo_model_predictions.py
```

Optional paths:

```bash
python demo/demo_model_predictions.py \
  --dataset /tmp/pathsnap-forensic-audit/member-a/member_a/results/training_candidates_features.parquet \
  --model /tmp/pathsnap-phase2-artifacts/best_baseline_model.joblib \
  --sample-observations 12 \
  --output-dir demo/outputs
```
