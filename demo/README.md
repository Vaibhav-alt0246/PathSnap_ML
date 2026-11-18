# PathSnap model demonstration

This demonstration loads a generated training candidate table and the
serialized reconstructed HistGradientBoosting model. It scores a small deterministic sample
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
  --dataset /path/to/generated_training_features.parquet \
  --model models/artifacts/hist_gradient_boosting_reconstructed_baseline.joblib \
  --sample-observations 12 \
  --output-dir demo/outputs
```
