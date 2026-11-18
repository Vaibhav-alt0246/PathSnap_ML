# Reconstructed baseline model

The historical `best_baseline_model.joblib` was unavailable and was not recovered from Git history or local artifacts. PathSnap now includes a new, explicitly labeled reconstruction at:

`models/artifacts/hist_gradient_boosting_reconstructed_baseline.joblib`

It was generated on 18 November 2026 from the local `train` partition supplied at `/Users/vaby/Downloads/Student folder` using all 80 valid training trips. The source dataset is not committed. The resulting feature table contained 1,514,702 candidate rows; no trips were skipped. Its SHA-256 dataset fingerprint is recorded in the artifact metadata.

The reconstruction uses the existing pipeline: 50 m candidate radius, maximum 15 candidates, nearest-candidate fallback, the configured 24 features, and this baseline classifier:

```text
HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=120, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=0.1, random_state=42
)
```

Grouped 5-fold validation was run by trip. The fold metrics are in `models/artifacts/reconstructed_baseline_grouped_metrics.csv`; their mean is approximately Top-1 0.844, log loss 0.436, and macro F1 0.777. These are new reconstruction results and are not historical-model results.

Rebuild with:

```bash
python scripts/train_baseline.py \
  --dataset-root /absolute/path/to/dataset \
  --output-dir models/artifacts
```
