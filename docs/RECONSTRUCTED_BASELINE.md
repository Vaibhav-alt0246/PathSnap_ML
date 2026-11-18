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

## Evaluation separation

The repository distinguishes four result groups:

1. **Historical experiments:** earlier reports under `docs/experiments/` and `report/`; these are separate from the reconstructed baseline.
2. **Reconstructed baseline grouped validation:** 80 training trips, 1,514,702 rows, zero skipped trips; mean Top-1 approximately `0.844`, mean log loss approximately `0.436`, and mean macro F1 approximately `0.777`.
3. **Reconstructed baseline student-test:** all 10 student-test trips completed, with mean arc membership `0.5474`.
4. **Reconstructed baseline evaluator-test:** the supplied `evaluator_test.zip` was extracted outside the repository and was not committed. All 10 evaluator trips completed:

| Metric | Result |
|---|---:|
| Mean arc-set precision | 0.5396 |
| Mean arc-set recall | 0.9318 |
| Mean arc-set F1 | 0.6748 |
| Mean LCS route recall | 0.9261 |
| Mean processing time | 3.54 seconds |

These evaluator values are reconstructed-model results, not historical evaluator metrics. Streamlit browser rendering was not directly exercised; API responses and generated map HTML were verified programmatically.

Rebuild with:

```bash
python scripts/train_baseline.py \
  --dataset-root /absolute/path/to/dataset \
  --output-dir models/artifacts
```
