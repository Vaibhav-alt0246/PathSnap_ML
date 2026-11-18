# Real-data baseline model comparison

This experiment used the recovered Parquet dataset with SHA-256
`72db0f90fb1cf19a34bc7e125c945b5ef72d11f4aba1c26bc8bb0e873543d3e4`, 24
approved features, seed `42`, and identical 5-fold `GroupKFold` splits by
`trip_id`. Outputs were written outside the repository to
`/tmp/pathsnap-phase2-results` and the selected artifact to
`/tmp/pathsnap-phase2-artifacts`.

| Model | Top-1 mean | Top-1 std | Log loss | Macro F1 | Candidate accuracy | Candidate recall | Inference s/fold | Train s/fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.832225 | 0.039348 | 0.181852 | 0.854653 | 0.927576 | 0.915166 | 0.130099 | 4.030917 |
| Random Forest | 0.831020 | 0.041599 | 0.156590 | 0.891398 | 0.955581 | 0.773435 | 0.231864 | 29.055427 |
| Logistic Regression | 0.830633 | 0.038931 | 0.261245 | 0.785282 | 0.877433 | 0.920903 | 0.016476 | 0.726038 |
| Gaussian Naive Bayes | 0.808701 | 0.039959 | 0.919605 | 0.767459 | 0.862671 | 0.914046 | 0.034740 | 0.205764 |
| K-Nearest Neighbors | 0.783792 | 0.042154 | 0.845744 | 0.856677 | 0.939976 | 0.737093 | 62.909941 | 0.117599 |

HistGradientBoosting ranked first by the existing selection rule: observation
Top-1 accuracy, then log loss, macro F1, fold stability, and inference time.
This is a real held-out-trip comparison, not a synthetic test. It is not yet
an end-to-end map-matching result because raw trip files and a trained-model
inference pipeline are not available in the integration checkout.

The serialized artifact was reloaded in a fresh Python process. Predictions on
the same 1,000-row sample matched exactly (`max absolute difference = 0.0`).
