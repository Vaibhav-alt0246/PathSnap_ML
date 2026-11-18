# Final evaluation status

## Cross-validation

The previously completed 5-fold GroupKFold experiment used all 80 training trips
with trip-disjoint validation. HistGradientBoosting achieved grouped classifier
Top-1 accuracy `0.832225` and mean log loss `0.181852`.

The new raw-trip OOF/HMM runner is implemented in
`experiments/final_evaluation.py`, but the complete 80-trip HMM run was not
completed: graph shortest-path decoding was too slow at the current candidate
volume. Three partial trip files were produced and are not used as aggregate
results. No HMM improvement claim is made.

## Student test

All 10 student-test trips produced predictions before reference routes were read.
Prediction files and maps are under `report/evaluation/student_test_predictions/`.
Separate route evaluation produced `report/evaluation/student_test_evaluation.csv`.
The original positional sequence-similarity mean was `0.000459`. That metric is
`sum(pred[i] == route[i] for i in zip(pred, route)) / max(len(pred), len(route))`.
It compares one prediction per GPS observation with a route containing each road
arc once, so repeated observations and unequal sequence lengths make it unsuitable
as the primary route metric. The original file is preserved unchanged.

An independent corrected audit is in
`report/evaluation/student_test_evaluation_corrected.csv`. Mean metrics are:

- Arc-set precision: `0.495066`
- Arc-set recall: `0.919452`
- Arc-set F1: `0.637769`
- LCS route recall: `0.913063`

The trip-local arc IDs were preserved: predicted IDs are looked up directly against
each trip's local `.route` IDs, with no global remapping. The low original score is
therefore a metric-definition issue, compounded by repeated per-observation arc
predictions; it is not evidence of an ID offset.

## Runtime verification

Streamlit started successfully with `streamlit run ui/app.py --server.headless true
--server.port 8502` and returned HTTP 200 at `http://localhost:8502`. Browser
interaction and visual widget clicking were not available in this environment.

## HMM performance audit

The decoder caches graph endpoints and one single-source Dijkstra result per
reachable previous endpoint, avoiding repeated NetworkX traversal inside each state
pair. Measured benchmarks on trip `00000033` were:

| observations | candidates | decode seconds | seconds/observation | peak traced memory |
|---:|---:|---:|---:|---:|
| 100 | 1,168 | 1.621 | 0.01621 | 18.9 MB |
| 250 | 2,705 | 4.271 | 0.01708 | 52.1 MB |
| 1,000 | 11,143 | 8.696 | 0.00870 | 68.0 MB |

The optimized sparse decoder completed the full 80-trip OOF run. Every trip appears
exactly once, with 16 validation trips per fold. Results are in
`report/evaluation/oof_per_trip.csv` and `oof_aggregate.csv`; fold candidate
probabilities are in `oof_candidates_fold_0.csv` through `_4.csv`.

The current evaluator's route-membership diagnostics average:

- Classifier top-1 route membership: `0.975598`
- HMM top-1 route membership: `0.555742`
- Classifier continuity ratio: `0.332611`
- HMM continuity ratio: `0.924116`

These results do not support claiming that this HMM configuration improves
classifier accuracy. The HMM strongly improves continuity but currently lowers the
route-membership diagnostic, consistent with its distance penalty and fallback
behavior. The legacy positional route-similarity fields remain in the raw CSV for
traceability and must not be used as the primary route metric.

Benchmark data is preserved in `report/evaluation/hmm_benchmarks.json`.
