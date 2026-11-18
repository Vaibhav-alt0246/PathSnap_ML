# Evaluator execution report

The evaluator API was launched with the exact configured evaluator path. GET /evaluation/trips returned exactly 10 trips. All 10 evaluator predictions returned HTTP 200. Prediction loaded only track, nodes, and arcs; route metrics were calculated afterward on the server. No route IDs were returned to the UI.

Per-trip output: report/evaluation/evaluator_test_results.csv
Aggregate output: report/evaluation/evaluator_test_aggregate.csv

Mean metrics:

- Arc-set precision: 0.532459
- Arc-set recall: 0.911916
- Arc-set F1: 0.663888
- LCS route recall: 0.905942
- Mean processing time: 3.685 seconds

