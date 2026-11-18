# Candidate radius experiment

# Candidate radius experiment

Candidates and features were regenerated from all 80 training trips for each radius
with `max_candidates=15`. Each radius used the same five GroupKFold-by-trip split
structure and retrained HistGradientBoosting inside every training fold. Student and
evaluator data were not used. Existing production defaults and artifacts were not
changed.

The route-membership recall proxy is the percentage of observations with at least
one generated candidate whose arc ID appears in the training reference route. It is
an offline candidate-coverage diagnostic, not a replacement for exact route
alignment. Fallback usage records observations whose nearest candidate was outside
the requested radius.

| Radius | Candidate recall | Mean candidates | Candidate rows | Fallback observations | Top-1 | Macro F1 | Log loss | Total seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 m | 0.890623 | 5.139 | 1,070,996 | 20,280 | 0.854604 | 0.783981 | 0.457268 | 45.20 |
| 50 m | 0.894414 | 7.268 | 1,514,702 | 18,230 | 0.844538 | 0.776969 | 0.461433 | 54.53 |
| 75 m | 0.895891 | 9.081 | 1,892,614 | 13,604 | 0.837201 | 0.769394 | 0.467404 | 63.89 |
| 100 m | 0.897408 | 10.389 | 2,165,180 | 1,413 | 0.835756 | 0.768583 | 0.464957 | 74.92 |
| 150 m | 0.934175 | 12.310 | 2,565,567 | 198 | 0.803020 | 0.753707 | 0.473990 | 98.17 |

Recommendation: retain the 50 m production default. The 150 m radius raises
candidate coverage but adds substantial candidate volume and lowers every measured
classifier metric. The 30 m radius has the highest measured Top-1 in this controlled
run but lower coverage and much more fallback usage; it should not replace 50 m
without a separate route-level and inference robustness review.
