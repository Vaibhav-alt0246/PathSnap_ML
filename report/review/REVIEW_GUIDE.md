# PathSnap review guide

This folder contains verified review artifacts generated from local PathSnap sources.

## Dataset

The training candidate table has one row per (trip_id, point_idx, arc_id) candidate, 80 training trips, grouped GPS observations, 24 approved numeric features, and a binary label. Raw trip files remain outside the repository under the documented Downloads dataset root.

## Files

- training_candidates_sample.csv: 30 actual Parquet rows with identifiers, features, and label.
- dataset_summary.json: measured counts, classes, quality checks, and source SHA-256.
- feature_dictionary.csv: exact configured features described from src/features/engineering.py.
- model_comparison.csv: existing five-model GroupKFold comparison.
- oof_aggregate.csv and oof_per_trip.csv: completed 80-trip five-fold held-out evaluation.
- student_test_evaluation_corrected.csv: corrected independent student-test metrics, not used for tuning.
- hmm_benchmarks.json: measured decoder timings.

## Pipeline and leakage prevention

Raw GPS and directed trip-local arcs are projected to metric coordinates. Candidates are generated geometrically, features are computed without .route, and the classifier emits candidate probabilities. The directed HMM uses graph connectivity, transition distances, and probabilities. GroupKFold by trip keeps validation trips out of training. Reference routes are accessed only by evaluation after prediction.

## Findings and demonstration

HistGradientBoosting was selected from five grouped models. The completed OOF HMM improved continuity but lowered the current route-membership diagnostic, so no overall accuracy improvement is claimed. Demonstrate by reviewing the sample, feature dictionary, model comparison, OOF tables, student metrics, maps, tests, Streamlit, and report/PathSnap_Final_Report.pdf.
