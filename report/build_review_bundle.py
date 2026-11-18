"""Build professor-facing review artifacts from verified local sources."""
from pathlib import Path
import json, shutil, hashlib
import pandas as pd

ROOT = Path(__file__).parents[1]
OUT = ROOT / "report" / "review"; OUT.mkdir(parents=True, exist_ok=True)
PARQUET = Path("/tmp/pathsnap-forensic-audit/member-a/member_a/results/training_candidates_features.parquet")
RESULTS = Path("/tmp/pathsnap-phase2-results")
config = json.loads((ROOT / "models/config/feature_columns.json").read_text())
df = pd.read_parquet(PARQUET)
df.head(30).to_csv(OUT/"training_candidates_sample.csv", index=False)
summary = {
 "training_trips": int(df.trip_id.nunique()),
 "observations": int(df[["trip_id","point_idx"]].drop_duplicates().shape[0]),
 "candidate_rows": int(len(df)), "features": len(config["feature_columns"]),
 "class_distribution": {str(k): int(v) for k,v in df.label.value_counts().sort_index().items()},
 "missing_values_total": int(df.isna().sum().sum()),
 "source_file": str(PARQUET), "source_size_bytes": PARQUET.stat().st_size,
 "source_sha256": hashlib.sha256(PARQUET.read_bytes()).hexdigest()}
(OUT/"dataset_summary.json").write_text(json.dumps(summary, indent=2))
desc = {
"distance_m":"Perpendicular GPS-to-candidate segment distance in metres.",
"distance_to_closest_cand":"Candidate distance minus the closest candidate distance.",
"distance_ratio_closest":"Candidate distance divided by closest distance with stabilizer and clip.",
"dist_rank":"Distance rank within observation candidates; zero is closest.",
"cand_count_for_point":"Number of candidates retained for the observation.",
"projection_fraction":"Clamped fractional position of GPS projection along the segment.",
"dist_to_segment_start_m":"Projected distance from segment start in metres.",
"dist_to_segment_end_m":"Projected distance from segment end in metres.",
"arc_length_m":"Metric length of the candidate road arc.",
"gps_heading_deg":"GPS trajectory compass heading in degrees.",
"road_bearing_deg":"Directed candidate road bearing in degrees.",
"heading_bearing_diff_deg":"Smallest absolute GPS-heading/road-bearing difference.",
"heading_bearing_cos_sim":"Cosine similarity of GPS heading and road bearing.",
"is_low_speed":"Indicator for stationary, low-speed, or gap observations.",
"gps_delta_t_s":"Elapsed time from previous GPS observation.",
"gps_step_distance_m":"Metric displacement from previous GPS observation.",
"gps_speed_mps":"GPS step speed in metres per second.",
"gps_accel_mps2":"Change in GPS speed per second.",
"speed_ratio_to_mean":"GPS speed divided by trip mean speed.",
"u_in_degree":"Incoming directed-arc count at candidate start node.",
"u_out_degree":"Outgoing directed-arc count at candidate start node.",
"v_in_degree":"Incoming directed-arc count at candidate end node.",
"v_out_degree":"Outgoing directed-arc count at candidate end node.",
"has_reverse_twin":"Indicator that the reverse directed arc exists."}
pd.DataFrame({"feature_name": config["feature_columns"], "description":[desc[x] for x in config["feature_columns"]]}).to_csv(OUT/"feature_dictionary.csv", index=False)
shutil.copy2(RESULTS/"baseline_model_comparison.csv", OUT/"model_comparison.csv")
for name in ["oof_aggregate.csv","oof_per_trip.csv","student_test_evaluation_corrected.csv"]:
 shutil.copy2(ROOT/"report/evaluation"/name, OUT/name)
shutil.copy2(ROOT/"report/evaluation/hmm_benchmarks.json", OUT/"hmm_benchmarks.json")
guide = """# PathSnap review guide

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
"""
(OUT/"REVIEW_GUIDE.md").write_text(guide)
print("created", len(list(OUT.iterdir())), "files in", OUT)

