from pathlib import Path
import json, shutil
import pandas as pd
from sklearn.inspection import permutation_importance
import joblib

root=Path("/Users/vaby/Desktop/PathSnap_ML")
out=root/"report/performance_improvement"; out.mkdir(parents=True,exist_ok=True)
source=Path("/tmp/pathsnap-forensic-audit/member-a/member_a/results/training_candidates_features.parquet")
config=json.loads((root/"models/config/feature_columns.json").read_text())
df=pd.read_parquet(source)
model_artifact=joblib.load("/tmp/pathsnap-phase2-artifacts/best_baseline_model.joblib")
model=model_artifact.get("model_pipeline",model_artifact)
features=config["feature_columns"]
sample=df.sample(n=min(10000,len(df)),random_state=42)
perm=permutation_importance(model,sample[features],sample.label,n_repeats=3,random_state=42,scoring="neg_log_loss",n_jobs=-1)
pd.DataFrame({"feature":features,"importance_mean":perm.importances_mean,"importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False).to_csv(out/"FEATURE_IMPORTANCE.csv",index=False)
for src,dst in [
 (Path("/tmp/pathsnap-phase2-results/baseline_model_comparison.csv"),"MODEL_COMPARISON.csv"),
 (Path("/tmp/pathsnap-phase2-results/candidate_recall_diagnostics.csv"),"CANDIDATE_RECALL_ANALYSIS.csv"),
 (Path("/tmp/pathsnap-optuna-20/optuna_trials.csv"),"OPTUNA_TRIALS.csv"),
 (root/"report/evaluation/oof_per_trip.csv","PER_TRIP_ERROR_ANALYSIS.csv"),
 (root/"report/evaluation/oof_aggregate.csv","HMM_EXPERIMENT_COMPARISON.csv")]:
 shutil.copy2(src,out/dst)
(out/"experiment_config.json").write_text(json.dumps({"dataset":str(source),"dataset_sha256":config["dataset_sha256"],"features":features,"group_split":"GroupKFold(n_splits=5)","seed":42,"permutation_sample_rows":len(sample),"note":"No evaluator or student-test data used."},indent=2))
(out/"BASELINE_REPRODUCTION.md").write_text("""# Baseline reproduction

The preserved baseline comparison is the existing five-model, five-fold GroupKFold
by trip experiment. HistGradientBoosting Top-1 is 0.832225, log loss 0.181852 and
macro F1 0.854653. Random Forest has better recorded log loss and macro F1, so
HistGradientBoosting is not claimed to win every metric. The selected artifact was
not replaced. Source values are copied from the preserved phase-2 result files.
""")
(out/"FINAL_EXPERIMENT_REPORT.md").write_text("""# Performance improvement report

The preserved baseline and 20-trial Optuna experiment were audited. The Optuna
study improved Top-1 from 0.832225 to 0.836964 but worsened mean log loss from
0.181852 to 0.192618, so the untuned baseline remains preferable for probability
quality and was not replaced.

Candidate diagnostics and feature permutation importance are provided in the CSV
files. The OOF HMM completed all 80 trips: classifier route-membership diagnostic
0.975598 versus HMM 0.555742, while continuity was 0.332611 versus 0.924116.
Thus HMM improves continuity but not the recorded membership diagnostic.

No evaluator or student-test data was used for model development. No reliable new
production improvement was demonstrated, so no model or feature configuration was
changed. Further candidate-radius experiments should be run as separate GroupKFold
studies before changing inference defaults.
""")
print(out)

