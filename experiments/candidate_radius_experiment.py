from pathlib import Path
import json,time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, f1_score
from src.data.loader import load_trip
from src.candidates.generator import generate_candidates
from src.features.engineering import build_features

ROOT=Path("/Users/vaby/Downloads/Student folder"); OUT=Path("report/performance_improvement/candidate_radius_experiment"); ART=Path("/tmp/pathsnap-radius-experiment"); OUT.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
FEATURES=json.loads(Path("models/config/feature_columns.json").read_text())["feature_columns"]
trip_ids=sorted(p.name for p in (ROOT/"train").iterdir() if p.is_dir())
radii=[30,50,75,100,150]
def top1(frame,prob):
 d=frame[["trip_id","point_idx","label"]].copy(); d["p"]=prob; vals=[]
 for _,g in d.groupby(["trip_id","point_idx"]):
  if g.label.sum(): vals.append(int(g.iloc[g.p.to_numpy().argmax()].label==1))
 return float(np.mean(vals)) if vals else 0.
all_fold=[]; all_trip=[]; summaries=[]
for radius in radii:
 started=time.perf_counter(); frames=[]; recall_rows=[]; fallback=0
 cache=ART/f"candidates_{radius}m.parquet"
 if cache.exists(): table=pd.read_parquet(cache); gen_time=0
 else:
  for tid in trip_ids:
   trip=load_trip(ROOT,"train",tid,include_route=False); route=load_trip(ROOT,"train",tid,include_route=True).route.astype(int); route_set=set(route)
   t0=time.perf_counter(); c=generate_candidates(trip,search_radius_m=radius,max_candidates=15,fallback_nearest=True); gen_time=time.perf_counter()-t0
   fallback += int((c.groupby("point_idx").distance_m.min()>radius).sum())
   recall_rows.extend([{"radius_m":radius,"trip_id":tid,"point_idx":int(i),"candidate_recall":int(any(int(a) in route_set for a in g.arc_id)),"fallback_used":int(g.distance_m.min()>radius)} for i,g in c.groupby("point_idx")])
   f=build_features(trip,c); f["trip_id"]=tid; f["label"]=f.arc_id.astype(int).isin(route_set).astype(int); frames.append(f)
  table=pd.concat(frames,ignore_index=True); table.to_parquet(cache,index=False); pd.DataFrame(recall_rows).to_csv(OUT/f"recall_{radius}m.csv",index=False)
  gen_time=time.perf_counter()-started
 if not (OUT/f"recall_{radius}m.csv").exists():
  # recover recall from generated table and route labels
  rows=[]
  for (tid,point),g in table.groupby(["trip_id","point_idx"]):
   rows.append({"radius_m":radius,"trip_id":tid,"point_idx":point,"candidate_recall":int(g.label.any()),"fallback_used":int(g.distance_m.min()>radius)})
  pd.DataFrame(rows).to_csv(OUT/f"recall_{radius}m.csv",index=False)
 recall=pd.read_csv(OUT/f"recall_{radius}m.csv")
 groups=table.trip_id.astype(str); splitter=GroupKFold(5); fold_times=[]
 for fold,(tr,va) in enumerate(splitter.split(table,table.label,groups)):
  t0=time.perf_counter(); clf=HistGradientBoostingClassifier(class_weight="balanced",random_state=42,max_iter=100,max_leaf_nodes=31,min_samples_leaf=20)
  clf.fit(table.iloc[tr][FEATURES],table.iloc[tr].label); probs=clf.predict_proba(table.iloc[va][FEATURES])[:,1]; valid=table.iloc[va].copy()
  fold_times.append(time.perf_counter()-t0); all_fold.append({"radius_m":radius,"fold":fold,"top1_accuracy":top1(valid,probs),"macro_f1":f1_score(valid.label,probs>=.5,average="macro"),"log_loss":log_loss(valid.label,probs,labels=[0,1]),"train_seconds":fold_times[-1]})
  for tid in valid.trip_id.unique():
   mask=valid.trip_id==tid; all_trip.append({"radius_m":radius,"fold":fold,"trip_id":tid,"top1_accuracy":top1(valid.loc[mask],probs[mask.to_numpy()])})
 summaries.append({"radius_m":radius,"candidate_rows":len(table),"mean_candidates_per_observation":len(table)/table[["trip_id","point_idx"]].drop_duplicates().shape[0],"median_candidates":float(table.groupby(["trip_id","point_idx"]).size().median()),"max_candidates":int(table.groupby(["trip_id","point_idx"]).size().max()),"candidate_recall":float(recall.candidate_recall.mean()),"fallback_observations":int(recall.fallback_used.sum()),"generation_seconds":gen_time,"total_seconds":time.perf_counter()-started})
pd.DataFrame(summaries).to_csv(OUT/"radius_comparison.csv",index=False); pd.DataFrame(all_fold).to_csv(OUT/"radius_fold_results.csv",index=False); pd.DataFrame(all_trip).to_csv(OUT/"candidate_recall_per_trip.csv",index=False)
(OUT/"experiment_config.json").write_text(json.dumps({"radii_m":radii,"max_candidates":15,"folds":5,"group":"trip_id","features":FEATURES,"training_root":str(ROOT),"artifact_cache":str(ART)},indent=2))
(OUT/"CANDIDATE_RADIUS_EXPERIMENT.md").write_text("# Candidate radius experiment\n\nThis experiment regenerates candidates/features from the 80 training trips for each radius, with 15 maximum candidates and five GroupKFold splits. Each fold retrains HistGradientBoosting; no student or evaluator data is used. See the CSV outputs for measured results. Production defaults were not changed.\n")
print(pd.DataFrame(summaries).to_string(index=False))

