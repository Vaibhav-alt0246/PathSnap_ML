"""Fair grouped evaluation and student-test prediction runner."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support
from src.data.loader import load_trip
from src.pipeline.inference import predict_trip
from src.candidates.generator import generate_candidates
from src.features.engineering import build_features
from src.models.prediction import predict_proba
from src.hmm.decoder import decode
from src.visualization.mapping import create_map

DATA = Path('/tmp/pathsnap-forensic-audit/member-a/member_a/results/training_candidates_features.parquet')
FEATURES = json.loads(Path('models/config/feature_columns.json').read_text())['feature_columns']
ROOT = Path('/Users/vaby/Downloads/Student folder')
OUT = Path('report/evaluation'); OUT.mkdir(parents=True, exist_ok=True)

def model():
    return HistGradientBoostingClassifier(learning_rate=.05, max_iter=120, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42)

def sequence_similarity(pred, ref):
    if not ref: return 0.0
    p, r = list(map(int, pred)), list(map(int, ref)); matches = sum(a == b for a, b in zip(p, r))
    return matches / max(len(p), len(r))

def run_oof():
    df = pd.read_parquet(DATA); groups = df.trip_id.astype(str)
    splitter = GroupKFold(5); rows=[]; trip_results=[]
    for fold, (train_i, test_i) in enumerate(splitter.split(df, df.label, groups)):
        clf=model(); clf.fit(df.iloc[train_i][FEATURES], df.iloc[train_i].label)
        test=df.iloc[test_i].copy(); test['probability']=clf.predict_proba(test[FEATURES])[:,1]
        test.to_csv(OUT/f'oof_candidates_fold_{fold}.csv', index=False)
        for tid in sorted(test.trip_id.astype(str).unique()):
            trip=load_trip(ROOT,'train',tid,include_route=False); c=generate_candidates(trip); f=build_features(trip,c); p=predict_proba(f,clf,FEATURES)
            c=c.copy(); c['probability']=p; best=c.loc[c.groupby('point_idx').probability.idxmax()].sort_values('point_idx')
            hmm=decode(trip,c,p)
            route=load_trip(ROOT,'train',tid,include_route=True).route.astype(int).tolist()
            pred=best.arc_id.astype(int).tolist(); continuity=lambda x: sum(1 for a,b in zip(x,x[1:]) if a!=b)/max(1,len(x)-1)
            route_set=set(route); truth=[int(a) for a in c.loc[c.point_idx.isin(best.point_idx) & c.arc_id.isin(route_set)].groupby('point_idx').arc_id.first()]
            trip_results.append({'trip_id':tid,'fold':fold,'observations':len(trip.track),'candidate_rows':len(c),
                'classifier_top1_route_membership':len(truth)/max(1,len(pred)),'hmm_top1_route_membership':sum(int(a in route_set) for a in hmm)/max(1,len(hmm)),
                'classifier_route_similarity':sequence_similarity(pred,route),'hmm_route_similarity':sequence_similarity(hmm,route),
                'classifier_continuity':continuity(pred),'hmm_continuity':continuity(hmm),
                'hmm_arc_count':len(hmm),'reference_arc_count':len(route)})
            pd.DataFrame({'point_idx':best.point_idx,'classifier_arc_id':pred}).to_csv(OUT/f'{tid}_oof_predictions.csv',index=False)
    result=pd.DataFrame(trip_results); result.to_csv(OUT/'oof_per_trip.csv',index=False); result.mean(numeric_only=True).to_frame('mean').to_csv(OUT/'oof_aggregate.csv')
    return result

def run_student():
    out=OUT/'student_test_predictions'; out.mkdir(exist_ok=True); ids=sorted(p.name for p in (ROOT/'student_test').iterdir() if p.is_dir()); rows=[]
    for tid in ids:
        trip=load_trip(ROOT,'student_test',tid,include_route=False); result=predict_trip(trip); pd.DataFrame(result['candidates']).to_csv(out/f'{tid}_candidates.csv',index=False); (out/f'{tid}_arcs.json').write_text(json.dumps({'trip_id':tid,'predicted_arc_ids':result['predicted_arc_ids'],'classifier_arc_ids':result['classifier_arc_ids'],'diagnostics':result['diagnostics']})); create_map(trip,result['predicted_arc_ids']).save(str(out/f'{tid}_map.html')); rows.append({'trip_id':tid,'observations':len(trip.track),'predicted_arcs':len(result['predicted_arc_ids']),'missing_candidates':len(result['diagnostics']['missing_candidate_observations'])})
    pd.DataFrame(rows).to_csv(OUT/'student_test_prediction_summary.csv',index=False)
    # Evaluation is intentionally separate and only after all prediction files exist.
    eval_rows=[]
    for tid in ids:
        route=pd.read_csv(ROOT/'student_test'/tid/f'{tid}.route', sep=r'\s+', header=None, names=['arc_id']).arc_id.astype(int).tolist(); pred=json.loads((out/f'{tid}_arcs.json').read_text())['predicted_arc_ids']; eval_rows.append({'trip_id':tid,'route_similarity':sequence_similarity(pred,route),'predicted_count':len(pred),'reference_count':len(route)})
    pd.DataFrame(eval_rows).to_csv(OUT/'student_test_evaluation.csv',index=False)

if __name__=='__main__':
    run_oof(); run_student()
