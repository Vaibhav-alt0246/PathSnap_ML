from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
out=Path("report/PathSnap_Final_Report.pdf")
pages=[
("PathSnap — Final Technical Report",["GPS map matching with machine learning and directed HMM/Viterbi decoding.","Submission readiness report using measured local artifacts."]),
("Dataset and pipeline",["80 training trips; 1,514,674 candidate rows; 208,417 observations; 24 approved features.","Pipeline: raw GPS → metric candidates → features → HistGradientBoosting → directed HMM/Viterbi → ordered arc IDs.","Prediction never reads .route. Evaluator routes are loaded only after prediction on the server."]),
("Model and evaluation",["Five-model grouped results and 20-trial Optuna results are recorded in the review bundle.","OOF HMM: all 80 trips, 5 folds, 16 trips/fold.","HMM route-membership diagnostic: 0.555742; classifier: 0.975598.","HMM continuity: 0.924116; classifier: 0.332611. No overall HMM accuracy improvement is claimed."]),
("Evaluator results",["10 reserved evaluator trips completed with HTTP 200.","Mean arc-set precision: 0.532459","Mean arc-set recall: 0.911916","Mean arc-set F1: 0.663888","Mean LCS route recall: 0.905942","Mean processing time: 3.685 seconds"]),
("Architecture and limitations",["FastAPI endpoints: /health, /trips, /predict, /evaluation/trips, /evaluation/predict, /evaluation/map/{trip_id}.","Streamlit supports development and evaluator workflows; evaluator routes remain server-side.","Browser-level interaction was not automated. Five-model student-test comparison artifact is missing.","See report/review/FINAL_SUBMISSION_AUDIT.md and FINAL_LAUNCH_GUIDE.md."])]
with PdfPages(out) as pdf:
 for title, lines in pages:
  fig=plt.figure(figsize=(8.27,11.69)); fig.text(.08,.92,title,fontsize=21,weight="bold",color="#102a43")
  fig.text(.08,.84,"\n\n".join(lines),fontsize=13,va="top",linespacing=1.6); pdf.savefig(fig); plt.close(fig)
print(out)

