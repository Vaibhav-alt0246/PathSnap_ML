from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

out = Path(__file__).with_name("PathSnap_Final_Report.pdf")
with PdfPages(out) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "PathSnap — Final Engineering Report", fontsize=22, weight="bold")
    body = (
        "Assignment demonstration status\n\n"
        "Objective: classify GPS candidate road arcs and provide directed graph/HMM foundations.\n\n"
        "Verified real training table\n"
        "• 1,514,674 candidate rows, 80 trips, 208,417 observations\n"
        "• 185,461 positive and 1,329,213 negative labels\n"
        "• 24 approved model features; no missing or infinite numeric values\n\n"
        "Grouped 5-fold model comparison\n"
        "• HistGradientBoosting Top-1: 0.832225; log loss: 0.181852\n"
        "• Random Forest Top-1: 0.831020\n"
        "• Logistic Regression Top-1: 0.830633\n"
        "• Gaussian NB Top-1: 0.808701\n"
        "• KNN Top-1: 0.783792\n\n"
        "Optuna\n"
        "• 20 valid TPE trials, seed 42, grouped by trip\n"
        "• Best Top-1: 0.836964; mean log loss: 0.192618\n"
        "• Untuned baseline remains preferred for probability-quality HMM emissions.\n\n"
        "End-to-end status\n"
        "A genuine raw GPS-to-road run succeeded on training trip 00000033: 5,844\n"
        "observations, 43,219 candidates, and complete classifier/HMM outputs. The\n"
        "same path was verified through the API. Route-level out-of-fold HMM and\n"
        "student-test evaluation remain separate from tuning. All 10 student-test\n"
        "predictions were generated. The original positional score 0.000459 was\n"
        "audited as unsuitable for unequal per-observation versus unique-route\n"
        "sequences; corrected mean arc-set recall is 0.919452 and LCS recall is\n"
        "0.913063. The complete 80-trip OOF HMM aggregate was not claimed because\n"
        "the optimized sparse decoder completed all 80 OOF trips; HMM continuity\n"
        "improved but route-membership diagnostics decreased. No student-test .route\n"
        "file was read during prediction or model tuning."
    )
    fig.text(0.08, 0.86, body, fontsize=11, va="top", linespacing=1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
print(out)
