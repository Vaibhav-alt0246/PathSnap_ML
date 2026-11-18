"""Simple real-data model demonstration for the PathSnap assignment.

This scores candidate rows from the training table with the already-trained
HistGradientBoosting artifact. It is a candidate-classification demo, not a
complete GPS-to-road map-matching evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


DEFAULT_DATASET = "/tmp/pathsnap-forensic-audit/member-a/member_a/results/training_candidates_features.parquet"
DEFAULT_MODEL = "/tmp/pathsnap-phase2-artifacts/best_baseline_model.joblib"
FEATURE_CONFIG = Path(__file__).parents[1] / "models/config/feature_columns.json"


def run_demo(dataset_path: str, model_path: str, output_dir: Path, sample_observations: int = 12) -> dict[str, float]:
    """Run deterministic candidate-level scoring on a small real-data sample."""
    config = json.loads(FEATURE_CONFIG.read_text(encoding="utf-8"))
    features = config["feature_columns"]
    data = pd.read_parquet(dataset_path)
    bundle = joblib.load(model_path)
    model = bundle["model_pipeline"]

    if list(bundle["feature_columns"]) != features:
        raise ValueError("Feature whitelist does not match serialized model metadata")
    if list(data[features].columns) != features:
        raise ValueError("Dataset feature order does not match the approved whitelist")

    # Select complete observations, spread deterministically across the table.
    observation_keys = data[["trip_id", "point_idx"]].drop_duplicates().head(sample_observations)
    sample = data.merge(observation_keys, on=["trip_id", "point_idx"], how="inner", sort=False)
    probabilities = model.predict_proba(sample[features].to_numpy())[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    actual = sample["label"].to_numpy(dtype=int)

    results = sample[["trip_id", "point_idx", "arc_id", "label"]].copy()
    results["predicted_probability"] = probabilities
    results["predicted_label"] = predicted
    results.to_csv(output_dir / "candidate_predictions.csv", index=False)

    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("PathSnap sample candidate classification")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Print one readable summary per sampled observation.
    print("Sample candidate classification results (not a held-out test set)")
    for (trip_id, point_idx), group in results.groupby(["trip_id", "point_idx"], sort=False):
        actual_positive = group.loc[group["label"] == 1, "arc_id"].tolist()
        best = group.loc[group["predicted_probability"].idxmax()]
        print(f"trip={trip_id} point={point_idx} candidates={len(group)} "
              f"actual_positive={actual_positive} predicted_best_arc={int(best.arc_id)} "
              f"best_probability={best.predicted_probability:.4f}")

    metrics = {
        "sample_rows": int(len(results)),
        "sample_observations": int(len(observation_keys)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }
    print(json.dumps(metrics, indent=2))
    print("Outputs:", output_dir / "candidate_predictions.csv", output_dir / "confusion_matrix.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--sample-observations", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_demo(args.dataset, args.model, args.output_dir, args.sample_observations)


if __name__ == "__main__":
    main()
