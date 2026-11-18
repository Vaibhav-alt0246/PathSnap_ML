"""Rebuild the PathSnap baseline model from partitioned trip files.

This reconstructs a new baseline artifact from source data; it is not claimed
to be the missing historical joblib artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.candidates.generator import generate_candidates
from src.config import load_config
from src.data.loader import load_trip
from src.features.engineering import build_features
from src.labels.alignment import label_candidates
from src.models.training import train_models


def dataset_fingerprint(root: Path, trip_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for trip_id in trip_ids:
        for suffix in (".track", ".nodes", ".arcs", ".route"):
            path = root / "train" / trip_id / f"{trip_id}{suffix}"
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_training_table(root: Path, trip_ids: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    frames, skipped = [], []
    for trip_id in trip_ids:
        try:
            trip = load_trip(root, "train", trip_id, include_route=True)
            candidates = generate_candidates(trip, search_radius_m=50.0, max_candidates=15, fallback_nearest=True)
            labeled = label_candidates(candidates, trip)
            features = build_features(trip, labeled)
            features.insert(0, "trip_id", trip_id)
            frames.append(features)
        except (FileNotFoundError, ValueError, IndexError, KeyError) as exc:
            skipped.append({"trip_id": trip_id, "reason": str(exc)})
    if not frames:
        raise RuntimeError("No valid training trips were found")
    return pd.concat(frames, ignore_index=True), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--feature-config", type=Path, default=Path("models/config/feature_columns.json"))
    args = parser.parse_args()
    config = json.loads(args.feature_config.read_text())
    trip_ids = sorted(path.name for path in (args.dataset_root / "train").iterdir() if path.is_dir())
    table, skipped = build_training_table(args.dataset_root, trip_ids)
    groups = table.trip_id.to_numpy()
    result = train_models(table, groups, config["feature_columns"], folds=5, seed=42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(args.dataset_root, [tid for tid in trip_ids if tid not in {x["trip_id"] for x in skipped}])
    artifact = {
        "model_pipeline": result["model"],
        "model_name": "hist_gradient_boosting_reconstructed_baseline",
        "feature_columns": config["feature_columns"],
        "dataset_sha256": fingerprint,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "training_trips": len(trip_ids) - len(skipped),
        "training_rows": len(table),
        "parameters": {"learning_rate": 0.05, "max_iter": 120, "max_leaf_nodes": 31,
                       "min_samples_leaf": 20, "l2_regularization": 0.1, "random_state": 42},
        "provenance": "New reconstruction from local train partition; historical artifact was unavailable.",
    }
    model_path = args.output_dir / "hist_gradient_boosting_reconstructed_baseline.joblib"
    joblib.dump(artifact, model_path)
    result["fold_metrics"].to_csv(args.output_dir / "reconstructed_baseline_grouped_metrics.csv", index=False)
    (args.output_dir / "reconstructed_baseline_metadata.json").write_text(json.dumps({k: v for k, v in artifact.items() if k != "model_pipeline"}, indent=2))
    (args.output_dir / "reconstructed_baseline_skipped_trips.json").write_text(json.dumps(skipped, indent=2))
    print(json.dumps({"model": str(model_path), "rows": len(table), "trips": len(trip_ids) - len(skipped),
                      "skipped": skipped, "fold_metrics": result["fold_metrics"].to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
