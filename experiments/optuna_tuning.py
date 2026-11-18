"""Small reproducible Optuna/TPE study for HistGradientBoosting.

Uses all candidate rows and GroupKFold by trip. Outputs are written to the
requested directory and never overwrite the baseline artifact.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss


def top1_accuracy(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    work = frame[["trip_id", "point_idx", "label"]].copy()
    work["prob"] = probabilities
    correct = evaluable = 0
    for _, group in work.groupby(["trip_id", "point_idx"], sort=False):
        if not (group["label"] == 1).any():
            continue
        evaluable += 1
        correct += int(group.iloc[int(np.argmax(group["prob"].to_numpy()))]["label"] == 1)
    return correct / evaluable if evaluable else 0.0


def run(dataset: Path, feature_config: Path, output_dir: Path, trials: int, seed: int) -> None:
    config = json.loads(feature_config.read_text())
    features = config["feature_columns"]
    df = pd.read_parquet(dataset)
    X, y, groups = df[features], df["label"].to_numpy(), df["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 180),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
        top1_scores, losses = [], []
        for train_idx, valid_idx in folds:
            model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **params)
            model.fit(X.iloc[train_idx], y[train_idx])
            probs = model.predict_proba(X.iloc[valid_idx])[:, 1]
            valid = df.iloc[valid_idx]
            top1_scores.append(top1_accuracy(valid, probs))
            losses.append(log_loss(y[valid_idx], probs, labels=[0, 1]))
        trial.set_user_attr("mean_log_loss", float(np.mean(losses)))
        trial.set_user_attr("top1_std", float(np.std(top1_scores)))
        return float(np.mean(top1_scores))

    started = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)
    (output_dir / "optuna_best.json").write_text(json.dumps({"best_value": study.best_value, "best_params": study.best_params, "elapsed_seconds": elapsed, "trials": trials, "seed": seed}, indent=2))
    final = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **study.best_params)
    final.fit(X, y)
    joblib.dump({"model_pipeline": final, "model_name": "hist_gradient_boosting_tuned", "feature_columns": features, "dataset_sha256": config["dataset_sha256"], "best_params": study.best_params}, output_dir / "hist_gradient_boosting_tuned.joblib")
    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params, "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, default=Path("models/config/feature_columns.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pathsnap-optuna-results"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.dataset, args.feature_config, args.output_dir, args.trials, args.seed)
