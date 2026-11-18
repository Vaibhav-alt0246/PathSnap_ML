"""Reproducible grouped training helpers for candidate classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import GroupKFold


def baseline_classifier(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=120, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.1, random_state=seed,
    )


def _top1(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    work = frame[["trip_id", "point_idx", "label"]].copy()
    work["probability"] = probabilities
    correct = total = 0
    for _, group in work.groupby(["trip_id", "point_idx"], sort=False):
        if not (group.label == 1).any():
            continue
        total += 1
        correct += int(group.iloc[int(np.argmax(group.probability.to_numpy()))].label == 1)
    return correct / total if total else 0.0


def train_models(feature_table: pd.DataFrame, groups, feature_columns: list[str], folds: int = 5, seed: int = 42):
    """Fit the verified baseline with GroupKFold-by-trip diagnostics."""
    missing = set(feature_columns) - set(feature_table.columns)
    if missing:
        raise ValueError(f"missing training features: {sorted(missing)}")
    if "label" not in feature_table or "trip_id" not in feature_table or "point_idx" not in feature_table:
        raise ValueError("training table requires trip_id, point_idx, and label")
    splitter = GroupKFold(n_splits=folds)
    oof = np.full(len(feature_table), np.nan)
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(feature_table, feature_table.label, groups)):
        model = baseline_classifier(seed)
        model.fit(feature_table.iloc[train_idx][feature_columns].to_numpy(), feature_table.iloc[train_idx].label)
        probabilities = model.predict_proba(feature_table.iloc[valid_idx][feature_columns].to_numpy())[:, 1]
        oof[valid_idx] = probabilities
        fold_rows.append({
            "fold": fold,
            "train_trips": int(len(set(np.asarray(groups)[train_idx]))),
            "validation_trips": int(len(set(np.asarray(groups)[valid_idx]))),
            "top1": _top1(feature_table.iloc[valid_idx], probabilities),
            "log_loss": float(log_loss(feature_table.iloc[valid_idx].label, probabilities, labels=[0, 1])),
            "macro_f1": float(f1_score(feature_table.iloc[valid_idx].label, probabilities >= 0.5, average="macro")),
        })
    final = baseline_classifier(seed)
    final.fit(feature_table[feature_columns].to_numpy(), feature_table.label)
    return {"model": final, "oof_probabilities": oof, "fold_metrics": pd.DataFrame(fold_rows)}


def compare_models(results: dict) -> pd.DataFrame:
    """Return grouped fold metrics from a training result."""
    if "fold_metrics" not in results:
        raise ValueError("results does not contain fold_metrics")
    return results["fold_metrics"].copy()
