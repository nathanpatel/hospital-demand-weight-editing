"""Evaluate the CNN, logistic regression and seasonal persistence baselines."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_monthly_attendances, make_dataset, seasonal_persistence_predictions, split_dataset
from model import DemandCNN
from train_baseline import train_cnn, train_logistic


def metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision_high": float(precision_score(targets, predictions, zero_division=0)),
        "recall_high": float(recall_score(targets, predictions, zero_division=0)),
        "f1_high": float(f1_score(targets, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "high_predictions": int(predictions.sum()),
        "high_actual": int(targets.sum()),
    }


def evaluate(data_path: Path) -> dict[str, object]:
    frame = load_monthly_attendances(data_path)
    dataset = make_dataset(frame)
    train, test = split_dataset(dataset)

    cnn = train_cnn(train["features"], train["targets"])
    with torch.no_grad():
        cnn_predictions = cnn(torch.from_numpy(test["features"])).argmax(dim=1).numpy()
    logistic = train_logistic(train["features"], train["targets"])
    logistic_predictions = logistic.predict(test["features"])

    seasonal_predictions = (
        seasonal_persistence_predictions(dataset)[dataset.split_index:]
        >= dataset.thresholds[dataset.split_index:]
    ).astype(int)
    return {
        "settings": {
            "source": str(data_path),
            "window_size": 12,
            "threshold_quantile": 0.75,
            "threshold_definition": "preceding 12-month 75th percentile",
            "split": "first 75 percent of chronological examples for training; final 25 percent for testing",
        },
        "train": {
            "count": len(train["targets"]),
            "high_demand_count": int(train["targets"].sum()),
            "first_date": str(train["dates"][0])[:10],
            "last_date": str(train["dates"][-1])[:10],
        },
        "test": {
            "count": len(test["targets"]),
            "high_demand_count": int(test["targets"].sum()),
            "first_date": str(test["dates"][0])[:10],
            "last_date": str(test["dates"][-1])[:10],
        },
        "results": {
            "seasonal_persistence": metrics(seasonal_predictions, test["targets"].astype(int)),
            "logistic_regression": metrics(logistic_predictions, test["targets"].astype(int)),
            "cnn": metrics(cnn_predictions, test["targets"].astype(int)),
        },
        "test_predictions": {
            "dates": [str(value)[:10] for value in test["dates"]],
            "actual": test["targets"].astype(int).tolist(),
            "demand": test["target_values"].tolist(),
            "seasonal": seasonal_predictions.tolist(),
            "logistic": logistic_predictions.astype(int).tolist(),
            "cnn": cnn_predictions.astype(int).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/monthly_ae_time_series.xls"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_evaluation.json"))
    args = parser.parse_args()
    report = evaluate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
