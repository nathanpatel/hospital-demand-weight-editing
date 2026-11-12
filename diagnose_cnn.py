"""Diagnose the saved CNN without retraining it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from data import make_dataset, load_monthly_attendances, seasonal_persistence_predictions, split_dataset
from model import DemandCNN
from train_baseline import train_logistic


def load_saved_model(checkpoint_path: Path) -> DemandCNN:
    """Load the existing CNN checkpoint without creating a new trained model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DemandCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def high_demand_probabilities(model: DemandCNN, features: np.ndarray) -> np.ndarray:
    """Return the model's probability for the high-demand class."""
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float())
        return torch.softmax(logits, dim=1)[:, 1].numpy()


def classification_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    negatives = targets == 0
    false_alarms = int(np.sum(predictions & negatives))
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "high_demand_recall": float(np.sum(predictions & (targets == 1)) / max(np.sum(targets == 1), 1)),
        "false_alarms": false_alarms,
        "false_alarm_rate": float(false_alarms / max(np.sum(negatives), 1)),
        "high_predictions": int(np.sum(predictions)),
    }


def select_threshold(validation_targets: np.ndarray, validation_probabilities: np.ndarray) -> tuple[float, dict[str, float | int]]:
    """Choose a threshold using only a validation portion of the training data."""
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], validation_probabilities)))
    scores = [balanced_accuracy_score(validation_targets, validation_probabilities >= candidate) for candidate in candidates]
    best_score = max(scores)
    best_candidates = [candidate for candidate, score in zip(candidates, scores) if np.isclose(score, best_score)]
    threshold = min(best_candidates, key=lambda candidate: abs(candidate - 0.5))
    return float(threshold), classification_metrics(validation_targets, validation_probabilities, threshold)


def threshold_curve(targets: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float | int]]:
    return [
        classification_metrics(targets, probabilities, float(threshold))
        for threshold in np.linspace(0.0, 1.0, 101)
    ]


def error_examples(
    dates: np.ndarray,
    target_values: np.ndarray,
    thresholds: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, list[dict[str, object]]]:
    predictions = probabilities >= threshold
    examples: dict[str, list[dict[str, object]]] = {"false_positives": [], "false_negatives": []}
    for label, mask in (
        ("false_positives", predictions & (targets == 0)),
        ("false_negatives", ~predictions & (targets == 1)),
    ):
        for index in np.flatnonzero(mask)[:3]:
            examples[label].append(
                {
                    "date": str(np.datetime_as_string(dates[index], unit="D")),
                    "attendance": float(target_values[index]),
                    "preceding_year_threshold": float(thresholds[index]),
                    "actual_high_demand": int(targets[index]),
                    "high_demand_probability": float(probabilities[index]),
                }
            )
    return examples


def plot_diagnosis(
    curve: list[dict[str, float | int]],
    targets: np.ndarray,
    probabilities: np.ndarray,
    selected_threshold: float,
    output_path: Path,
) -> None:
    thresholds = np.asarray([row["threshold"] for row in curve], dtype=float)
    recall = np.asarray([row["high_demand_recall"] for row in curve], dtype=float)
    false_alarm_rate = np.asarray([row["false_alarm_rate"] for row in curve], dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(thresholds, recall, label="High-demand recall", color="tab:blue")
    axes[0].plot(thresholds, false_alarm_rate, label="False-alarm rate", color="tab:orange")
    axes[0].axvline(selected_threshold, color="black", linestyle="--", label=f"Training-selected threshold ({selected_threshold:.2f})")
    axes[0].set(xlabel="Decision threshold", ylabel="Rate", ylim=(-0.02, 1.02), title="Threshold trade-off on held-out data")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    bins = np.linspace(0, 1, 11)
    axes[1].hist(probabilities[targets == 0], bins=bins, alpha=0.65, label="Ordinary months", color="tab:gray")
    axes[1].hist(probabilities[targets == 1], bins=bins, alpha=0.65, label="High-demand months", color="tab:red")
    axes[1].axvline(selected_threshold, color="black", linestyle="--")
    axes[1].set(xlabel="Predicted high-demand probability", ylabel="Number of months", title="CNN probabilities on held-out data")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/monthly_ae_time_series.xls"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cnn_diagnosis.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/cnn_diagnosis.png"))
    args = parser.parse_args()

    frame = load_monthly_attendances(args.data)
    dataset = make_dataset(frame)
    train, test = split_dataset(dataset)
    model = load_saved_model(args.checkpoint)
    train_probabilities = high_demand_probabilities(model, train["features"])
    test_probabilities = high_demand_probabilities(model, test["features"])

    validation_start = int(len(train["targets"]) * 0.8)
    selected_threshold, validation_metrics = select_threshold(
        train["targets"][validation_start:], train_probabilities[validation_start:]
    )
    cnn_metrics = classification_metrics(test["targets"], test_probabilities, selected_threshold)
    argmax_metrics = classification_metrics(test["targets"], test_probabilities, 0.5)

    logistic = train_logistic(train["features"], train["targets"])
    logistic_probabilities = logistic.predict_proba(test["features"])[:, 1]
    logistic_metrics = classification_metrics(test["targets"], logistic_probabilities, 0.5)

    seasonal_values = seasonal_persistence_predictions(dataset)[dataset.split_index:]
    seasonal_predictions = seasonal_values >= dataset.thresholds[dataset.split_index:]
    seasonal_metrics = classification_metrics(test["targets"], seasonal_predictions.astype(float), 0.5)

    curve = threshold_curve(test["targets"], test_probabilities)
    result = {
        "checkpoint": str(args.checkpoint),
        "test_examples": int(len(test["targets"])),
        "test_high_demand_examples": int(np.sum(test["targets"] == 1)),
        "threshold_selection": {
            "method": "maximise balanced accuracy on the final 20% of the chronological training examples; ties choose the value nearest 0.5",
            "validation_examples": int(len(train["targets"]) - validation_start),
            "threshold": selected_threshold,
            "validation_metrics": validation_metrics,
        },
        "cnn_probability_metrics": {
            "roc_auc": float(roc_auc_score(test["targets"], test_probabilities)),
            "pr_auc_average_precision": float(average_precision_score(test["targets"], test_probabilities)),
            "ordinary_probability_mean": float(np.mean(test_probabilities[test["targets"] == 0])),
            "high_demand_probability_mean": float(np.mean(test_probabilities[test["targets"] == 1])),
        },
        "methods": {
            "cnn_selected_threshold": cnn_metrics,
            "cnn_default_threshold_0.5": argmax_metrics,
            "logistic_regression_threshold_0.5": logistic_metrics,
            "seasonal_persistence": seasonal_metrics,
        },
        "error_examples": {
            "selected_threshold": error_examples(
                test["dates"], test["target_values"], dataset.thresholds[dataset.split_index:], test["targets"], test_probabilities, selected_threshold
            ),
            "default_threshold_0.5": error_examples(
                test["dates"], test["target_values"], dataset.thresholds[dataset.split_index:], test["targets"], test_probabilities, 0.5
            ),
        },
        "threshold_curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_diagnosis(curve, test["targets"], test_probabilities, selected_threshold, args.figure)
    print(json.dumps({"cnn": cnn_metrics, "logistic": logistic_metrics, "seasonal": seasonal_metrics}, indent=2))
    print(f"ROC-AUC: {result['cnn_probability_metrics']['roc_auc']:.3f}")
    print(f"PR-AUC: {result['cnn_probability_metrics']['pr_auc_average_precision']:.3f}")
    print(f"Saved diagnosis to {args.output}")
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
