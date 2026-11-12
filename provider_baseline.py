"""Train and evaluate one small provider-level CNN baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from model import DemandCNN
from provider_data import make_provider_examples, load_provider_panel, split_provider_examples
from train_baseline import train_cnn, train_logistic


def probabilities(model: DemandCNN, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float())
        return torch.softmax(logits, dim=1)[:, 1].numpy()


def select_threshold(targets: np.ndarray, probabilities_: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities_)))
    scores = [balanced_accuracy_score(targets, probabilities_ >= value) for value in candidates]
    best = max(scores)
    return float(min((value for value, score in zip(candidates, scores) if np.isclose(score, best)), key=lambda value: abs(value - 0.5)))


def metrics(targets: np.ndarray, probabilities_: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = probabilities_ >= threshold
    negatives = targets == 0
    positives = targets == 1
    false_alarms = int(np.sum(predictions & negatives))
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "high_demand_recall": float(np.sum(predictions & positives) / max(np.sum(positives), 1)),
        "high_demand_precision": float(np.sum(predictions & positives) / max(np.sum(predictions), 1)),
        "false_alarms": false_alarms,
        "false_alarm_rate": float(false_alarms / max(np.sum(negatives), 1)),
        "examples": int(len(targets)),
        "high_demand_examples": int(np.sum(positives)),
    }


def macro_provider_balanced_accuracy(provider_codes: np.ndarray, targets: np.ndarray, probabilities_: np.ndarray, threshold: float) -> dict[str, float | int]:
    values = []
    for provider in np.unique(provider_codes):
        mask = provider_codes == provider
        if len(np.unique(targets[mask])) == 2:
            values.append(balanced_accuracy_score(targets[mask], probabilities_[mask] >= threshold))
    return {
        "provider_macro_balanced_accuracy": float(np.mean(values)) if values else float("nan"),
        "providers_with_both_classes": len(values),
        "providers_evaluated": int(len(np.unique(provider_codes))),
    }


def plot_results(result: dict[str, object], output_path: Path) -> None:
    periods = result["test_periods"]
    cnn = result["test_period_metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["CNN", "Logistic", "Seasonal"]
    scores = [result["methods"]["cnn"]["roc_auc"], result["methods"]["logistic"]["roc_auc"], result["methods"]["seasonal"]["roc_auc"]]
    axes[0].bar(labels, scores, color=["tab:blue", "tab:orange", "tab:gray"])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Held-out probability discrimination")
    axes[0].grid(axis="y", alpha=0.25)

    dates = [str(period)[:7] for period in periods]
    recalls = [cnn["by_period"][period]["high_demand_recall"] for period in periods]
    axes[1].bar(dates, recalls, color="tab:blue")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("High-demand recall")
    axes[1].set_title("CNN recall by test month")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw/provider_monthly"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/provider_baseline.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/provider_baseline_evaluation.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/provider_baseline.png"))
    args = parser.parse_args()

    panel = load_provider_panel(args.raw_directory)
    examples = make_provider_examples(panel)
    masks = split_provider_examples(examples)
    train_mask, validation_mask, test_mask = (masks[name] for name in ("train", "validation", "test"))

    cnn = train_cnn(examples.features[train_mask], examples.targets[train_mask])
    logistic = train_logistic(examples.features[train_mask], examples.targets[train_mask])
    cnn_validation = probabilities(cnn, examples.features[validation_mask])
    logistic_validation = logistic.predict_proba(examples.features[validation_mask])[:, 1]
    cnn_threshold = select_threshold(examples.targets[validation_mask], cnn_validation)
    logistic_threshold = select_threshold(examples.targets[validation_mask], logistic_validation)

    cnn_test = probabilities(cnn, examples.features[test_mask])
    logistic_test = logistic.predict_proba(examples.features[test_mask])[:, 1]
    seasonal_test = (examples.seasonal_values[test_mask] >= examples.thresholds[test_mask]).astype(float)
    test_targets = examples.targets[test_mask]
    test_codes = examples.provider_codes[test_mask]
    test_dates = examples.dates[test_mask]

    method_metrics = {
        "cnn": {**metrics(test_targets, cnn_test, cnn_threshold), "roc_auc": float(roc_auc_score(test_targets, cnn_test)), "pr_auc": float(average_precision_score(test_targets, cnn_test)), **macro_provider_balanced_accuracy(test_codes, test_targets, cnn_test, cnn_threshold)},
        "logistic": {**metrics(test_targets, logistic_test, logistic_threshold), "roc_auc": float(roc_auc_score(test_targets, logistic_test)), "pr_auc": float(average_precision_score(test_targets, logistic_test)), **macro_provider_balanced_accuracy(test_codes, test_targets, logistic_test, logistic_threshold)},
        "seasonal": {**metrics(test_targets, seasonal_test, 0.5), "roc_auc": float(roc_auc_score(test_targets, seasonal_test)), "pr_auc": float(average_precision_score(test_targets, seasonal_test)), **macro_provider_balanced_accuracy(test_codes, test_targets, seasonal_test, 0.5)},
    }

    by_period: dict[str, dict[str, float | int]] = {}
    for period in sorted(np.unique(test_dates)):
        period_mask = test_dates == period
        period_label = str(period)[:10]
        by_period[period_label] = metrics(test_targets[period_mask], cnn_test[period_mask], cnn_threshold)

    result = {
        "source_period": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "providers": int(panel["provider_code"].nunique()),
        "panel_rows": int(len(panel)),
        "examples": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "target_prevalence": {name: float(np.mean(examples.targets[mask])) for name, mask in masks.items()},
        "target_definition": "next month is high demand when its provider-specific attendance is at least the preceding 12-month 75th percentile",
        "feature_definition": "previous 12 monthly provider attendance values divided by their window median, minus one",
        "thresholds": {"cnn": cnn_threshold, "logistic": logistic_threshold},
        "methods": method_metrics,
        "test_periods": [str(period)[:10] for period in sorted(np.unique(test_dates))],
        "test_period_metrics": {"by_period": by_period},
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": cnn.state_dict(), "cnn_threshold": cnn_threshold}, args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_results(result, args.figure)
    print(json.dumps({"methods": method_metrics, "examples": result["examples"], "target_prevalence": result["target_prevalence"]}, indent=2))


if __name__ == "__main__":
    main()
