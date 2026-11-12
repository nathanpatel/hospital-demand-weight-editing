"""Inspect repeated provider-CNN errors without retraining or using the test period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import DemandCNN
from provider_baseline import probabilities, select_threshold
from provider_data import make_provider_examples, load_provider_panel, split_provider_examples
from train_baseline import train_logistic


def load_saved_cnn(checkpoint_path: Path) -> tuple[DemandCNN, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DemandCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint["cnn_threshold"])


def forward_details(model: DemandCNN, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        inputs = torch.from_numpy(features).float()
        hidden = model.features(inputs.unsqueeze(1)).squeeze(-1)
        logits = model.classifier(hidden)
        probabilities_ = torch.softmax(logits, dim=1)[:, 1]
    return probabilities_.numpy(), logits.numpy(), hidden.numpy()


def recent_change(features: np.ndarray) -> np.ndarray:
    """Compare the last three months with the preceding three months."""
    return features[:, -3:].mean(axis=1) - features[:, -6:-3].mean(axis=1)


def group_summary(
    examples,
    mask: np.ndarray,
    probabilities_: np.ndarray,
    threshold: float,
    group_mask: np.ndarray,
) -> dict[str, float | int]:
    selected = mask & group_mask
    targets = examples.targets[selected]
    predictions = probabilities_[selected] >= threshold
    positives = targets == 1
    negatives = targets == 0
    positive_indices = np.flatnonzero(selected)
    positive_indices = positive_indices[examples.targets[positive_indices] == 1]
    return {
        "examples": int(np.sum(selected)),
        "providers": int(len(np.unique(examples.provider_codes[selected]))),
        "high_demand_examples": int(np.sum(positives)),
        "high_demand_providers": int(len(np.unique(examples.provider_codes[positive_indices]))),
        "high_demand_recall": float(np.mean(predictions[positives])) if np.any(positives) else 0.0,
        "false_alarm_rate": float(np.mean(predictions[negatives])) if np.any(negatives) else 0.0,
    }


def representative_examples(
    examples,
    mask: np.ndarray,
    candidate_mask: np.ndarray,
    probabilities_: np.ndarray,
    logits: np.ndarray,
    hidden: np.ndarray,
    threshold: float,
    limit: int = 3,
) -> list[dict[str, object]]:
    selected = mask & candidate_mask & (examples.targets == 1) & (probabilities_ < threshold)
    output = []
    for index in np.flatnonzero(selected)[:limit]:
        output.append(
            {
                "provider_code": str(examples.provider_codes[index]),
                "date": str(examples.dates[index].date()),
                "normalised_input": [round(float(value), 3) for value in examples.features[index]],
                "target_value": float(examples.target_values[index]),
                "target_threshold": float(examples.thresholds[index]),
                "target_ratio": float(examples.target_values[index] / examples.thresholds[index]),
                "high_demand_probability": float(probabilities_[index]),
                "logits": [float(value) for value in logits[index]],
                "high_minus_ordinary_score": float(logits[index, 1] - logits[index, 0]),
                "hidden_activations": [float(value) for value in hidden[index]],
            }
        )
    return output


def plot_pattern(
    examples,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    candidate_mask: np.ndarray,
    probabilities_: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    group_masks = [candidate_mask, ~candidate_mask]
    group_labels = ["Flat or falling lead-in", "Recent rise"]
    x = np.arange(2)
    width = 0.35
    for offset, (split_label, split_mask) in enumerate((("Training", train_mask), ("Validation", validation_mask))):
        recalls = []
        for group_mask in group_masks:
            selected = split_mask & group_mask & (examples.targets == 1)
            recalls.append(np.mean((probabilities_[selected] >= threshold)) if np.any(selected) else 0.0)
        axes[0].bar(x + (offset - 0.5) * width, recalls, width, label=split_label)
    axes[0].set_xticks(x, group_labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("High-demand recall")
    axes[0].set_title("CNN misses more flat/falling cases")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    candidate_false_negatives = train_mask & candidate_mask & (examples.targets == 1) & (probabilities_ < threshold)
    indices = np.flatnonzero(candidate_false_negatives)[:3]
    positions = np.arange(-11, 1)
    for index in indices:
        axes[1].plot(positions, examples.features[index], marker="o", label=f"{examples.provider_codes[index]} {str(examples.dates[index])[:7]}")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Months before prediction", ylabel="Normalised attendance", title="Representative missed high-demand cases")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw/provider_monthly"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/provider_baseline.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/provider_error_diagnosis.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/provider_error_pattern.png"))
    args = parser.parse_args()

    panel = load_provider_panel(args.raw_directory)
    examples = make_provider_examples(panel)
    masks = split_provider_examples(examples)
    train_mask, validation_mask, test_mask = (masks[name] for name in ("train", "validation", "test"))
    model, threshold = load_saved_cnn(args.checkpoint)
    probabilities_, logits, hidden = forward_details(model, examples.features)
    change = recent_change(examples.features)
    candidate_mask = change <= 0

    logistic = train_logistic(examples.features[train_mask], examples.targets[train_mask])
    logistic_probabilities = np.full(len(examples.targets), np.nan)
    known_mask = train_mask | validation_mask
    logistic_probabilities[known_mask] = logistic.predict_proba(examples.features[known_mask])[:, 1]
    logistic_threshold = select_threshold(examples.targets[validation_mask], logistic_probabilities[validation_mask])
    seasonal_probabilities = (examples.seasonal_values >= examples.thresholds).astype(float)

    result = {
        "selection_rule": "candidate errors have a non-positive change from the mean of months -6 to -4 to the mean of months -3 to -1; only training and validation are used",
        "threshold": threshold,
        "test_used_for_selection": False,
        "test_examples_excluded": int(np.sum(test_mask)),
        "candidate_pattern": {
            "definition": "a high-demand month after a flat or falling recent three-month lead-in",
            "train": group_summary(examples, train_mask, probabilities_, threshold, candidate_mask),
            "validation": group_summary(examples, validation_mask, probabilities_, threshold, candidate_mask),
        },
        "comparable_pattern": {
            "definition": "a high-demand month after a positive recent three-month lead-in",
            "train": group_summary(examples, train_mask, probabilities_, threshold, ~candidate_mask),
            "validation": group_summary(examples, validation_mask, probabilities_, threshold, ~candidate_mask),
        },
        "baseline_comparison": {
            "cnn": {"threshold": threshold, "train": group_summary(examples, train_mask, probabilities_, threshold, candidate_mask), "validation": group_summary(examples, validation_mask, probabilities_, threshold, candidate_mask)},
            "logistic": {"threshold": logistic_threshold, "train": group_summary(examples, train_mask, logistic_probabilities, logistic_threshold, candidate_mask), "validation": group_summary(examples, validation_mask, logistic_probabilities, logistic_threshold, candidate_mask)},
            "seasonal": {"threshold": 0.5, "train": group_summary(examples, train_mask, seasonal_probabilities, 0.5, candidate_mask), "validation": group_summary(examples, validation_mask, seasonal_probabilities, 0.5, candidate_mask)},
        },
        "cnn_threshold_comparison": {
            "selected_threshold": {
                "candidate": group_summary(examples, known_mask, probabilities_, threshold, candidate_mask),
                "all_known_examples": group_summary(examples, known_mask, probabilities_, threshold, np.ones(len(examples.targets), dtype=bool)),
            },
            "threshold_0.5": {
                "candidate": group_summary(examples, known_mask, probabilities_, 0.5, candidate_mask),
                "all_known_examples": group_summary(examples, known_mask, probabilities_, 0.5, np.ones(len(examples.targets), dtype=bool)),
            },
        },
        "representative_training_errors": representative_examples(examples, train_mask, candidate_mask, probabilities_, logits, hidden, threshold),
        "representative_validation_errors": representative_examples(examples, validation_mask, candidate_mask, probabilities_, logits, hidden, threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_pattern(examples, train_mask, validation_mask, candidate_mask, probabilities_, threshold, args.figure)
    print(json.dumps(result, indent=2))
    print(f"Saved diagnosis to {args.output}")
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
