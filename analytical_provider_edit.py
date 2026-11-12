"""Apply one fixed, sparse analytical final-layer edit to the provider CNN."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import DemandCNN
from provider_baseline import metrics, probabilities, select_threshold
from provider_data import make_provider_examples, load_provider_panel, split_provider_examples
from train_baseline import train_logistic


MAX_CHANGES = 8
MAX_ABS_DELTA = 0.10
MAX_L2_DELTA = 0.20
TARGET_EXAMPLES = 16
PRESERVATION_PENALTY = 1.0


def load_saved_cnn(checkpoint_path: Path) -> tuple[DemandCNN, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DemandCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint["cnn_threshold"])


def hidden_and_logits(model: DemandCNN, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        inputs = torch.from_numpy(features).float()
        hidden = model.features(inputs.unsqueeze(1)).squeeze(-1)
        logits = model.classifier(hidden)
    return hidden.numpy(), logits.numpy()


def select_target_indices(examples, train_mask: np.ndarray, target_mask: np.ndarray, limit: int = TARGET_EXAMPLES) -> np.ndarray:
    """Select one target example per provider, then take sorted providers deterministically."""
    candidates = np.flatnonzero(train_mask & target_mask & (examples.targets == 1))
    selected: list[int] = []
    seen: set[str] = set()
    for index in candidates:
        provider = str(examples.provider_codes[index])
        if provider not in seen:
            selected.append(int(index))
            seen.add(provider)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(f"Only found {len(selected)} distinct target providers")
    return np.asarray(selected, dtype=int)


def margin_coefficients(hidden: np.ndarray) -> np.ndarray:
    """Map final-layer parameter changes to high-minus-ordinary margin changes."""
    return np.concatenate([-hidden, hidden, -np.ones((len(hidden), 1)), np.ones((len(hidden), 1))], axis=1)


def bounded_direction(coefficients: np.ndarray, max_abs_delta: float, max_l2_delta: float) -> np.ndarray:
    """Maximise c dot d on one support with box and L2 constraints."""
    magnitudes = np.abs(coefficients)
    if not np.any(magnitudes):
        return np.zeros_like(coefficients)
    capped = np.full_like(magnitudes, max_abs_delta)
    if np.linalg.norm(capped) <= max_l2_delta:
        allocation = capped
    else:
        low, high = 0.0, max_abs_delta / max(float(np.min(magnitudes[magnitudes > 0])), 1e-12)
        for _ in range(80):
            scale = (low + high) / 2.0
            allocation = np.minimum(max_abs_delta, scale * magnitudes)
            if np.linalg.norm(allocation) > max_l2_delta:
                high = scale
            else:
                low = scale
        allocation = np.minimum(max_abs_delta, low * magnitudes)
    return np.sign(coefficients) * allocation


def optimal_sparse_delta(
    objective_coefficients: np.ndarray,
    max_changes: int = MAX_CHANGES,
    max_abs_delta: float = MAX_ABS_DELTA,
    max_l2_delta: float = MAX_L2_DELTA,
) -> tuple[np.ndarray, float]:
    """Enumerate all supports up to the budget and return the exact best edit."""
    if objective_coefficients.shape != (18,):
        raise ValueError("The provider CNN final layer must expose 18 editable scalars")
    best_delta = np.zeros(18, dtype=np.float64)
    best_value = 0.0
    for size in range(1, max_changes + 1):
        for support in itertools.combinations(range(18), size):
            support_coefficients = objective_coefficients[list(support)]
            support_delta = bounded_direction(support_coefficients, max_abs_delta, max_l2_delta)
            delta = np.zeros(18, dtype=np.float64)
            delta[list(support)] = support_delta
            value = float(objective_coefficients @ delta)
            if value > best_value + 1e-12:
                best_value = value
                best_delta = delta
    return best_delta, best_value


def apply_final_layer_delta(model: DemandCNN, delta: np.ndarray) -> DemandCNN:
    if delta.shape != (18,):
        raise ValueError("Final-layer delta must contain 18 scalars")
    edited = copy.deepcopy(model)
    with torch.no_grad():
        edited.classifier.weight.add_(torch.from_numpy(delta[:16].reshape(2, 8)).float())
        edited.classifier.bias.add_(torch.from_numpy(delta[16:]).float())
    edited.eval()
    return edited


def pattern_metrics(examples, mask: np.ndarray, probabilities_: np.ndarray, threshold: float, target_mask: np.ndarray) -> dict[str, object]:
    groups = {"flat_or_falling": mask & target_mask, "recent_rise": mask & ~target_mask, "all": mask}
    return {name: metrics(examples.targets[group], probabilities_[group], threshold) for name, group in groups.items()}


def plot_comparison(result: dict[str, object], output_path: Path) -> None:
    labels = ["Original", "Analytical edit", "Threshold 0.5"]
    target_recall = [result["validation"][key]["flat_or_falling"]["high_demand_recall"] for key in ("original", "edited", "threshold_0.5")]
    false_alarm_rate = [result["validation"][key]["all"]["false_alarm_rate"] for key in ("original", "edited", "threshold_0.5")]
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    axes[0].bar(x, target_recall, color=["tab:gray", "tab:blue", "tab:orange"])
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Recall on flat/falling high-demand cases")
    axes[0].set_title("Target-pattern recall")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, false_alarm_rate, color=["tab:gray", "tab:blue", "tab:orange"], width=width)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("False-alarm rate on validation")
    axes[1].set_title("Cost of changing the decision")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw/provider_monthly"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/provider_baseline.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/provider_analytical_edit.json"))
    parser.add_argument("--patch", type=Path, default=Path("outputs/provider_analytical_patch.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/provider_analytical_edit.png"))
    args = parser.parse_args()

    panel = load_provider_panel(args.raw_directory)
    examples = make_provider_examples(panel)
    masks = split_provider_examples(examples)
    train_mask, validation_mask, test_mask = (masks[name] for name in ("train", "validation", "test"))
    model, cnn_threshold = load_saved_cnn(args.checkpoint)
    hidden, original_logits = hidden_and_logits(model, examples.features)
    original_probabilities = torch.softmax(torch.from_numpy(original_logits), dim=1)[:, 1].numpy()

    recent_change = examples.features[:, -3:].mean(axis=1) - examples.features[:, -6:-3].mean(axis=1)
    target_pattern = recent_change <= 0
    target_indices = select_target_indices(examples, train_mask, target_pattern)
    preservation_mask = train_mask & (examples.targets == 0)
    target_coefficients = margin_coefficients(hidden[target_indices]).mean(axis=0)
    preservation_coefficients = margin_coefficients(hidden[preservation_mask]).mean(axis=0)
    objective_coefficients = target_coefficients - PRESERVATION_PENALTY * preservation_coefficients
    delta, objective_gain = optimal_sparse_delta(objective_coefficients)
    edited_model = apply_final_layer_delta(model, delta)
    _, edited_logits = hidden_and_logits(edited_model, examples.features)
    edited_probabilities = torch.softmax(torch.from_numpy(edited_logits), dim=1)[:, 1].numpy()

    logistic = train_logistic(examples.features[train_mask], examples.targets[train_mask])
    logistic_validation = logistic.predict_proba(examples.features[validation_mask])[:, 1]
    logistic_threshold = select_threshold(examples.targets[validation_mask], logistic_validation)
    logistic_all = np.full(len(examples.targets), np.nan)
    logistic_all[validation_mask] = logistic_validation

    target_predictions_original = original_probabilities[target_indices] >= cnn_threshold
    target_predictions_edited = edited_probabilities[target_indices] >= cnn_threshold
    all_known = train_mask | validation_mask
    original_margins = original_logits[:, 1] - original_logits[:, 0]
    edited_margins = edited_logits[:, 1] - edited_logits[:, 0]

    def margin_summary(indices: np.ndarray) -> dict[str, float]:
        return {
            "original_mean": float(np.mean(original_margins[indices])),
            "edited_mean": float(np.mean(edited_margins[indices])),
            "change": float(np.mean(edited_margins[indices] - original_margins[indices])),
        }

    result = {
        "constraints": {"max_changes": MAX_CHANGES, "max_abs_delta": MAX_ABS_DELTA, "max_l2_delta": MAX_L2_DELTA, "editable_parameters": 18},
        "selection_rule": "one training target example per provider, sorted by provider code and date, taking the first 16 distinct providers",
        "target_indices": [int(index) for index in target_indices],
        "target_examples": [{"provider_code": str(examples.provider_codes[index]), "date": str(examples.dates[index].date())} for index in target_indices],
        "preservation_examples": int(np.sum(preservation_mask)),
        "test_used_for_edit": False,
        "test_examples_excluded": int(np.sum(test_mask)),
        "objective": {"description": "mean target margin improvement minus mean ordinary-training margin improvement", "penalty": PRESERVATION_PENALTY, "optimal_linear_objective_gain": objective_gain},
        "patch": {"changed_parameters": int(np.sum(np.abs(delta) > 1e-9)), "l2_norm": float(np.linalg.norm(delta)), "max_abs_change": float(np.max(np.abs(delta))), "delta": [float(value) for value in delta]},
        "margin_changes": {"selected_training_targets": margin_summary(target_indices), "ordinary_training_examples": margin_summary(np.flatnonzero(preservation_mask))},
        "target_training_examples": {"selected": int(len(target_indices)), "original_correct": int(np.sum(target_predictions_original)), "edited_correct": int(np.sum(target_predictions_edited)), "corrected": int(np.sum(~target_predictions_original & target_predictions_edited))},
        "validation": {
            "original": pattern_metrics(examples, validation_mask, original_probabilities, cnn_threshold, target_pattern),
            "edited": pattern_metrics(examples, validation_mask, edited_probabilities, cnn_threshold, target_pattern),
            "threshold_0.5": pattern_metrics(examples, validation_mask, original_probabilities, 0.5, target_pattern),
            "logistic": pattern_metrics(examples, validation_mask, logistic_all, logistic_threshold, target_pattern),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    patch = {"parameter_order": "classifier.weight row-major then classifier.bias", "delta": result["patch"]["delta"], "constraints": result["constraints"]}
    args.patch.write_text(json.dumps(patch, indent=2), encoding="utf-8")
    torch.save({"model_state": edited_model.state_dict(), "source_checkpoint": str(args.checkpoint), "patch": patch}, args.output.with_name("provider_analytical_model.pt"))
    plot_comparison(result, args.figure)
    print(json.dumps({"patch": result["patch"], "target_training_examples": result["target_training_examples"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
