"""Run one controlled LLM proposal for the provider-level CNN."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel, Field

from analytical_provider_edit import (
    apply_final_layer_delta,
    hidden_and_logits,
    load_saved_cnn,
    pattern_metrics,
    select_target_indices,
)
from model import DemandCNN
from provider_baseline import select_threshold
from provider_data import make_provider_examples, load_provider_panel, split_provider_examples
from train_baseline import train_logistic


MODEL_NAME = "gpt-6.1-sol"
REASONING_EFFORT = "medium"
MAX_CHANGES = 8
MAX_ABS_DELTA = 0.10
MAX_L2_DELTA = 0.20
ORDINARY_CONTEXT_EXAMPLES = 8


def recent_change(features: np.ndarray) -> np.ndarray:
    """Compare the final three months with the preceding three months."""
    return features[:, -3:].mean(axis=1) - features[:, -6:-3].mean(axis=1)


def load_model_state(checkpoint_path: Path) -> DemandCNN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DemandCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


class PatchChange(BaseModel):
    parameter: Literal["classifier.weight", "classifier.bias"]
    index: list[int] = Field(min_length=1, max_length=2)
    delta: float


class WeightPatch(BaseModel):
    changes: list[PatchChange]
    explanation: str


def validate_patch(patch: WeightPatch | dict[str, object]) -> tuple[WeightPatch | None, list[str]]:
    """Validate an LLM patch without changing a model."""
    errors: list[str] = []
    try:
        parsed = patch if isinstance(patch, WeightPatch) else WeightPatch.model_validate(patch)
    except Exception as error:
        return None, [f"invalid patch schema: {error}"]

    if len(parsed.changes) > MAX_CHANGES:
        errors.append(f"too many changes: {len(parsed.changes)} > {MAX_CHANGES}")
    seen: set[tuple[str, tuple[int, ...]]] = set()
    deltas: list[float] = []
    for change in parsed.changes:
        key = (change.parameter, tuple(change.index))
        if key in seen:
            errors.append(f"duplicate change: {key}")
        seen.add(key)
        if change.parameter == "classifier.weight" and (len(change.index) != 2 or not (0 <= change.index[0] < 2 and 0 <= change.index[1] < 8)):
            errors.append(f"invalid classifier.weight index: {change.index}")
        if change.parameter == "classifier.bias" and (len(change.index) != 1 or not (0 <= change.index[0] < 2)):
            errors.append(f"invalid classifier.bias index: {change.index}")
        if not np.isfinite(change.delta):
            errors.append(f"non-finite delta at {key}")
        elif abs(change.delta) > MAX_ABS_DELTA + 1e-12:
            errors.append(f"delta too large at {key}: {change.delta}")
        deltas.append(float(change.delta))
    if np.linalg.norm(deltas) > MAX_L2_DELTA + 1e-12:
        errors.append(f"total L2 change too large: {np.linalg.norm(deltas)}")
    return (parsed if not errors else None), errors


def patch_to_delta(patch: WeightPatch) -> np.ndarray:
    delta = np.zeros(18, dtype=np.float64)
    for change in patch.changes:
        if change.parameter == "classifier.weight":
            delta[change.index[0] * 8 + change.index[1]] = change.delta
        else:
            delta[16 + change.index[0]] = change.delta
    return delta


def select_ordinary_context(examples, train_mask: np.ndarray, limit: int = ORDINARY_CONTEXT_EXAMPLES) -> np.ndarray:
    """Select one ordinary training example per provider deterministically."""
    candidates = np.flatnonzero(train_mask & (examples.targets == 0))
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
        raise ValueError(f"Only found {len(selected)} distinct ordinary providers")
    return np.asarray(selected, dtype=int)


def _example_record(examples, index: int, hidden: np.ndarray, logits: np.ndarray) -> dict[str, object]:
    return {
        "provider_code": str(examples.provider_codes[index]),
        "date": str(examples.dates[index].date()),
        "normalised_input": [round(float(value), 6) for value in examples.features[index]],
        "hidden_activations": [round(float(value), 6) for value in hidden[index]],
        "output_scores": [round(float(value), 6) for value in logits[index]],
        "correct_label": "high_demand" if int(examples.targets[index]) else "ordinary_demand",
    }


def build_prompt(examples, target_indices: np.ndarray, ordinary_indices: np.ndarray, hidden: np.ndarray, logits: np.ndarray, model: DemandCNN) -> tuple[str, str]:
    weights = model.classifier.weight.detach().numpy().round(6).tolist()
    biases = model.classifier.bias.detach().numpy().round(6).tolist()
    instructions = """You are proposing one small, direct weight patch for a trained neural network.

The goal is to improve the model's recognition of a repeated training-time failure: a genuinely high-demand month following a flat or falling six-month lead-in. This is a controlled experiment, not a request to guarantee success. If you do not see a sensible selective change, return an empty changes list.

The model receives 12 provider-relative monthly attendance values. It has Conv1d(1, 8, kernel_size=3, padding=1), ReLU, adaptive average pooling, and Linear(8, 2). The two final scores are [ordinary_demand, high_demand], and the larger score is the predicted class.

You may change only classifier.weight or classifier.bias. A weight index is [row, column], where row is 0 or 1 and column is 0 through 7. A bias index is [index], where index is 0 or 1. Return JSON with exactly this shape:
{"changes":[{"parameter":"classifier.weight","index":[0,0],"delta":0.01}],"explanation":"short explanation"}

Rules: at most 8 scalar changes; each absolute delta is at most 0.10; the total L2 norm of all deltas is at most 0.20; no duplicate parameter/index pairs; an empty patch is allowed. Do not write code. Do not change convolutional parameters. The explanation is a hypothesis, not evidence that the patch works.

The following examples are the only examples you may use. The first group contains 16 fixed target training examples, and the second group contains 8 fixed ordinary training examples for context. Validation and test examples and results are not provided. The analytical solution and its result are not provided.
"""
    data = {
        "editable_final_layer": {"classifier.weight": weights, "classifier.bias": biases},
        "target_training_examples": [_example_record(examples, int(index), hidden, logits) for index in target_indices],
        "ordinary_training_context": [_example_record(examples, int(index), hidden, logits) for index in ordinary_indices],
    }
    return instructions, json.dumps(data, indent=2)


def request_patch(instructions: str, numerical_data: str) -> WeightPatch:
    """Make exactly one structured-output request; callers do not retry failures."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=MODEL_NAME,
        reasoning={"effort": REASONING_EFFORT},
        input=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": numerical_data},
        ],
        text_format=WeightPatch,
    )
    if response.output_parsed is None:
        raise RuntimeError("The API returned no structured patch")
    return response.output_parsed


def safe_error_message(error: Exception) -> str:
    """Keep provider error details useful without retaining key fragments."""
    message = str(error)
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        message = message.replace(key, "[REDACTED]")
    return re.sub(r'(Incorrect API key provided: )[^.]+(?=\. You can find|$)', r'\1[REDACTED]', message)


def evaluate_methods(examples, masks, original_model: DemandCNN, llm_model: DemandCNN, analytical_model: DemandCNN, threshold: float) -> dict[str, object]:
    train_mask, validation_mask, test_mask = (masks[name] for name in ("train", "validation", "test"))
    target_pattern = recent_change(examples.features) <= 0
    models = {"original": original_model, "llm_edit": llm_model, "analytical_edit": analytical_model}
    probabilities_by_method: dict[str, np.ndarray] = {}
    margins_by_method: dict[str, np.ndarray] = {}
    for name, model in models.items():
        _, logits = hidden_and_logits(model, examples.features)
        probabilities_by_method[name] = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
        margins_by_method[name] = logits[:, 1] - logits[:, 0]

    logistic = train_logistic(examples.features[train_mask], examples.targets[train_mask])
    logistic_probabilities = logistic.predict_proba(examples.features[validation_mask])[:, 1]
    logistic_threshold = select_threshold(examples.targets[validation_mask], logistic_probabilities)
    validation_methods: dict[str, object] = {}
    for name, probabilities_ in probabilities_by_method.items():
        validation_methods[name] = pattern_metrics(examples, validation_mask, probabilities_, threshold, target_pattern)
    validation_methods["threshold_0.5"] = pattern_metrics(examples, validation_mask, probabilities_by_method["original"], 0.5, target_pattern)
    logistic_all = np.full(len(examples.targets), np.nan)
    logistic_all[validation_mask] = logistic_probabilities
    validation_methods["logistic"] = pattern_metrics(examples, validation_mask, logistic_all, logistic_threshold, target_pattern)

    target_indices = select_target_indices(examples, train_mask, target_pattern)
    target_training = {}
    for name, probabilities_ in probabilities_by_method.items():
        target_training[name] = {
            "examples": int(len(target_indices)),
            "original_target_correct": int(np.sum(probabilities_by_method["original"][target_indices] >= threshold)),
            "target_correct": int(np.sum(probabilities_[target_indices] >= threshold)),
            "corrected_from_original": int(np.sum((probabilities_by_method["original"][target_indices] < threshold) & (probabilities_[target_indices] >= threshold))),
        }
    score_changes = {
        name: {"target_mean_margin_change": float(np.mean(margins[target_indices] - margins_by_method["original"][target_indices]))}
        for name, margins in margins_by_method.items()
        if name != "original"
    }
    return {
        "validation": validation_methods,
        "target_training": target_training,
        "score_changes": score_changes,
        "test_used_for_edit": False,
        "test_examples_excluded": int(np.sum(test_mask)),
        "logistic_threshold": float(logistic_threshold),
    }


def plot_comparison(validation: dict[str, object], output_path: Path) -> None:
    labels = ["Original", "LLM edit", "Analytical edit", "Threshold 0.5", "Logistic"]
    keys = ["original", "llm_edit", "analytical_edit", "threshold_0.5", "logistic"]
    recall = [validation[key]["flat_or_falling"]["high_demand_recall"] for key in keys]
    false_alarm = [validation[key]["all"]["false_alarm_rate"] for key in keys]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, values, title, ylabel in ((axes[0], recall, "Target-pattern recall", "High-demand recall"), (axes[1], false_alarm, "Validation false alarms", "False-alarm rate")):
        axis.bar(x, values, color=["tab:gray", "tab:green", "tab:blue", "tab:orange", "tab:red"])
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw/provider_monthly"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/provider_baseline.pt"))
    parser.add_argument("--analytical-checkpoint", type=Path, default=Path("outputs/provider_analytical_model.pt"))
    parser.add_argument("--prompt", type=Path, default=Path("outputs/provider_llm_prompt.txt"))
    parser.add_argument("--response", type=Path, default=Path("outputs/provider_llm_response.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/provider_llm_edit_result.json"))
    parser.add_argument("--model-output", type=Path, default=Path("outputs/provider_llm_model.pt"))
    parser.add_argument("--figure", type=Path, default=Path("figures/provider_llm_edit.png"))
    parser.add_argument("--prepare-only", action="store_true", help="write the exact prompt and stop before the API request")
    args = parser.parse_args()

    panel = load_provider_panel(args.raw_directory)
    examples = make_provider_examples(panel)
    masks = split_provider_examples(examples)
    train_mask = masks["train"]
    original_model, cnn_threshold = load_saved_cnn(args.checkpoint)
    analytical_model = load_model_state(args.analytical_checkpoint)
    hidden, logits = hidden_and_logits(original_model, examples.features)
    target_indices = select_target_indices(examples, train_mask, recent_change(examples.features) <= 0)
    ordinary_indices = select_ordinary_context(examples, train_mask)
    instructions, numerical_data = build_prompt(examples, target_indices, ordinary_indices, hidden, logits, original_model)
    args.prompt.parent.mkdir(parents=True, exist_ok=True)
    args.prompt.write_text(instructions + "\n\nNUMERICAL DATA\n" + numerical_data, encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"status": "prompt_prepared", "prompt_file": str(args.prompt), "target_indices": [int(value) for value in target_indices], "ordinary_context_indices": [int(value) for value in ordinary_indices]}, indent=2))
        return
    result: dict[str, object] = {"model": MODEL_NAME, "reasoning_effort": REASONING_EFFORT, "constraints": {"max_changes": MAX_CHANGES, "max_abs_delta": MAX_ABS_DELTA, "max_l2_delta": MAX_L2_DELTA}, "target_indices": [int(value) for value in target_indices], "ordinary_context_indices": [int(value) for value in ordinary_indices], "api_called": False}
    try:
        patch = request_patch(instructions, numerical_data)
        result["api_called"] = True
        result["response"] = patch.model_dump()
        validated, errors = validate_patch(patch)
        result["validation"] = {"valid": not errors, "errors": errors}
        args.response.write_text(json.dumps(patch.model_dump(), indent=2), encoding="utf-8")
        if validated is None:
            result["status"] = "invalid_patch"
        else:
            delta = patch_to_delta(validated)
            edited_model = apply_final_layer_delta(original_model, delta)
            evaluation = evaluate_methods(examples, masks, original_model, edited_model, analytical_model, cnn_threshold)
            result["status"] = "evaluated"
            result["patch_summary"] = {"changed_parameters": int(np.count_nonzero(delta)), "l2_norm": float(np.linalg.norm(delta)), "max_abs_change": float(np.max(np.abs(delta))) if np.any(delta) else 0.0}
            result["evaluation"] = evaluation
            torch.save({"model_state": edited_model.state_dict(), "source_checkpoint": str(args.checkpoint), "patch": patch.model_dump()}, args.model_output)
            plot_comparison(evaluation["validation"], args.figure)
    except Exception as error:
        result["status"] = "api_error"
        result["error"] = {"type": type(error).__name__, "message": safe_error_message(error)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "response_file": str(args.response), "result_file": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
