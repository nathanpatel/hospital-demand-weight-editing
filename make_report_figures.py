"""Generate the report figures from saved results and the existing checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from model import DemandCNN
from provider_data import load_provider_panel, make_provider_examples


ROOT = Path(__file__).parent
FIGURES = ROOT / "figures"


def load_json(name: str) -> dict[str, object]:
    return json.loads((ROOT / "outputs" / name).read_text(encoding="utf-8"))


def style_axes(axes) -> None:
    for axis in np.ravel(axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)


def task_diagram() -> None:
    figure, axis = plt.subplots(figsize=(10, 3.1))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.3, 0.65, 2.0, 1.7, "Previous 12 months\nof one provider", "attendance history"),
        (3.0, 0.65, 2.0, 1.7, "Small 1D CNN", "convolution + pooling"),
        (5.7, 0.65, 1.6, 1.7, "Two scores", "ordinary / high"),
        (8.0, 0.65, 1.7, 1.7, "Prediction", "whichever score is larger"),
    ]
    colours = ["#e8eef7", "#e8f3ed", "#f7efe1", "#f2e8f4"]
    for (x, y, width, height, title, subtitle), colour in zip(boxes, colours):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.08", facecolor=colour, edgecolor="#425466", linewidth=1.1)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + 1.35, title, ha="center", va="center", fontsize=11, weight="bold")
        axis.text(x + width / 2, y + 0.72, subtitle, ha="center", va="center", fontsize=9, color="#425466")
    for start, end in ((2.3, 3.0), (5.0, 5.7), (7.3, 8.0)):
        axis.add_patch(FancyArrowPatch((start, 1.5), (end, 1.5), arrowstyle="-|>", mutation_scale=16, linewidth=1.2, color="#425466"))
    axis.text(1.3, 2.63, "input", ha="center", fontsize=9, color="#425466")
    axis.text(6.5, 2.63, "ordinary demand\nhigh demand", ha="center", fontsize=9, color="#425466")
    figure.tight_layout()
    figure.savefig(FIGURES / "report_task.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def baseline_comparison() -> None:
    result = load_json("provider_baseline_evaluation.json")
    methods = ["CNN", "Logistic\nregression", "Seasonal\npersistence"]
    values = [result["methods"][key]["roc_auc"] for key in ("cnn", "logistic", "seasonal")]
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    bars = axis.bar(methods, values, color=["#3d6f9e", "#d58b36", "#8a8f98"], width=0.58)
    axis.axhline(0.5, color="#444444", linestyle="--", linewidth=1, label="chance")
    axis.set_ylim(0.45, 0.72)
    axis.set_ylabel("ROC-AUC on final six months")
    axis.set_title("The provider-level CNN learned a measurable signal")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    axis.legend(frameon=False, loc="upper left")
    style_axes([axis])
    figure.tight_layout()
    figure.savefig(FIGURES / "report_baseline_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def failure_example() -> None:
    panel = load_provider_panel(ROOT / "data/raw/provider_monthly")
    provider = panel[panel["provider_code"] == "AAH"].sort_values("date")
    target_date = provider.loc[provider["date"] == "2025-07-01", "date"].iloc[0]
    target_position = int(np.flatnonzero(provider["date"].to_numpy() == target_date.to_datetime64())[0])
    target_row = provider.iloc[target_position]
    history = provider.iloc[target_position - 12 : target_position]["attendance"].to_numpy()
    threshold = float(np.quantile(history, 0.75))
    dates = [str(value)[:7] for value in provider.iloc[target_position - 12 : target_position + 1]["date"]]
    values = np.concatenate([history, [float(target_row["attendance"])]])
    figure, axis = plt.subplots(figsize=(8.2, 3.9))
    axis.plot(range(13), values, marker="o", color="#3d6f9e", linewidth=2, label="A\&E attendances")
    axis.axhline(threshold, color="#c26d3b", linestyle="--", linewidth=1.5, label=f"75th-percentile threshold ({threshold:.0f})")
    axis.scatter([12], [values[-1]], s=80, color="#b64b4b", zorder=4, label="Actual July 2025 demand")
    axis.annotate("CNN predicted ordinary demand", xy=(12, values[-1]), xytext=(8.1, values[-1] + 180), arrowprops={"arrowstyle": "->", "color": "#b64b4b"}, fontsize=9, color="#7b3030")
    axis.set_xticks(range(13), dates, rotation=40, ha="right")
    axis.set_ylabel("Monthly attendances")
    axis.set_title("A real missed high-demand month: provider AAH")
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    style_axes([axis])
    figure.tight_layout()
    figure.savefig(FIGURES / "report_failure_example.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def edit_mechanism() -> None:
    panel = load_provider_panel(ROOT / "data/raw/provider_monthly")
    examples = make_provider_examples(panel)
    checkpoint = torch.load(ROOT / "outputs/provider_baseline.pt", map_location="cpu", weights_only=True)
    model = DemandCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    index = int(np.flatnonzero((examples.provider_codes == "AAH") & (examples.dates == np.datetime64("2025-01-01")))[0])
    with torch.no_grad():
        hidden = model.features(torch.from_numpy(examples.features[index : index + 1]).float().unsqueeze(1)).squeeze(-1)[0].numpy()
    ordinary = model.classifier.weight.detach().numpy()[0]
    high = model.classifier.weight.detach().numpy()[1]
    contributions = hidden * (high - ordinary)
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].bar(np.arange(1, 9), hidden, color="#6e9f8a")
    axes[0].set_xlabel("Hidden channel")
    axes[0].set_ylabel("Activation")
    axes[0].set_title("Hidden activations $h$")
    axes[1].bar(np.arange(1, 9), contributions, color=["#b64b4b" if value < 0 else "#3d6f9e" for value in contributions])
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_xlabel("Hidden channel")
    axes[1].set_ylabel("Contribution to high $-$ ordinary score")
    axes[1].set_title("Final-layer margin contributions")
    style_axes(axes)
    figure.suptitle("The final layer turns hidden activations into a decision", y=1.03, fontsize=12)
    figure.tight_layout()
    figure.savefig(FIGURES / "report_edit_mechanism.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def edit_comparison() -> None:
    result = load_json("provider_llm_edit_result.json")
    methods = ["Original", "Threshold", "Analytical", "LLM edit", "Logistic"]
    keys = ["original", "threshold_0.5", "analytical_edit", "llm_edit", "logistic"]
    recall = [result["evaluation"]["validation"][key]["flat_or_falling"]["high_demand_recall"] for key in keys]
    false_alarms = [result["evaluation"]["validation"][key]["all"]["false_alarms"] for key in keys]
    balanced = [result["evaluation"]["validation"][key]["all"]["balanced_accuracy"] for key in keys]
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    colours = ["#777777", "#d58b36", "#b64b4b", "#3d6f9e", "#6e9f8a"]
    x = np.arange(len(methods))
    for axis, values, title, ylabel in ((axes[0], recall, "Target-pattern recall", "Recall"), (axes[1], false_alarms, "False alarms", "Count on validation"), (axes[2], balanced, "Overall balance", "Balanced accuracy")):
        bars = axis.bar(x, values, color=colours)
        axis.set_xticks(x, methods, rotation=30, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        if axis is axes[0] or axis is axes[2]:
            axis.set_ylim(0, 1)
        for bar, value in zip(bars, values):
            label = f"{value:.0%}" if axis is not axes[1] else f"{value:.0f}"
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.025 if axis is not axes[1] else 8), label, ha="center", va="bottom", fontsize=8)
    style_axes(axes)
    figure.suptitle("The edits improve recall by moving the wider decision boundary", y=1.03, fontsize=12)
    figure.tight_layout()
    figure.savefig(FIGURES / "report_edit_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    task_diagram()
    baseline_comparison()
    failure_example()
    edit_mechanism()
    edit_comparison()


if __name__ == "__main__":
    main()
