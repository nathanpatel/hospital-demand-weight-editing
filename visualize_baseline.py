"""Create the first useful plots for the NHS demand baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data import load_monthly_attendances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/monthly_ae_time_series.xls"))
    parser.add_argument("--report", type=Path, default=Path("outputs/baseline_evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("figures/baseline_overview.png"))
    args = parser.parse_args()

    frame = load_monthly_attendances(args.data)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    predictions = report["test_predictions"]
    dates = pd.to_datetime(predictions["dates"])

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(frame["date"], frame["total_attendances"], color="tab:blue", linewidth=1.4)
    axes[0].set_title("NHS England monthly A&E attendances")
    axes[0].set_ylabel("Attendances")

    axes[1].plot(dates, predictions["actual"], "o-", label="actual high demand")
    axes[1].plot(dates, predictions["seasonal"], "o--", label="seasonal persistence")
    axes[1].plot(dates, predictions["cnn"], "o--", label="CNN")
    axes[1].set_title("Held-out high-demand predictions")
    axes[1].set_ylabel("High demand (1=yes)")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(ncol=3)
    figure.savefig(args.output, dpi=160)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
