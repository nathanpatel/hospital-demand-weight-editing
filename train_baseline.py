"""Train the first NHS demand classification baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from data import make_dataset, load_monthly_attendances, split_dataset
from model import DemandCNN


def train_cnn(features: np.ndarray, targets: np.ndarray, seed: int = 7) -> DemandCNN:
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DemandCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    class_counts = np.bincount(targets.astype(np.int64), minlength=2)
    class_weights = len(targets) / (2.0 * np.maximum(class_counts, 1))
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float())
    inputs = torch.from_numpy(features)
    labels = torch.from_numpy(targets.astype(np.int64))
    for _ in range(250):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def train_logistic(features: np.ndarray, targets: np.ndarray) -> LogisticRegression:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=7))
    model.fit(features, targets.astype(np.int64))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/monthly_ae_time_series.xls"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/baseline_training.json"))
    args = parser.parse_args()

    frame = load_monthly_attendances(args.data)
    dataset = make_dataset(frame)
    train, _ = split_dataset(dataset)
    cnn = train_cnn(train["features"], train["targets"])
    logistic = train_logistic(train["features"], train["targets"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": cnn.state_dict(), "threshold_quantile": 0.75}, args.output)
    args.report.write_text(
        json.dumps(
            {
                "data_source": str(args.data),
                "rows": len(frame),
                "first_date": str(frame["date"].min().date()),
                "last_date": str(frame["date"].max().date()),
                "window_size": 12,
                "high_demand_definition": "next month at or above the preceding 12-month 75th percentile",
                "train_examples": len(train["features"]),
                "cnn_parameters": sum(parameter.numel() for parameter in cnn.parameters()),
                "logistic_coefficients": logistic[-1].coef_.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved CNN checkpoint to {args.output}")
    print(f"Saved training summary to {args.report}")


if __name__ == "__main__":
    main()
