"""Load the NHS England monthly A&E series and build leakage-safe examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE = 12
TRAIN_FRACTION = 0.75
HIGH_QUANTILE = 0.75


@dataclass(frozen=True)
class DemandDataset:
    dates: pd.DatetimeIndex
    values: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    target_values: np.ndarray
    thresholds: np.ndarray
    split_index: int


def load_monthly_attendances(path: Path) -> pd.DataFrame:
    """Read the England-level Total Attendances column from NHS England's workbook."""
    raw = pd.read_excel(path, sheet_name="Activity", header=None)
    rows = raw.iloc[14:, [1, 5]].copy()
    rows.columns = ["date", "total_attendances"]
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["total_attendances"] = pd.to_numeric(rows["total_attendances"], errors="coerce")
    rows = rows.dropna().sort_values("date").drop_duplicates("date")
    if rows.empty:
        raise ValueError("No monthly attendance rows were found")
    return rows.reset_index(drop=True)


def make_dataset(
    frame: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
    train_fraction: float = TRAIN_FRACTION,
    high_quantile: float = HIGH_QUANTILE,
) -> DemandDataset:
    """Create past-only windows and a chronological train/test split."""
    if len(frame) <= window_size + 2:
        raise ValueError("The series is too short for the requested window")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")

    dates = pd.DatetimeIndex(frame["date"])
    values = frame["total_attendances"].to_numpy(dtype=np.float32)
    split_index = int(len(values) * train_fraction)
    target_values = values[window_size:]
    thresholds = np.asarray(
        [np.quantile(values[index - window_size : index], high_quantile)
         for index in range(window_size, len(values))],
        dtype=np.float32,
    )

    windows = np.stack(
        [values[index - window_size : index] for index in range(window_size, len(values))]
    )
    # The scale of national attendance counts changes over time. Expressing each
    # window relative to its most recent observed month keeps the input numerical
    # scale stable without using any future information.
    features = windows / windows[:, -1:,] - 1.0
    targets = (target_values >= thresholds).astype(np.float32)
    return DemandDataset(
        dates=dates[window_size:],
        values=values,
        features=features,
        targets=targets,
        target_values=target_values,
        thresholds=thresholds,
        split_index=split_index - window_size,
    )


def split_dataset(dataset: DemandDataset) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return chronological train and test dictionaries."""
    split = dataset.split_index
    train = {
        "features": dataset.features[:split],
        "targets": dataset.targets[:split],
        "target_values": dataset.target_values[:split],
        "dates": dataset.dates[:split].to_numpy(),
    }
    test = {
        "features": dataset.features[split:],
        "targets": dataset.targets[split:],
        "target_values": dataset.target_values[split:],
        "dates": dataset.dates[split:].to_numpy(),
    }
    return train, test


def seasonal_persistence_predictions(dataset: DemandDataset) -> np.ndarray:
    """Predict each target month using the value from twelve months earlier."""
    values = dataset.values
    target_indices = np.arange(WINDOW_SIZE, len(values))
    return values[target_indices - 12]
