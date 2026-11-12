"""Build a provider-level, leakage-safe NHS demand classification dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ATTENDANCE_COLUMNS = [
    "A&E attendances Type 1",
    "A&E attendances Type 2",
    "A&E attendances Other A&E Department",
    "A&E attendances Booked Appointments Type 1",
    "A&E attendances Booked Appointments Type 2",
    "A&E attendances Booked Appointments Other Department",
]
WINDOW_SIZE = 12
VALIDATION_START = pd.Timestamp("2025-04-01")
TEST_START = pd.Timestamp("2025-10-01")


@dataclass(frozen=True)
class ProviderExamples:
    provider_codes: np.ndarray
    dates: pd.DatetimeIndex
    features: np.ndarray
    targets: np.ndarray
    target_values: np.ndarray
    thresholds: np.ndarray
    seasonal_values: np.ndarray


def _period_from_label(label: str) -> pd.Timestamp:
    label = str(label).strip()
    if label.upper() in {"TOTAL", "TOTAl".upper()}:
        raise ValueError("Aggregate total rows do not represent a provider-month")
    return pd.to_datetime(label.replace("MSitAE-", ""), format="%B-%Y")


def load_provider_panel(raw_directory: Path) -> pd.DataFrame:
    """Load and aggregate monthly provider files, excluding published totals."""
    frames: list[pd.DataFrame] = []
    files = sorted(raw_directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No provider CSV files found in {raw_directory}")

    expected_columns = set(ATTENDANCE_COLUMNS) | {"Period", "Org Code", "Org name"}
    for path in files:
        frame = pd.read_csv(path)
        missing = expected_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame[~frame["Period"].astype(str).str.strip().str.upper().eq("TOTAL")].copy()
        frame["date"] = frame["Period"].map(_period_from_label)
        frame["attendance"] = frame[ATTENDANCE_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        frame = frame.rename(columns={"Org Code": "provider_code", "Org name": "provider_name"})
        frames.append(frame[["date", "provider_code", "provider_name", "attendance"]])

    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["provider_code", "date"]).any():
        raise ValueError("Duplicate provider-month rows were found")
    if panel["attendance"].isna().any() or (panel["attendance"] < 0).any():
        raise ValueError("Provider attendance contains missing or negative values")

    complete_dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="MS")
    counts = panel.groupby("provider_code")["date"].nunique()
    complete_codes = counts[counts == len(complete_dates)].index
    panel = panel[panel["provider_code"].isin(complete_codes)].copy()
    panel = panel.sort_values(["provider_code", "date"]).reset_index(drop=True)
    panel.attrs["expected_dates"] = complete_dates
    panel.attrs["complete_provider_count"] = len(complete_codes)
    return panel


def make_provider_examples(panel: pd.DataFrame, window_size: int = WINDOW_SIZE) -> ProviderExamples:
    """Create provider-relative windows and future labels without leakage."""
    rows: list[dict[str, object]] = []
    feature_rows: list[np.ndarray] = []
    for provider_code, provider_frame in panel.groupby("provider_code", sort=True):
        provider_frame = provider_frame.sort_values("date")
        values = provider_frame["attendance"].to_numpy(dtype=np.float32)
        dates = pd.DatetimeIndex(provider_frame["date"])
        for target_index in range(window_size, len(provider_frame)):
            window = values[target_index - window_size : target_index]
            scale = float(np.median(window))
            if scale <= 0:
                continue
            threshold = float(np.quantile(window, 0.75))
            feature_rows.append(window / scale - 1.0)
            rows.append(
                {
                    "provider_code": provider_code,
                    "date": dates[target_index],
                    "target": int(values[target_index] >= threshold),
                    "target_value": float(values[target_index]),
                    "threshold": threshold,
                    "seasonal_value": float(values[target_index - 12]),
                }
            )

    metadata = pd.DataFrame(rows)
    return ProviderExamples(
        provider_codes=metadata["provider_code"].to_numpy(),
        dates=pd.DatetimeIndex(metadata["date"]),
        features=np.asarray(feature_rows, dtype=np.float32),
        targets=metadata["target"].to_numpy(dtype=np.int64),
        target_values=metadata["target_value"].to_numpy(dtype=np.float32),
        thresholds=metadata["threshold"].to_numpy(dtype=np.float32),
        seasonal_values=metadata["seasonal_value"].to_numpy(dtype=np.float32),
    )


def split_provider_examples(examples: ProviderExamples) -> dict[str, np.ndarray]:
    """Return fixed calendar train, validation and test masks."""
    return {
        "train": examples.dates < VALIDATION_START,
        "validation": (examples.dates >= VALIDATION_START) & (examples.dates < TEST_START),
        "test": examples.dates >= TEST_START,
    }
