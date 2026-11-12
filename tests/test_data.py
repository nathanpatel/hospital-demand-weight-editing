import numpy as np
import pandas as pd

from data import make_dataset, seasonal_persistence_predictions


def test_dataset_uses_only_previous_values() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            "total_attendances": np.arange(20, dtype=float),
        }
    )
    dataset = make_dataset(frame, window_size=3, train_fraction=0.7, high_quantile=0.5)

    assert dataset.features[0].tolist() == [-1.0, -0.5, 0.0]
    assert dataset.target_values[0] == 3.0
    assert dataset.thresholds[0] == 1.0
    assert dataset.split_index == 11


def test_seasonal_persistence_uses_twelve_months_earlier() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            "total_attendances": np.arange(20, dtype=float),
        }
    )
    dataset = make_dataset(frame, window_size=12)

    assert seasonal_persistence_predictions(dataset).tolist() == list(np.arange(8, dtype=float))
