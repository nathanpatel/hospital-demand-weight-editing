import numpy as np
import pandas as pd

from provider_data import make_provider_examples, split_provider_examples


def test_provider_examples_are_relative_to_previous_window() -> None:
    dates = pd.date_range("2023-04-01", periods=14, freq="MS")
    panel = pd.DataFrame({"date": dates, "provider_code": "A", "provider_name": "Example", "attendance": np.arange(1, 15, dtype=float)})

    examples = make_provider_examples(panel)

    assert examples.features.shape == (2, 12)
    assert np.isclose(np.median(examples.features[0]), 0.0, atol=1e-6)
    assert examples.dates[0] == pd.Timestamp("2024-04-01")


def test_provider_split_uses_shared_calendar_boundaries() -> None:
    dates = pd.date_range("2023-04-01", periods=30, freq="MS")
    panel = pd.DataFrame({"date": dates, "provider_code": "A", "provider_name": "Example", "attendance": np.arange(1, 31, dtype=float)})

    masks = split_provider_examples(make_provider_examples(panel))

    assert masks["train"].sum() == 12
    assert masks["validation"].sum() == 6
    assert masks["test"].sum() == 0
