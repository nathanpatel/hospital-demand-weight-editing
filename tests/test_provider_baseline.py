import numpy as np

from provider_baseline import metrics, select_threshold


def test_threshold_is_selected_from_validation_scores() -> None:
    threshold = select_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))

    assert threshold == 0.5


def test_provider_metrics_report_false_alarms_and_recall() -> None:
    result = metrics(np.array([0, 0, 1]), np.array([0.9, 0.1, 0.8]), 0.5)

    assert result["false_alarms"] == 1
    assert result["high_demand_recall"] == 1.0
