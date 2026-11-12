import numpy as np

from diagnose_cnn import classification_metrics, select_threshold, threshold_curve


def test_threshold_selection_uses_validation_predictions_and_returns_metrics() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, metrics = select_threshold(targets, probabilities)

    assert threshold == 0.5
    assert metrics["balanced_accuracy"] == 1.0


def test_metrics_count_false_alarms() -> None:
    metrics = classification_metrics(np.array([0, 0, 1]), np.array([0.9, 0.1, 0.8]), 0.5)

    assert metrics["false_alarms"] == 1
    assert metrics["high_demand_recall"] == 1.0


def test_threshold_curve_has_expected_endpoints() -> None:
    curve = threshold_curve(np.array([0, 1]), np.array([0.2, 0.8]))

    assert curve[0]["high_predictions"] == 2
    assert curve[-1]["high_predictions"] == 0
