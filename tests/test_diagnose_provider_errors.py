import numpy as np

from diagnose_provider_errors import recent_change


def test_recent_change_compares_last_three_months_with_previous_three() -> None:
    features = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]])

    assert np.isclose(recent_change(features)[0], 0.1)
