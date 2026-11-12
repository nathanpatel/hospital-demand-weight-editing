import numpy as np
import torch

from analytical_provider_edit import apply_final_layer_delta, optimal_sparse_delta
from model import DemandCNN


def test_sparse_optimizer_respects_all_constraints() -> None:
    delta, value = optimal_sparse_delta(np.arange(18, dtype=float) - 8.0)

    assert value > 0
    assert np.count_nonzero(np.abs(delta) > 1e-9) <= 8
    assert np.max(np.abs(delta)) <= 0.10 + 1e-8
    assert np.linalg.norm(delta) <= 0.20 + 1e-8


def test_final_layer_edit_does_not_mutate_original_model() -> None:
    model = DemandCNN()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    delta = np.zeros(18)
    delta[0] = 0.05

    edited = apply_final_layer_delta(model, delta)

    assert torch.equal(model.state_dict()["classifier.weight"], original["classifier.weight"])
    assert torch.equal(model.state_dict()["classifier.bias"], original["classifier.bias"])
    assert not torch.equal(edited.state_dict()["classifier.weight"], original["classifier.weight"])
