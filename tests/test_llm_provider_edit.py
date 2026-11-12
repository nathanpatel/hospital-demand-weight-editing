import numpy as np
import torch

from llm_provider_edit import (
    PatchChange,
    WeightPatch,
    apply_final_layer_delta,
    patch_to_delta,
    validate_patch,
)
from model import DemandCNN


def valid_patch() -> WeightPatch:
    return WeightPatch(changes=[PatchChange(parameter="classifier.weight", index=[1, 2], delta=0.05)], explanation="test")


def test_valid_patch_is_converted_to_final_layer_delta() -> None:
    delta = patch_to_delta(valid_patch())
    assert delta[10] == 0.05
    assert np.count_nonzero(delta) == 1


def test_invalid_parameter_name_is_rejected() -> None:
    patch, errors = validate_patch({"changes": [{"parameter": "features.0.weight", "index": [0, 0], "delta": 0.01}], "explanation": "bad"})
    assert patch is None
    assert errors


def test_invalid_index_is_rejected() -> None:
    patch, errors = validate_patch({"changes": [{"parameter": "classifier.bias", "index": [2], "delta": 0.01}], "explanation": "bad"})
    assert patch is None
    assert any("index" in error for error in errors)


def test_duplicate_changes_are_rejected() -> None:
    patch, errors = validate_patch({"changes": [{"parameter": "classifier.bias", "index": [1], "delta": 0.01}, {"parameter": "classifier.bias", "index": [1], "delta": 0.01}], "explanation": "bad"})
    assert patch is None
    assert any("duplicate" in error for error in errors)


def test_per_change_and_total_limits_are_rejected() -> None:
    too_large, errors = validate_patch({"changes": [{"parameter": "classifier.bias", "index": [1], "delta": 0.11}], "explanation": "bad"})
    assert too_large is None
    assert errors
    many = {"changes": [{"parameter": "classifier.weight", "index": [0, i], "delta": 0.1} for i in range(8)], "explanation": "bad"}
    patch, errors = validate_patch(many)
    assert patch is None
    assert any("L2" in error for error in errors)


def test_apply_patch_keeps_original_model_unchanged() -> None:
    model = DemandCNN()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    edited = apply_final_layer_delta(model, patch_to_delta(valid_patch()))
    assert torch.equal(model.state_dict()["classifier.weight"], original["classifier.weight"])
    assert torch.equal(model.state_dict()["classifier.bias"], original["classifier.bias"])
    assert not torch.equal(edited.state_dict()["classifier.weight"], original["classifier.weight"])
