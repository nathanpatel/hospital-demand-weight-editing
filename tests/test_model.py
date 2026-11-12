import torch

from model import DemandCNN


def test_cnn_returns_two_action_scores_for_each_window() -> None:
    model = DemandCNN()
    output = model(torch.zeros(4, 12))

    assert output.shape == (4, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 50
