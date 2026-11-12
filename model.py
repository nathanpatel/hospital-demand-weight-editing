"""Small 1D CNN for monthly high-demand classification."""

from __future__ import annotations

import torch
from torch import nn


class DemandCNN(nn.Module):
    """A deliberately small model whose parameters remain easy to inspect."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.features(inputs.unsqueeze(1)).squeeze(-1)
        return self.classifier(hidden)
