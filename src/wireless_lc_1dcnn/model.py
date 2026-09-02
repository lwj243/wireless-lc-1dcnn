from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBranch(nn.Module):
    def __init__(self, channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class MultiScaleFrequencyBlock(nn.Module):
    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        branch_channels = channels // 2
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableBranch(channels, branch_channels, 3),
                DepthwiseSeparableBranch(channels, branch_channels, 7),
                DepthwiseSeparableBranch(channels, branch_channels, 15),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv1d(branch_channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(6, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(x + self.project(multi_scale))


class FrequencyAttentionPooling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(x), dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        return pooled, weights.squeeze(1)


class LightweightEISNet(nn.Module):
    """Frequency-aware multi-task 1D-CNN for wireless LC spectrum decoding."""

    def __init__(self, input_channels: int = 5, n_states: int = 3) -> None:
        super().__init__()
        channels = 24
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(6, channels),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            MultiScaleFrequencyBlock(channels),
            nn.AvgPool1d(kernel_size=2),
            MultiScaleFrequencyBlock(channels),
        )
        self.attention_pool = FrequencyAttentionPooling(channels)
        self.shared = nn.Sequential(
            nn.Linear(channels * 2, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.concentration_head = nn.Sequential(nn.Linear(48, 1), nn.Sigmoid())
        self.state_head = nn.Linear(48, n_states)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.stem(x))
        attended, attention = self.attention_pool(encoded)
        maximum = torch.amax(encoded, dim=-1)
        embedding = self.shared(torch.cat([attended, maximum], dim=1))
        return {
            "concentration": self.concentration_head(embedding).squeeze(1),
            "state_logits": self.state_head(embedding),
            "attention": attention,
        }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
