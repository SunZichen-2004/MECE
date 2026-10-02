import math

import torch
from torch import nn


class RadialBasis(nn.Module):
    """MACE-style Bessel basis multiplied by a smooth polynomial cutoff."""

    def __init__(self, number_of_radial_functions: int, cutoff_radius: float, cutoff_order: int = 5):
        super().__init__()
        self.number_of_radial_functions = number_of_radial_functions
        self.cutoff_radius = cutoff_radius
        self.cutoff_order = cutoff_order
        frequencies = torch.arange(1, number_of_radial_functions + 1) * math.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        scaled_distances = distances.unsqueeze(-1) / self.cutoff_radius
        bessel_values = (
            math.sqrt(2.0 / self.cutoff_radius)
            * torch.sin(self.frequencies * scaled_distances)
            / distances.unsqueeze(-1).clamp_min(1.0e-8)
        )
        cutoff_values = self._polynomial_cutoff(scaled_distances)
        return bessel_values * cutoff_values

    def _polynomial_cutoff(self, scaled_distances: torch.Tensor) -> torch.Tensor:
        order = self.cutoff_order
        inside = (
            1.0
            - ((order + 1) * (order + 2) / 2.0) * scaled_distances**order
            + order * (order + 2) * scaled_distances ** (order + 1)
            - (order * (order + 1) / 2.0) * scaled_distances ** (order + 2)
        )
        return torch.where(scaled_distances < 1.0, inside, torch.zeros_like(inside))
