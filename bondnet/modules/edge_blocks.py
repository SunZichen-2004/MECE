import torch
from torch import nn

from .scatter import scatter_sum


class MagneticOrderLinear(nn.Module):
    """Channel mixing shared across both components of each nonzero magnetic order."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.channel_mixing = nn.Linear(input_channels, output_channels, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.channel_mixing(values)


class EdgeInteractionBlock(nn.Module):
    def __init__(self, number_of_channels: int, number_of_radial_functions: int, number_of_blocks: int):
        super().__init__()
        self.magnetic_order_linears = nn.ModuleList(
            [MagneticOrderLinear(number_of_channels, number_of_channels) for _ in range(number_of_blocks)]
        )
        self.radial_gate = nn.Sequential(
            nn.Linear(number_of_radial_functions + number_of_channels, number_of_channels),
            nn.SiLU(),
            nn.Linear(number_of_channels, number_of_channels),
            nn.Sigmoid(),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(number_of_channels * 2 + number_of_radial_functions, number_of_channels),
            nn.SiLU(),
            nn.Linear(number_of_channels, number_of_channels),
        )

    def forward(
        self,
        edge_blocks: list[torch.Tensor],
        radial_values: torch.Tensor,
        sender_atom_indices: torch.Tensor,
        receiver_atom_indices: torch.Tensor,
        number_of_atoms: int,
    ) -> list[torch.Tensor]:
        scalar_edges = edge_blocks[0].squeeze(1)
        atom_aggregates = scatter_sum(scalar_edges, receiver_atom_indices, number_of_atoms)
        atom_context = atom_aggregates[sender_atom_indices] + atom_aggregates[receiver_atom_indices]
        gate_values = self.radial_gate(torch.cat((radial_values, atom_context), dim=-1))
        updated_blocks: list[torch.Tensor] = []
        scalar_increment = self.scalar_update(torch.cat((scalar_edges, atom_context, radial_values), dim=-1))
        updated_blocks.append((scalar_edges + scalar_increment).unsqueeze(1))
        for block_index, block_values in enumerate(edge_blocks[1:], start=1):
            mixed_values = self.magnetic_order_linears[block_index](block_values)
            updated_blocks.append(block_values + mixed_values * gate_values.unsqueeze(1))
        return updated_blocks
