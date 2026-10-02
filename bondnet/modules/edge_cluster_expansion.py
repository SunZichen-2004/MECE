import torch
from torch import nn

from .scatter import scatter_sum
from .spherical_harmonics import real_wigner_transport_matrices


def split_packed_state(state_values: torch.Tensor, angular_momentum: int) -> list[torch.Tensor]:
    """Split [items, channels, 2*l+1] into real SO(2) blocks."""
    harmonic_values = state_values.transpose(1, 2)
    blocks = [harmonic_values[:, 0:1]]
    for absolute_order in range(1, angular_momentum + 1):
        start = 1 + 2 * (absolute_order - 1)
        blocks.append(harmonic_values[:, start : start + 2])
    return blocks


def multiply_so2_blocks(
    left_values: torch.Tensor,
    left_absolute_order: int,
    right_values: torch.Tensor,
    right_absolute_order: int,
) -> list[tuple[int, torch.Tensor]]:
    """Channelwise real SO(2) tensor product.

    Inputs are [items, 1 or 2, channels]. Positive orders store [cos, sin].
    """
    if left_absolute_order == 0:
        return [(right_absolute_order, left_values * right_values)]
    if right_absolute_order == 0:
        return [(left_absolute_order, left_values * right_values)]
    left_cosine, left_sine = left_values.unbind(dim=1)
    right_cosine, right_sine = right_values.unbind(dim=1)
    sum_values = torch.stack((
        left_cosine * right_cosine - left_sine * right_sine,
        left_cosine * right_sine + left_sine * right_cosine,
    ), dim=1)
    products = [(left_absolute_order + right_absolute_order, sum_values)]
    difference_order = abs(left_absolute_order - right_absolute_order)
    difference_cosine = left_cosine * right_cosine + left_sine * right_sine
    if difference_order == 0:
        difference_values = difference_cosine.unsqueeze(1)
    elif left_absolute_order > right_absolute_order:
        difference_values = torch.stack((
            difference_cosine,
            left_sine * right_cosine - left_cosine * right_sine,
        ), dim=1)
    else:
        difference_values = torch.stack((
            difference_cosine,
            left_cosine * right_sine - left_sine * right_cosine,
        ), dim=1)
    products.append((difference_order, difference_values))
    return products


def unpack_all_orders(
    states: list[torch.Tensor],
) -> list[tuple[int, int, torch.Tensor]]:
    result = []
    for angular_momentum, values in enumerate(states):
        result.extend(
            (angular_momentum, absolute_order, block)
            for absolute_order, block in enumerate(split_packed_state(values, angular_momentum))
        )
    return result


def pack_all_orders(
    blocks: dict[tuple[int, int], torch.Tensor], maximum_angular_momentum: int
) -> list[torch.Tensor]:
    packed = []
    for angular_momentum in range(maximum_angular_momentum + 1):
        pieces = [blocks[(angular_momentum, order)] for order in range(angular_momentum + 1)]
        packed.append(torch.cat(pieces, dim=1).transpose(1, 2))
    return packed


class EdgeClusterExpansionBlock(nn.Module):
    """Relative-frame message and explicit SO(2) edge product basis."""

    def __init__(self, number_of_channels: int, maximum_angular_momentum: int, maximum_correlation_order: int):
        super().__init__()
        self.maximum_angular_momentum = maximum_angular_momentum
        self.maximum_correlation_order = maximum_correlation_order
        self.source_channel_mixing = nn.ModuleList([
            nn.Linear(number_of_channels, number_of_channels, bias=False)
            for _ in range(maximum_angular_momentum + 1)
        ])
        self.geometry_channel_mixing = nn.ModuleList([
            nn.Linear(number_of_channels, number_of_channels, bias=False)
            for _ in range(maximum_angular_momentum + 1)
        ])
        self.output_channel_mixing = nn.ModuleDict({
            f"{angular_momentum}_{absolute_order}": nn.Linear(
                number_of_channels * maximum_correlation_order, number_of_channels, bias=False
            )
            for angular_momentum in range(maximum_angular_momentum + 1)
            for absolute_order in range(angular_momentum + 1)
        })
        self.residual_weights = nn.Parameter(torch.ones(maximum_angular_momentum + 1))
        self.message_weights = nn.Parameter(torch.full((maximum_angular_momentum + 1,), 0.1))

    def forward(
        self,
        states: list[torch.Tensor],
        neighbor_geometry: list[torch.Tensor],
        relative_cartesian_rotations: torch.Tensor,
        center_edge_indices: torch.Tensor,
        neighbor_edge_indices: torch.Tensor,
        number_of_edges: int,
    ) -> list[torch.Tensor]:
        """All state tensors are [edges, channels, 2*l+1] in local frames."""
        transported_states = []
        mixed_geometry = []
        for angular_momentum, state_values in enumerate(states):
            wigner_transport = real_wigner_transport_matrices(
                relative_cartesian_rotations, angular_momentum
            )  # [edge_pairs, 2*l+1, 2*l+1]
            selected_state = self.source_channel_mixing[angular_momentum](
                state_values[neighbor_edge_indices].transpose(1, 2)
            ).transpose(1, 2)  # [edge_pairs, channels, 2*l+1], source frame
            transported_states.append(torch.einsum(
                "pab,pcb->pca", wigner_transport, selected_state
            ))  # [edge_pairs, channels, 2*l+1], center frame
            mixed_geometry.append(self.geometry_channel_mixing[angular_momentum](
                neighbor_geometry[angular_momentum].transpose(1, 2)
            ).transpose(1, 2))

        one_particle_by_order: dict[int, torch.Tensor] = {}
        for _, state_order, state_block in unpack_all_orders(transported_states):
            for _, geometry_order, geometry_block in unpack_all_orders(mixed_geometry):
                for output_order, product_values in multiply_so2_blocks(
                    state_block, state_order, geometry_block, geometry_order
                ):
                    if output_order <= self.maximum_angular_momentum:
                        previous = one_particle_by_order.get(output_order)
                        one_particle_by_order[output_order] = product_values if previous is None else previous + product_values

        number_of_neighbors = scatter_sum(
            torch.ones_like(center_edge_indices, dtype=states[0].dtype),
            center_edge_indices,
            number_of_edges,
        ).clamp_min(1.0).view(number_of_edges, 1, 1)
        one_particle_by_order = {
            order: scatter_sum(values, center_edge_indices, number_of_edges) / number_of_neighbors
            for order, values in one_particle_by_order.items()
        }

        # Explicit cluster basis: B^(1)=M; B^(nu+1)=B^(nu) tensor_SO(2) M.
        product_bases = [one_particle_by_order]
        for _ in range(1, self.maximum_correlation_order):
            next_basis: dict[int, torch.Tensor] = {}
            for left_order, left_values in product_bases[-1].items():
                for right_order, right_values in one_particle_by_order.items():
                    for output_order, product_values in multiply_so2_blocks(
                        left_values, left_order, right_values, right_order
                    ):
                        if output_order <= self.maximum_angular_momentum:
                            previous = next_basis.get(output_order)
                            next_basis[output_order] = product_values if previous is None else previous + product_values
            product_bases.append(next_basis)

        old_blocks = {(l, m): values for l, m, values in unpack_all_orders(states)}
        updated_blocks: dict[tuple[int, int], torch.Tensor] = {}
        for angular_momentum in range(self.maximum_angular_momentum + 1):
            for absolute_order in range(angular_momentum + 1):
                template = old_blocks[(angular_momentum, absolute_order)]
                correlation_values = [
                    basis.get(absolute_order, torch.zeros_like(template)) for basis in product_bases
                ]
                increment = self.output_channel_mixing[f"{angular_momentum}_{absolute_order}"](
                    torch.cat(correlation_values, dim=-1)
                )
                updated_blocks[(angular_momentum, absolute_order)] = (
                    self.residual_weights[angular_momentum] * template
                    + self.message_weights[angular_momentum] * increment
                )
        return pack_all_orders(updated_blocks, self.maximum_angular_momentum)
