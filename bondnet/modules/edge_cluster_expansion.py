from functools import lru_cache

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


def _so2_contract_dense(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Contract every pair of absolute orders with batched multiplies.

    left, right: [orders, items, 2, channels], component 0 cosine and 1 sine.
    returns the same shape.
    """
    number_of_orders = left.shape[0]
    left_cosine, left_sine = left[:, :, 0], left[:, :, 1]
    right_cosine, right_sine = right[:, :, 0], right[:, :, 1]
    output_cosine = []
    output_sine = []
    for output_order in range(number_of_orders):
        sum_index = torch.arange(output_order + 1, device=left.device)
        partner = output_order - sum_index
        cosine = (left_cosine[sum_index] * right_cosine[partner] - left_sine[sum_index] * right_sine[partner]).sum(0)
        sine = (left_cosine[sum_index] * right_sine[partner] + left_sine[sum_index] * right_cosine[partner]).sum(0)
        if output_order == 0:
            positive = torch.arange(1, number_of_orders, device=left.device)
            if positive.numel():
                cosine = cosine + (left_cosine[positive] * right_cosine[positive] + left_sine[positive] * right_sine[positive]).sum(0)
        else:
            lower = torch.arange(1, number_of_orders - output_order, device=left.device)
            if lower.numel():
                higher = lower + output_order
                cosine = cosine + (
                    left_cosine[higher] * right_cosine[lower] + left_sine[higher] * right_sine[lower]
                    + left_cosine[lower] * right_cosine[higher] + left_sine[lower] * right_sine[higher]
                ).sum(0)
                sine = sine + (
                    left_sine[higher] * right_cosine[lower] - left_cosine[higher] * right_sine[lower]
                    + left_cosine[lower] * right_sine[higher] - left_sine[lower] * right_cosine[higher]
                ).sum(0)
        output_cosine.append(cosine)
        output_sine.append(sine)
    return torch.stack((torch.stack(output_cosine), torch.stack(output_sine)), dim=2)


def _slice_order(program, tensor, order: int, component: int, channels: int):
    return program.slice(tensor, (order * 2 + component) * channels, channels)


def _add_terms(program, terms):
    total = terms[0]
    for term in terms[1:]:
        total = program.binary("add", total, term)
    return total


@lru_cache(maxsize=8)
def _fused_so2_program(number_of_orders: int, channels: int) -> str:
    """One CUDA edge program for the dense SO(2) contraction."""
    from eqx.conv.program import Program

    program = Program()
    left = program.input(0, "edge", number_of_orders * 2 * channels)
    right = program.input(1, "edge", number_of_orders * 2 * channels)
    pieces = []
    for output_order in range(number_of_orders):
        cosine_terms = []
        sine_terms = []
        for first in range(output_order + 1):
            second = output_order - first
            left_cosine = _slice_order(program, left, first, 0, channels)
            left_sine = _slice_order(program, left, first, 1, channels)
            right_cosine = _slice_order(program, right, second, 0, channels)
            right_sine = _slice_order(program, right, second, 1, channels)
            cosine_terms.append(program.binary(
                "add",
                program.binary("mul", left_cosine, right_cosine),
                program.scale(program.binary("mul", left_sine, right_sine), -1),
            ))
            sine_terms.append(program.binary(
                "add",
                program.binary("mul", left_cosine, right_sine),
                program.binary("mul", left_sine, right_cosine),
            ))
        if output_order == 0:
            for order in range(1, number_of_orders):
                left_cosine = _slice_order(program, left, order, 0, channels)
                left_sine = _slice_order(program, left, order, 1, channels)
                right_cosine = _slice_order(program, right, order, 0, channels)
                right_sine = _slice_order(program, right, order, 1, channels)
                cosine_terms.append(program.binary(
                    "add",
                    program.binary("mul", left_cosine, right_cosine),
                    program.binary("mul", left_sine, right_sine),
                ))
        else:
            for lower in range(1, number_of_orders - output_order):
                higher = lower + output_order
                for first, second in ((higher, lower), (lower, higher)):
                    left_cosine = _slice_order(program, left, first, 0, channels)
                    left_sine = _slice_order(program, left, first, 1, channels)
                    right_cosine = _slice_order(program, right, second, 0, channels)
                    right_sine = _slice_order(program, right, second, 1, channels)
                    cosine_terms.append(program.binary(
                        "add",
                        program.binary("mul", left_cosine, right_cosine),
                        program.binary("mul", left_sine, right_sine),
                    ))
                higher_cosine = _slice_order(program, left, higher, 0, channels)
                higher_sine = _slice_order(program, left, higher, 1, channels)
                lower_cosine = _slice_order(program, left, lower, 0, channels)
                lower_sine = _slice_order(program, left, lower, 1, channels)
                right_higher_cosine = _slice_order(program, right, higher, 0, channels)
                right_higher_sine = _slice_order(program, right, higher, 1, channels)
                right_lower_cosine = _slice_order(program, right, lower, 0, channels)
                right_lower_sine = _slice_order(program, right, lower, 1, channels)
                sine_terms.append(program.binary(
                    "add",
                    program.binary(
                        "add",
                        program.binary("mul", higher_sine, right_lower_cosine),
                        program.scale(program.binary("mul", higher_cosine, right_lower_sine), -1),
                    ),
                    program.binary(
                        "add",
                        program.binary("mul", lower_cosine, right_higher_sine),
                        program.scale(program.binary("mul", lower_sine, right_higher_cosine), -1),
                    ),
                ))
        pieces.append(_add_terms(program, cosine_terms))
        pieces.append(_add_terms(program, sine_terms))
    output = program.concatenate(pieces)
    return repr((tuple(program.nodes), ((output, 0, "edge"),)))


def _so2_contract_fused(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    from eqx.conv.edge import evaluate

    number_of_orders, number_of_items, _, channels = left.shape
    flat_left = left.permute(1, 0, 2, 3).reshape(number_of_items, -1)
    flat_right = right.permute(1, 0, 2, 3).reshape(number_of_items, -1)
    index = torch.arange(number_of_items, device=left.device)
    output = evaluate(
        _fused_so2_program(number_of_orders, channels),
        [flat_left, flat_right],
        index,
        index,
        number_of_items,
    )[0]
    return output.reshape(number_of_items, number_of_orders, 2, channels).permute(1, 0, 2, 3)


def so2_contract(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Contract every pair of absolute orders.

    left, right: [orders, items, 2, channels], component 0 cosine and 1 sine.
    CUDA uses one EquivariantX edge kernel. CPU keeps the batched multiply.
    """
    if left.is_cuda and left.dtype in (torch.float32, torch.float64) and left.shape[1] > 0:
        return _so2_contract_fused(left, right)
    return _so2_contract_dense(left, right)


def stack_so2_blocks(states: list[torch.Tensor]) -> torch.Tensor:
    """Pack each (l, m) block as [items, 2, channels], with a zero sine for m = 0.

    Block order is l, then m. Returns [blocks, items, 2, channels].
    """
    blocks = []
    for values in states:
        components = values.transpose(1, 2)
        scalar = torch.cat((components[:, 0:1], torch.zeros_like(components[:, 0:1])), dim=1)
        blocks.append(scalar.unsqueeze(0))
        if components.shape[1] > 1:
            positive_orders = components[:, 1:].reshape(components.shape[0], -1, 2, components.shape[-1])
            blocks.append(positive_orders.permute(1, 0, 2, 3))
    return torch.cat(blocks, dim=0)


def sum_same_orders(blocks: torch.Tensor, block_orders: torch.Tensor, number_of_orders: int) -> torch.Tensor:
    """Sum blocks that share an absolute order. The SO(2) product is bilinear, so this is exact."""
    summed = blocks.new_zeros((number_of_orders,) + blocks.shape[1:])
    return summed.index_add_(0, block_orders, blocks)


def unpack_updated_blocks(updated: torch.Tensor, maximum_angular_momentum: int) -> list[torch.Tensor]:
    """Invert stack_so2_blocks. m = 0 keeps only its cosine component."""
    outputs = []
    cursor = 0
    number_of_items = updated.shape[1]
    for angular_momentum in range(maximum_angular_momentum + 1):
        scalar = updated[cursor, :, 0:1, :]
        cursor += 1
        if angular_momentum == 0:
            pieces = scalar
        else:
            positive = updated[cursor : cursor + angular_momentum].permute(1, 0, 2, 3)
            cursor += angular_momentum
            positive = positive.reshape(number_of_items, angular_momentum * 2, updated.shape[-1])
            pieces = torch.cat((scalar, positive), dim=1)
        outputs.append(pieces.transpose(1, 2))
    return outputs


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
        # Equivariant channel mix in the center-edge frame, applied to h before
        # the product with Y_lm(r_ij'). Weights are shared by the 2l+1 components.
        self.pre_product_mixing = nn.ModuleList([
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
        block_orders = []
        block_angular_momenta = []
        for angular_momentum in range(maximum_angular_momentum + 1):
            for absolute_order in range(angular_momentum + 1):
                block_orders.append(absolute_order)
                block_angular_momenta.append(angular_momentum)
        self.register_buffer("block_orders", torch.tensor(block_orders, dtype=torch.long))
        self.register_buffer("block_angular_momenta", torch.tensor(block_angular_momenta, dtype=torch.long))

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
            transported_state = torch.einsum(
                "pab,pcb->pca", wigner_transport, selected_state
            )  # [edge_pairs, channels, 2*l+1], center frame
            transported_states.append(self.pre_product_mixing[angular_momentum](
                transported_state.transpose(1, 2)
            ).transpose(1, 2))
            mixed_geometry.append(self.geometry_channel_mixing[angular_momentum](
                neighbor_geometry[angular_momentum].transpose(1, 2)
            ).transpose(1, 2))

        number_of_orders = self.maximum_angular_momentum + 1
        state_orders = sum_same_orders(
            stack_so2_blocks(transported_states), self.block_orders, number_of_orders
        )
        geometry_orders = sum_same_orders(
            stack_so2_blocks(mixed_geometry), self.block_orders, number_of_orders
        )
        pair_products = so2_contract(state_orders, geometry_orders)
        number_of_neighbors = scatter_sum(
            torch.ones_like(center_edge_indices, dtype=states[0].dtype),
            center_edge_indices,
            number_of_edges,
        ).clamp_min(1.0)
        flat_products = pair_products.reshape(number_of_orders, pair_products.shape[1], -1)
        scattered = flat_products.new_zeros(number_of_orders, number_of_edges, flat_products.shape[-1])
        scattered = scattered.index_add_(1, center_edge_indices, flat_products)
        one_particle = scattered.view(
            number_of_orders, number_of_edges, 2, pair_products.shape[-1]
        ) / number_of_neighbors.view(1, number_of_edges, 1, 1)

        # Explicit cluster basis: B^(1)=M; B^(nu+1)=B^(nu) tensor_SO(2) M.
        product_bases = [one_particle]
        current_basis = one_particle
        for _ in range(1, self.maximum_correlation_order):
            current_basis = so2_contract(current_basis, one_particle)
            product_bases.append(current_basis)
        basis_features = torch.stack(product_bases, dim=0).permute(1, 2, 3, 0, 4).reshape(
            number_of_orders, number_of_edges, 2, -1
        )
        block_features = basis_features[self.block_orders]
        output_weights = torch.stack([
            self.output_channel_mixing[f"{angular_momentum}_{absolute_order}"].weight
            for angular_momentum in range(number_of_orders)
            for absolute_order in range(angular_momentum + 1)
        ])
        increment = torch.einsum("boc,bnkc->bnko", output_weights, block_features)
        templates = stack_so2_blocks(states)
        angular_momenta = self.block_angular_momenta
        updated = (
            self.residual_weights[angular_momenta].view(-1, 1, 1, 1) * templates
            + self.message_weights[angular_momenta].view(-1, 1, 1, 1) * increment
        )
        return unpack_updated_blocks(updated, self.maximum_angular_momentum)
