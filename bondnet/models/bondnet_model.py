import torch
from torch import nn

from bondnet.data.atomic_data import AtomicData
from bondnet.modules.edge_cluster_expansion import EdgeClusterExpansionBlock, split_packed_state
from bondnet.modules.edge_frames import (
    construct_edge_frames,
    relative_frame_rotations,
    rotate_vectors_to_edge_frames,
)
from bondnet.modules.radial_basis import RadialBasis
from bondnet.modules.scatter import scatter_sum
from bondnet.modules.spherical_harmonics import real_spherical_harmonics


class BondNet(nn.Module):
    """Edge-centered SO(2) cluster-expansion interatomic potential."""

    def __init__(
        self,
        number_of_elements: int = 119,
        number_of_channels: int = 32,
        number_of_radial_functions: int = 8,
        maximum_angular_momentum: int = 2,
        number_of_interactions: int = 3,
        maximum_correlation_order: int = 3,
        cutoff_radius: float = 5.0,
    ):
        super().__init__()
        if maximum_angular_momentum > 2:
            raise ValueError("the transparent real-Wigner implementation currently supports l_max <= 2")
        self.cutoff_radius = cutoff_radius
        self.maximum_angular_momentum = maximum_angular_momentum
        self.element_embedding = nn.Embedding(number_of_elements, number_of_channels)
        self.radial_basis = RadialBasis(number_of_radial_functions, cutoff_radius)
        self.radial_to_channels = nn.ModuleList([
            nn.Linear(number_of_radial_functions, number_of_channels, bias=False)
            for _ in range(maximum_angular_momentum + 1)
        ])
        self.element_pair_to_channels = nn.Linear(2 * number_of_channels, number_of_channels)
        self.interactions = nn.ModuleList([
            EdgeClusterExpansionBlock(
                number_of_channels,
                maximum_angular_momentum,
                maximum_correlation_order,
            )
            for _ in range(number_of_interactions)
        ])
        number_of_magnetic_blocks = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 2) // 2
        self.edge_energy_readout = nn.Sequential(
            nn.Linear(number_of_channels * number_of_magnetic_blocks, number_of_channels),
            nn.SiLU(),
            nn.Linear(number_of_channels, 1),
        )

    def _geometry_in_center_frames(
        self,
        edge_rotation_matrices: torch.Tensor,
        global_unit_displacement_vectors: torch.Tensor,
        radial_values: torch.Tensor,
        pair_channels: torch.Tensor,
        center_edge_indices: torch.Tensor,
        neighbor_edge_indices: torch.Tensor,
    ) -> list[torch.Tensor]:
        directions = rotate_vectors_to_edge_frames(
            global_unit_displacement_vectors[neighbor_edge_indices],
            edge_rotation_matrices[center_edge_indices],
        )  # [n_edge_pairs, 3], expressed in the center-edge frame
        harmonic_values = real_spherical_harmonics(directions, self.maximum_angular_momentum)
        result = []
        for angular_momentum, angular_values in enumerate(harmonic_values):
            radial_channels = self.radial_to_channels[angular_momentum](
                radial_values[neighbor_edge_indices]
            ) * pair_channels[neighbor_edge_indices]  # [n_edge_pairs, channels]
            result.append(radial_channels.unsqueeze(-1) * angular_values.unsqueeze(1))
            # [n_edge_pairs, channels, 2*l+1], center-edge frame
        return result

    @staticmethod
    def _reverse_edge_indices(sender_indices: torch.Tensor, receiver_indices: torch.Tensor) -> torch.Tensor:
        reverse_mask = (
            (sender_indices[:, None] == receiver_indices[None, :])
            & (receiver_indices[:, None] == sender_indices[None, :])
        )
        if not torch.all(reverse_mask.sum(dim=1) == 1):
            raise ValueError("every directed edge must have exactly one reverse edge")
        return reverse_mask.to(torch.int64).argmax(dim=1)

    def forward(self, data: AtomicData, compute_forces: bool = False) -> dict[str, torch.Tensor]:
        if data.cutoff_radius != self.cutoff_radius:
            raise ValueError(f"data cutoff {data.cutoff_radius} does not match model cutoff {self.cutoff_radius}")
        positions = data["positions"]  # [n_atoms, 3], global frame
        atomic_numbers = data["atomic_numbers"]  # [n_atoms]
        batch_indices = data["batch_indices"]  # [n_atoms]
        sender_indices = data.edge.sender_atom_indices  # [n_edges], atom i
        receiver_indices = data.edge.receiver_atom_indices  # [n_edges], atom j
        center_edge_indices = data.edge_neighbor.edge1_indices  # [n_edge_pairs], edge i->j
        neighbor_edge_indices = data.edge_neighbor.edge2_indices  # [n_edge_pairs], edge i->j'

        if compute_forces and not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
        displacement_vectors = positions[receiver_indices] - positions[sender_indices]  # [n_edges, 3]
        distances = torch.linalg.vector_norm(displacement_vectors, dim=-1)  # [n_edges]
        unit_displacement_vectors = displacement_vectors / distances.unsqueeze(-1).clamp_min(1.0e-12)
        number_of_atoms = positions.shape[0]
        number_of_edges = sender_indices.shape[0]
        element_features = self.element_embedding(atomic_numbers)  # [n_atoms, channels]
        pair_channels = self.element_pair_to_channels(torch.cat((
            element_features[sender_indices], element_features[receiver_indices]
        ), dim=-1))  # [n_edges, channels]
        radial_values = self.radial_basis(distances)  # [n_edges, radial_functions]
        edge_rotation_matrices = construct_edge_frames(
            unit_displacement_vectors, sender_indices, receiver_indices, number_of_atoms
        )  # [n_edges, 3, 3], global Cartesian components -> edge components

        neighbor_geometry = self._geometry_in_center_frames(
            edge_rotation_matrices,
            unit_displacement_vectors,
            radial_values,
            pair_channels,
            center_edge_indices,
            neighbor_edge_indices,
        )
        # Include j itself in A_i = sum_j R(r_ij)Y(r_ij), while messages use j' != j.
        own_edge_indices = torch.arange(number_of_edges, device=positions.device)
        own_geometry = self._geometry_in_center_frames(
            edge_rotation_matrices,
            unit_displacement_vectors,
            radial_values,
            pair_channels,
            own_edge_indices,
            own_edge_indices,
        )
        neighbor_counts = scatter_sum(
            torch.ones_like(center_edge_indices, dtype=positions.dtype),
            center_edge_indices,
            number_of_edges,
        ).view(number_of_edges, 1, 1)
        states = [
            (scatter_sum(values, center_edge_indices, number_of_edges) + own_values)
            / (neighbor_counts + 1.0)
            for values, own_values in zip(neighbor_geometry, own_geometry, strict=True)
        ]  # states[l]: [n_edges, channels, 2*l+1], in frame i->j

        relative_rotations = relative_frame_rotations(
            edge_rotation_matrices[center_edge_indices],
            edge_rotation_matrices[neighbor_edge_indices],
        )  # [n_edge_pairs, 3, 3], neighbor-edge frame -> center-edge frame
        for interaction in self.interactions:
            states = interaction(
                states,
                neighbor_geometry,
                relative_rotations,
                center_edge_indices,
                neighbor_edge_indices,
                number_of_edges,
            )

        invariant_values = []
        for angular_momentum, angular_values in enumerate(states):
            for block_values in split_packed_state(angular_values, angular_momentum):
                if block_values.shape[1] == 1:
                    invariant_values.append(block_values.squeeze(1))
                else:
                    invariant_values.append(torch.sqrt(block_values.square().sum(dim=1) + 1.0e-12))
        directed_edge_energies = self.edge_energy_readout(torch.cat(invariant_values, dim=-1)).squeeze(-1)
        reverse_indices = self._reverse_edge_indices(sender_indices, receiver_indices)
        symmetric_directed_energies = 0.5 * (
            directed_edge_energies + directed_edge_energies[reverse_indices]
        )
        edge_energies = 0.5 * symmetric_directed_energies  # directed sum counts each bond twice
        number_of_structures = int(batch_indices.max().item()) + 1 if number_of_atoms else 0
        structure_energies = scatter_sum(
            edge_energies,
            batch_indices[sender_indices],
            number_of_structures,
        )
        result = {
            "energy": structure_energies,
            "edge_energy": edge_energies,
            "directed_edge_energy": directed_edge_energies,
            "edge_rotation_matrices": edge_rotation_matrices,
        }
        if compute_forces:
            result["forces"] = -torch.autograd.grad(
                structure_energies.sum(), positions, create_graph=self.training, retain_graph=self.training
            )[0]
        return result


# Preserve the public name used by the original prototype and its tests.
EdgeCenteredEquivariantModel = BondNet
