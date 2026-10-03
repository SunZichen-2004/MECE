import torch
from torch import nn

from bondnet.data.atomic_data import AtomicData
from bondnet.modules.edge_cluster_expansion import EdgeClusterExpansionBlock, split_packed_state
from bondnet.modules.edge_frames import (
    construct_edge_frames,
    relative_frame_rotations,
    rotate_vectors_to_edge_frames,
)
from bondnet.modules.radial_basis import ElementPairRadial, RadialBasis
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
        atomic_energies: dict[int, float] | None = None,
    ):
        super().__init__()
        if maximum_angular_momentum > 2:
            raise ValueError("the transparent real-Wigner implementation currently supports l_max <= 2")
        self.cutoff_radius = cutoff_radius
        self.maximum_angular_momentum = maximum_angular_momentum
        self.number_of_interactions = number_of_interactions
        self.element_embedding = nn.Embedding(number_of_elements, number_of_channels)
        self.radial_basis = RadialBasis(number_of_radial_functions, cutoff_radius)
        self.element_pair_radial = ElementPairRadial(
            number_of_radial_functions, number_of_channels, maximum_angular_momentum
        )
        atomic_energy_table = torch.zeros(number_of_elements)
        if atomic_energies is not None:
            for atomic_number, energy in atomic_energies.items():
                atomic_energy_table[int(atomic_number)] = float(energy)
        self.register_buffer("atomic_energy_table", atomic_energy_table)
        self.interactions = nn.ModuleList([
            EdgeClusterExpansionBlock(
                number_of_channels,
                maximum_angular_momentum,
                maximum_correlation_order,
            )
            for _ in range(number_of_interactions)
        ])
        number_of_magnetic_blocks = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 2) // 2
        invariant_size = number_of_channels * number_of_magnetic_blocks
        scalar_size = number_of_channels * (maximum_angular_momentum + 1)
        # Bond readouts. Every layer except the last is a linear map of the
        # invariant m=0 components. The last layer is an MLP of all invariants.
        self.layer_readouts = nn.ModuleList([
            nn.Linear(scalar_size, 1)
            for _ in range(number_of_interactions)
        ])
        self.layer_readouts.append(nn.Sequential(
            nn.Linear(invariant_size, number_of_channels),
            nn.SiLU(),
            nn.Linear(number_of_channels, 1),
        ))

    def _geometry_in_center_frames(
        self,
        edge_rotation_matrices: torch.Tensor,
        global_unit_displacement_vectors: torch.Tensor,
        radial_weights: torch.Tensor,
        center_edge_indices: torch.Tensor,
        neighbor_edge_indices: torch.Tensor,
    ) -> list[torch.Tensor]:
        directions = rotate_vectors_to_edge_frames(
            global_unit_displacement_vectors[neighbor_edge_indices],
            edge_rotation_matrices[center_edge_indices],
        )  # [n_edge_pairs, 3], expressed in the center-edge frame
        harmonic_values = real_spherical_harmonics(directions, self.maximum_angular_momentum)
        selected_weights = radial_weights[neighbor_edge_indices]  # [n_edge_pairs, lmax+1, channels]
        return [
            selected_weights[:, angular_momentum].unsqueeze(-1) * angular_values.unsqueeze(1)
            for angular_momentum, angular_values in enumerate(harmonic_values)
        ]

    @staticmethod
    def _invariant_features(states: list[torch.Tensor]) -> torch.Tensor:
        """Rotation-invariant features of one layer's edge state h_ij.

        Returns [n_edges, channels * number_of_magnetic_blocks]. Scalar blocks are
        kept as they are; each two-component block contributes its norm.
        """
        invariant_values = []
        for angular_momentum, angular_values in enumerate(states):
            for block_values in split_packed_state(angular_values, angular_momentum):
                if block_values.shape[1] == 1:
                    invariant_values.append(block_values.squeeze(1))
                else:
                    invariant_values.append(torch.sqrt(block_values.square().sum(dim=1) + 1.0e-12))
        return torch.cat(invariant_values, dim=-1)

    @staticmethod
    def _scalar_features(states: list[torch.Tensor]) -> torch.Tensor:
        """m = 0 component of every angular momentum. Shape [n_edges, channels * (lmax+1)]."""
        return torch.cat([state[:, :, 0] for state in states], dim=-1)

    def _bond_layer_energy(self, states: list[torch.Tensor], layer_index: int) -> torch.Tensor:
        """E_ij(T): one scalar bond energy for each directed edge at this layer."""
        if layer_index < self.number_of_interactions:
            features = self._scalar_features(states)
        else:
            features = self._invariant_features(states)
        return self.layer_readouts[layer_index](features).squeeze(-1)

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
        radial_values = self.radial_basis(distances)  # [n_edges, radial_functions]
        radial_weights = self.element_pair_radial(
            radial_values,
            element_features[sender_indices],
            element_features[receiver_indices],
        )  # [n_edges, lmax+1, channels], weights of the pair (Z_i, Z_j)
        edge_rotation_matrices = construct_edge_frames(
            unit_displacement_vectors, sender_indices, receiver_indices, number_of_atoms
        )  # [n_edges, 3, 3], global Cartesian components -> edge components

        neighbor_geometry = self._geometry_in_center_frames(
            edge_rotation_matrices,
            unit_displacement_vectors,
            radial_weights,
            center_edge_indices,
            neighbor_edge_indices,
        )  # neighbor_geometry[l]: [n_edge_pairs, channels, 2*l+1], center-edge frame. R_nl(r_ij')*Y_lm(r_ij'^(ij)).
        # Include j itself in A_i = sum_j R(r_ij)Y(r_ij), while messages use j' != j.
        own_edge_indices = torch.arange(number_of_edges, device=positions.device)
        own_geometry = self._geometry_in_center_frames(
            edge_rotation_matrices,
            unit_displacement_vectors,
            radial_weights,
            own_edge_indices,
            own_edge_indices,
        ) # own_geometry[l]: [n_edges, channels, 2*l+1], center-edge frame. R_nl(r_ij)*Y_lm(r_ij^(ij)).
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
        # states[l] = (sum_j' R(r_ij)Y(r_ij^(ij)) + sum_j' R(r_ij')Y(r_ij'^(ij))) / (number_of_neighbors + 1)
        relative_rotations = relative_frame_rotations(
            edge_rotation_matrices[center_edge_indices],
            edge_rotation_matrices[neighbor_edge_indices],
        )  # [n_edge_pairs, 3, 3], neighbor-edge frame -> center-edge frame
        # E = sum_{ij, T} E_ij(T). T runs over the initial edge state and every interaction.
        bond_energies = self._bond_layer_energy(states, 0)
        for layer_index, interaction in enumerate(self.interactions, start=1):
            states = interaction(
                states,
                neighbor_geometry,
                relative_rotations,
                center_edge_indices,
                neighbor_edge_indices,
                number_of_edges,
            )
            bond_energies = bond_energies + self._bond_layer_energy(states, layer_index)
        number_of_structures = int(batch_indices.max().item()) + 1 if number_of_atoms else 0
        atomic_reference = scatter_sum(
            self.atomic_energy_table[atomic_numbers],
            batch_indices,
            number_of_structures,
        )
        structure_energies = scatter_sum(
            bond_energies,
            batch_indices[sender_indices],
            number_of_structures,
        ) + atomic_reference
        result = {
            "energy": structure_energies,
            "bond_energy": bond_energies,
            "edge_rotation_matrices": edge_rotation_matrices,
        }
        if compute_forces:
            result["forces"] = -torch.autograd.grad(
                structure_energies.sum(), positions, create_graph=self.training, retain_graph=self.training
            )[0]
        return result


# Preserve the public name used by the original prototype and its tests.
EdgeCenteredEquivariantModel = BondNet
