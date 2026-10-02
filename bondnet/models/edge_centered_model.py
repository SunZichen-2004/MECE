import torch
from torch import nn

from bondnet.data.atomic_data import AtomicData
from bondnet.modules.edge_blocks import EdgeInteractionBlock
from bondnet.modules.edge_frames import construct_edge_frames, rotate_vectors_to_edge_frames
from bondnet.modules.radial_basis import RadialBasis
from bondnet.modules.scatter import scatter_sum
from bondnet.modules.spherical_harmonics import real_spherical_harmonics, split_magnetic_blocks


class LegacyEdgeCenteredEquivariantModel(nn.Module):
    """Small line-graph model using only tensor, harmonic, and scatter primitives.

    The caller builds an ``AtomicData`` first. This module only reads that batch.
    """

    def __init__(
        self,
        number_of_elements: int = 10,
        number_of_channels: int = 32,
        number_of_radial_functions: int = 8,
        maximum_angular_momentum: int = 2,
        number_of_interactions: int = 3,
        cutoff_radius: float = 5.0,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.maximum_angular_momentum = maximum_angular_momentum
        self.element_embedding = nn.Embedding(number_of_elements, number_of_channels)
        self.radial_basis = RadialBasis(number_of_radial_functions, cutoff_radius)
        self.radial_to_channels = nn.ModuleList([
            nn.Linear(number_of_radial_functions, number_of_channels, bias=False)
            for _ in range(maximum_angular_momentum + 1)
        ])
        self.element_pair_to_channels = nn.Linear(2 * number_of_channels, number_of_channels)
        number_of_magnetic_blocks = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 2) // 2
        self.interactions = nn.ModuleList([
            EdgeInteractionBlock(number_of_channels, number_of_radial_functions, number_of_magnetic_blocks)
            for _ in range(number_of_interactions)
        ])
        self.edge_energy_readout = nn.Sequential(
            nn.Linear(number_of_channels * number_of_magnetic_blocks, number_of_channels),
            nn.SiLU(),
            nn.Linear(number_of_channels, 1),
        )

    def forward(
        self,
        data: AtomicData,
        compute_forces: bool = False,
    ) -> dict[str, torch.Tensor]:
        if data.cutoff_radius != self.cutoff_radius:
            raise ValueError(
                f"data cutoff {data.cutoff_radius} does not match the model cutoff {self.cutoff_radius}"
            )

        # Flattened batch, same layout as a MACE AtomicData.
        positions = data["positions"]  # [n_atoms, 3]
        atomic_numbers = data["atomic_numbers"]  # [n_atoms]
        batch_indices = data["batch_indices"]  # [n_atoms]
        sender_atom_indices = data.edge.sender_atom_indices  # [n_edges]
        receiver_atom_indices = data.edge.receiver_atom_indices  # [n_edges]
        edge1_indices = data.edge_neighbor.edge1_indices  # [n_edge_of_edge], center edge
        edge2_indices = data.edge_neighbor.edge2_indices  # [n_edge_of_edge], neighbor edge

        # Precomputed edge vectors are detached from positions, so forces need
        # them rebuilt from the stored indices with the same shapes.
        if compute_forces and not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)  # [n_atoms, 3]
        if compute_forces or data.edge.displacement_vectors is None:
            global_displacement_vectors = positions[receiver_atom_indices] - positions[sender_atom_indices]  # [n_edges, 3]
            distances = torch.linalg.vector_norm(global_displacement_vectors, dim=-1)  # [n_edges]
            global_unit_displacement_vectors = global_displacement_vectors / distances.unsqueeze(-1).clamp_min(1.0e-12)  # [n_edges, 3]
        else:
            global_displacement_vectors = data.edge.displacement_vectors  # [n_edges, 3], receiver - sender
            distances = data.edge.distances  # [n_edges]
            global_unit_displacement_vectors = data.edge.unit_displacement_vectors  # [n_edges, 3]

        number_of_atoms = positions.shape[0]
        number_of_edges = distances.shape[0]
        element_features = self.element_embedding(atomic_numbers)  # [n_atoms, n_channels]
        pair_values = self.element_pair_to_channels(torch.cat((
            element_features[sender_atom_indices],  # [n_edges, n_channels]
            element_features[receiver_atom_indices],  # [n_edges, n_channels]
        ), dim=-1))  # [n_edges, n_channels]
        radial_values = self.radial_basis(distances)  # [n_edges, n_radial]

        # Row k of each matrix is edge axis k written in the global frame, so
        # edge_vector = edge_rotation_matrices[e] @ global_vector.
        edge_rotation_matrices = construct_edge_frames(
            global_unit_displacement_vectors,
            sender_atom_indices,
            receiver_atom_indices,
            number_of_atoms,
        )  # [n_edges, 3, 3]

        # Every bond is the +z axis of its own edge frame.
        edge_bond_directions = global_unit_displacement_vectors.new_tensor([0.0, 0.0, 1.0]).expand(number_of_edges, 3)  # [n_edges, 3]
        edge_bond_harmonics = real_spherical_harmonics(edge_bond_directions, self.maximum_angular_momentum)
        # edge_bond_harmonics[l]: [n_edges, 2 * l + 1], edge frame

        # Direction of edge2 written in the frame of edge1, one row per edge-of-edge pair.
        directions_in_edge_frames = rotate_vectors_to_edge_frames(
            global_unit_displacement_vectors[edge2_indices],  # [n_edge_of_edge, 3], global frame
            edge_rotation_matrices[edge1_indices],  # [n_edge_of_edge, 3, 3]
        )  # [n_edge_of_edge, 3], edge frame
        edge_neighbor_harmonics = real_spherical_harmonics(directions_in_edge_frames, self.maximum_angular_momentum)
        # edge_neighbor_harmonics[l]: [n_edge_of_edge, 2 * l + 1], edge frame

        # Mean over the neighbors of each edge1.
        edge_neighbor_counts = scatter_sum(
            torch.ones_like(directions_in_edge_frames[:, 0]),  # [n_edge_of_edge]
            edge1_indices,
            number_of_edges,
        ).clamp_min(1.0).unsqueeze(-1)  # [n_edges, 1]
        edge_environment_harmonics = [
            scatter_sum(neighbor_values, edge1_indices, number_of_edges) / neighbor_counts
            for neighbor_values in edge_neighbor_harmonics
        ]
        # edge_environment_harmonics[l]: [n_edges, 2 * l + 1], edge frame

        magnetic_blocks: list[torch.Tensor] = []
        for angular_momentum, bond_values in enumerate(edge_bond_harmonics):
            angular_values = bond_values + edge_environment_harmonics[angular_momentum]  # [n_edges, 2 * l + 1], edge frame
            radial_channels = self.radial_to_channels[angular_momentum](radial_values) * pair_values  # [n_edges, n_channels]
            for magnetic_values in split_magnetic_blocks(angular_values, angular_momentum):
                # magnetic_values: [n_edges, 1] for m = 0, otherwise [n_edges, 2]
                magnetic_blocks.append(magnetic_values.unsqueeze(-1) * radial_channels.unsqueeze(1))
                # block: [n_edges, n_components, n_channels]
        # len(magnetic_blocks) = (L + 1) * (L + 2) / 2

        for interaction in self.interactions:
            magnetic_blocks = interaction(
                magnetic_blocks,
                radial_values,  # [n_edges, n_radial]
                sender_atom_indices,  # [n_edges]
                receiver_atom_indices,  # [n_edges]
                number_of_atoms,
            )
            # each block stays [n_edges, n_components, n_channels]

        invariant_edge_values = []
        for block_values in magnetic_blocks:
            if block_values.shape[1] == 1:
                invariant_edge_values.append(block_values.squeeze(1))  # [n_edges, n_channels]
            else:
                invariant_edge_values.append(torch.sqrt(block_values.square().sum(dim=1) + 1.0e-12))  # [n_edges, n_channels]
        directed_edge_energies = self.edge_energy_readout(
            torch.cat(invariant_edge_values, dim=-1)  # [n_edges, n_blocks * n_channels]
        ).squeeze(-1)  # [n_edges]
        edge_energies = 0.5 * directed_edge_energies  # [n_edges]
        number_of_structures = int(batch_indices.max().item()) + 1 if number_of_atoms else 0
        structure_energies = scatter_sum(
            edge_energies,  # [n_edges]
            batch_indices[sender_atom_indices],  # [n_edges]
            number_of_structures,
        )  # [n_structures]
        result = {
            "energy": structure_energies,  # [n_structures]
            "edge_energy": edge_energies,  # [n_edges]
            "edge_rotation_matrices": edge_rotation_matrices,  # [n_edges, 3, 3], global -> edge frame
        }
        if compute_forces:
            forces = -torch.autograd.grad(
                structure_energies.sum(), positions, create_graph=self.training, retain_graph=self.training
            )[0]  # [n_atoms, 3]
            result["forces"] = forces
        return result


# Backward-compatible module path. The old prototype remains above for readers,
# while all imports of EdgeCenteredEquivariantModel now use the complete BondNet.
from .bondnet_model import BondNet as EdgeCenteredEquivariantModel  # noqa: E402
