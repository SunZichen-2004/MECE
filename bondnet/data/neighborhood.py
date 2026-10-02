from dataclasses import dataclass

import torch


@dataclass
class DirectedEdgeData:
    """Directed edges of one flattened batch. Geometry is fixed before the model runs.

    sender_atom_indices: [n_edges]
    receiver_atom_indices: [n_edges]
    displacement_vectors: [n_edges, 3], positions[receiver] - positions[sender]
    distances: [n_edges]
    unit_displacement_vectors: [n_edges, 3]
    """

    sender_atom_indices: torch.Tensor
    receiver_atom_indices: torch.Tensor
    displacement_vectors: torch.Tensor
    distances: torch.Tensor
    unit_displacement_vectors: torch.Tensor


def construct_directed_edges(
    positions: torch.Tensor,
    cutoff_radius: float,
    batch_indices: torch.Tensor | None = None,
) -> DirectedEdgeData:
    """Build every directed pair inside the cutoff. Atoms in different structures are not connected.

    positions: [n_atoms, 3]
    batch_indices: [n_atoms], structure id of each atom. Missing ids are treated as one structure.
    """
    number_of_atoms = positions.shape[0]
    if batch_indices is None:
        batch_indices = torch.zeros(number_of_atoms, dtype=torch.long, device=positions.device)
    else:
        batch_indices = batch_indices.to(device=positions.device, dtype=torch.long)

    # displacement_matrix[sender, receiver] = positions[receiver] - positions[sender]
    displacement_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)  # [n_atoms, n_atoms, 3]
    distance_matrix = torch.linalg.vector_norm(displacement_matrix, dim=-1)  # [n_atoms, n_atoms]
    same_structure = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)  # [n_atoms, n_atoms]
    not_self = ~torch.eye(number_of_atoms, dtype=torch.bool, device=positions.device)  # [n_atoms, n_atoms]
    edge_mask = same_structure & not_self & (distance_matrix < cutoff_radius)  # [n_atoms, n_atoms]

    sender_atom_indices, receiver_atom_indices = torch.where(edge_mask)  # [n_edges], [n_edges]
    displacement_vectors = displacement_matrix[sender_atom_indices, receiver_atom_indices]  # [n_edges, 3]
    distances = distance_matrix[sender_atom_indices, receiver_atom_indices]  # [n_edges]
    unit_displacement_vectors = displacement_vectors / distances.unsqueeze(-1).clamp_min(1.0e-12)  # [n_edges, 3]
    return DirectedEdgeData(
        sender_atom_indices.detach(),
        receiver_atom_indices.detach(),
        displacement_vectors.detach(),
        distances.detach(),
        unit_displacement_vectors.detach(),
    )


@dataclass
class EdgeNeighborData:
    """Directed edge-to-edge pairs: the neighbor list of the line graph.

    Pair p connects edge1 = (sender1 -> receiver1) with its neighbor edge2 = (sender2 -> receiver2).
    Edges are the "nodes" of the line graph, so edge1 plays the role of the center node.

    edge1_indices: [n_edge_of_edge], index into the n_edges edge list
    edge2_indices: [n_edge_of_edge], index into the n_edges edge list
    sender1_indices: [n_edge_of_edge], atom index
    receiver1_indices: [n_edge_of_edge], atom index
    sender2_indices: [n_edge_of_edge], atom index
    receiver2_indices: [n_edge_of_edge], atom index
    """

    edge1_indices: torch.Tensor
    edge2_indices: torch.Tensor
    sender1_indices: torch.Tensor
    receiver1_indices: torch.Tensor
    sender2_indices: torch.Tensor
    receiver2_indices: torch.Tensor


def construct_edge_neighbors(
    edge: DirectedEdgeData,
    number_of_atoms: int,
    edge_cutoff_radius: float,
) -> EdgeNeighborData:
    """Pair every directed edge with the directed edges that point into it.

    For edge1 = (i -> j), edge2 runs over k -> i and k -> j with k not in {i, j}
    and length below edge_cutoff_radius. Both orientations of edge1 are centers,
    e.g. bonds {1, 2}, {2, 3} give (1->2, 3->2), (2->1, 3->2), (2->3, 1->2), (3->2, 1->2).
    """
    sender_atom_indices = edge.sender_atom_indices  # [n_edges]
    receiver_atom_indices = edge.receiver_atom_indices  # [n_edges]
    number_of_edges = sender_atom_indices.shape[0]
    edge_indices = torch.arange(number_of_edges, device=sender_atom_indices.device)  # [n_edges]

    # Edge1 touches two atoms: one (atom, edge1) incidence entry per endpoint.
    incidence_atoms = torch.cat((sender_atom_indices, receiver_atom_indices))  # [2 * n_edges]
    incidence_edges = torch.cat((edge_indices, edge_indices))  # [2 * n_edges]

    # Candidates for edge2 are keyed by their receiver, grouped so every atom owns a contiguous slice.
    is_short = edge.distances < edge_cutoff_radius  # [n_edges]
    short_receivers = receiver_atom_indices[is_short]  # [n_short_edges]
    atom_order = torch.argsort(short_receivers, stable=True)  # [n_short_edges]
    short_edges_by_atom = edge_indices[is_short][atom_order]  # [n_short_edges]
    short_counts = torch.bincount(short_receivers, minlength=number_of_atoms)  # [n_atoms]
    short_starts = torch.cumsum(short_counts, dim=0) - short_counts  # [n_atoms]

    # For every (atom, edge1) incidence, take every edge2 whose receiver is that atom.
    pairs_per_incidence = short_counts[incidence_atoms]  # [2 * n_edges]
    edge1_candidates = torch.repeat_interleave(incidence_edges, pairs_per_incidence)  # [n_candidate_pairs]
    shared_atoms = torch.repeat_interleave(incidence_atoms, pairs_per_incidence)  # [n_candidate_pairs]
    first_pair_of_incidence = torch.cumsum(pairs_per_incidence, dim=0) - pairs_per_incidence  # [2 * n_edges]
    offsets = torch.arange(edge1_candidates.shape[0], device=edge_indices.device) - torch.repeat_interleave(
        first_pair_of_incidence, pairs_per_incidence
    )  # [n_candidate_pairs], position inside the atom's slice
    edge2_candidates = short_edges_by_atom[short_starts[shared_atoms] + offsets]  # [n_candidate_pairs]

    # Drop k in {i, j}: that is edge1 itself or its reverse.
    sender2_atoms = sender_atom_indices[edge2_candidates]  # [n_candidate_pairs]
    same_bond = (sender2_atoms == sender_atom_indices[edge1_candidates]) | (
        sender2_atoms == receiver_atom_indices[edge1_candidates]
    )  # [n_candidate_pairs]
    edge1_indices = edge1_candidates[~same_bond]  # [n_edge_of_edge]
    edge2_indices = edge2_candidates[~same_bond]  # [n_edge_of_edge]
    pair_order = torch.argsort(edge1_indices * number_of_edges + edge2_indices)  # [n_edge_of_edge]
    edge1_indices = edge1_indices[pair_order]  # [n_edge_of_edge]
    edge2_indices = edge2_indices[pair_order]  # [n_edge_of_edge]
    return EdgeNeighborData(
        edge1_indices,
        edge2_indices,
        sender_atom_indices[edge1_indices],
        receiver_atom_indices[edge1_indices],
        sender_atom_indices[edge2_indices],
        receiver_atom_indices[edge2_indices],
    )


def construct_same_center_edge_neighbors(edge: DirectedEdgeData, edge_cutoff_radius: float) -> EdgeNeighborData:
    """Build pairs ``(i -> j, i -> j')`` used by the BondNet equation.

    The first edge is the center edge and the second edge contributes a message.
    Both share atom ``i`` and ``j' != j``.  Arrays have length
    ``n_edge_pairs = sum_i degree(i) * (degree(i) - 1)``.
    """
    sender_atom_indices = edge.sender_atom_indices
    receiver_atom_indices = edge.receiver_atom_indices
    number_of_edges = sender_atom_indices.shape[0]
    all_edge_indices = torch.arange(number_of_edges, device=sender_atom_indices.device)
    same_sender = sender_atom_indices[:, None] == sender_atom_indices[None, :]
    different_edge = all_edge_indices[:, None] != all_edge_indices[None, :]
    neighbor_inside_cutoff = edge.distances[None, :] < edge_cutoff_radius
    center_edge_indices, neighbor_edge_indices = torch.where(
        same_sender & different_edge & neighbor_inside_cutoff
    )
    return EdgeNeighborData(
        center_edge_indices,
        neighbor_edge_indices,
        sender_atom_indices[center_edge_indices],
        receiver_atom_indices[center_edge_indices],
        sender_atom_indices[neighbor_edge_indices],
        receiver_atom_indices[neighbor_edge_indices],
    )
