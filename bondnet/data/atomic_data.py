import torch

from bondnet.data.neighborhood import (
    DirectedEdgeData,
    EdgeNeighborData,
    construct_directed_edges,
    construct_edge_neighbors,
    construct_same_center_edge_neighbors,
)


class AtomicData:
    """Flattened atomic batch, laid out the way MACE consumes structures.

    The atom neighbor list, edge geometry and edge neighbor list are built here,
    before the model. Several molecules are concatenated into one atom list;
    ``batch_indices`` says which structure each atom belongs to. Vectors are in
    the global (laboratory) frame.

    data["atomic_numbers"]: [n_atoms]
    data["positions"]: [n_atoms, 3]
    data["batch_indices"]: [n_atoms]
    data.edge.sender_atom_indices: [n_edges]
    data.edge.receiver_atom_indices: [n_edges]
    data.edge.displacement_vectors: [n_edges, 3]
    data.edge.unit_displacement_vectors: [n_edges, 3]
    data.edge.distances: [n_edges]
    data.edge_neighbor.edge1_indices: [n_edge_of_edge]
    data.edge_neighbor.edge2_indices: [n_edge_of_edge]
    data.edge_neighbor.sender1_indices: [n_edge_of_edge]
    data.edge_neighbor.receiver1_indices: [n_edge_of_edge]
    data.edge_neighbor.sender2_indices: [n_edge_of_edge]
    data.edge_neighbor.receiver2_indices: [n_edge_of_edge]
    """

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        cutoff_radius: float = 5.0,
        edge_cutoff_radius: float | None = None,
    ):
        number_of_atoms = positions.shape[0]
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(f"positions must have shape [n_atoms, 3], got {tuple(positions.shape)}")
        if atomic_numbers.shape != (number_of_atoms,):
            raise ValueError(
                f"atomic_numbers must have shape [{number_of_atoms}], got {tuple(atomic_numbers.shape)}"
            )
        if batch_indices is None:
            batch_indices = torch.zeros(number_of_atoms, dtype=torch.long, device=positions.device)
        else:
            batch_indices = batch_indices.to(device=positions.device, dtype=torch.long)
        if batch_indices.shape != (number_of_atoms,):
            raise ValueError(
                f"batch_indices must have shape [{number_of_atoms}], got {tuple(batch_indices.shape)}"
            )

        self.atomic_numbers = atomic_numbers.to(dtype=torch.long)  # [n_atoms]
        self.positions = positions  # [n_atoms, 3]
        self.batch_indices = batch_indices  # [n_atoms]
        self.cutoff_radius = float(cutoff_radius)
        self.edge_cutoff_radius = float(cutoff_radius if edge_cutoff_radius is None else edge_cutoff_radius)
        self.edge: DirectedEdgeData = construct_directed_edges(positions, self.cutoff_radius, batch_indices)
        # The model equation couples (i -> j) to (i -> j'), hence line-graph
        # neighbors share the same sender/center atom.  The older endpoint-based
        # constructor remains public for backward-compatible geometry tests.
        self.edge_neighbor: EdgeNeighborData = construct_same_center_edge_neighbors(
            self.edge, self.edge_cutoff_radius
        )

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in ("atomic_numbers", "positions", "batch_indices"):
            raise KeyError(key)
        return getattr(self, key)
