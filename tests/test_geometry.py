import torch

from bondnet.data.atomic_data import AtomicData
from bondnet.data.neighborhood import construct_directed_edges
from bondnet.modules.radial_basis import RadialBasis
from bondnet.modules.spherical_harmonics import real_spherical_harmonics


def test_directed_edge_construction() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    edges = construct_directed_edges(positions, cutoff_radius=1.5)
    assert edges.sender_atom_indices.tolist() == [0, 1]
    assert edges.receiver_atom_indices.tolist() == [1, 0]
    assert edges.displacement_vectors.shape == (2, 3)
    assert edges.unit_displacement_vectors.shape == (2, 3)
    assert edges.distances.shape == (2,)


def test_edges_stay_inside_each_structure() -> None:
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [1.2, 0.0, 0.0],
    ])
    batch_indices = torch.tensor([0, 0, 1, 1])
    edges = construct_directed_edges(positions, cutoff_radius=1.5, batch_indices=batch_indices)
    pairs = set(zip(edges.sender_atom_indices.tolist(), edges.receiver_atom_indices.tolist(), strict=True))
    assert pairs == {(0, 1), (1, 0), (2, 3), (3, 2)}


def test_edge_neighbors_share_the_same_center_atom() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    data = AtomicData(torch.tensor([1, 6, 1]), positions, cutoff_radius=1.5)
    neighbor = data.edge_neighbor
    quadruples = set(zip(
        neighbor.sender1_indices.tolist(), neighbor.receiver1_indices.tolist(),
        neighbor.sender2_indices.tolist(), neighbor.receiver2_indices.tolist(),
        strict=True,
    ))
    assert quadruples == {(1, 0, 1, 2), (1, 2, 1, 0)}
    assert neighbor.edge1_indices.shape == (2,)


def test_edge_cutoff_limits_neighbor_edges() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.2, 0.0, 0.0]])
    data = AtomicData(torch.tensor([1, 6, 1]), positions, cutoff_radius=2.5, edge_cutoff_radius=1.1)
    neighbor = data.edge_neighbor
    neighbor_lengths = data.edge.distances[neighbor.edge2_indices]
    assert torch.all(neighbor_lengths < 1.1)
    assert neighbor.edge1_indices.numel() == 2


def test_atomic_data_field_shapes() -> None:
    atomic_numbers = torch.tensor([1, 6, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    batch_indices = torch.tensor([0, 0, 0])
    data = AtomicData(atomic_numbers, positions, batch_indices, cutoff_radius=1.5)
    assert data["positions"].shape == (3, 3)
    assert data["atomic_numbers"].shape == (3,)
    assert data["batch_indices"].shape == (3,)
    assert data.edge.sender_atom_indices.shape == (2,)
    assert data.edge.receiver_atom_indices.shape == (2,)
    assert data.edge.displacement_vectors.shape == (2, 3)
    assert data.edge.unit_displacement_vectors.shape == (2, 3)
    assert data.edge.distances.shape == (2,)


def test_radial_cutoff_is_zero() -> None:
    basis = RadialBasis(4, cutoff_radius=2.0)
    assert torch.equal(basis(torch.tensor([2.0, 2.5])), torch.zeros(2, 4))


def test_spherical_harmonic_shapes() -> None:
    directions = torch.tensor([[0.0, 0.0, 1.0]])
    values = real_spherical_harmonics(directions, 2)
    assert [value.shape[-1] for value in values] == [1, 3, 5]
