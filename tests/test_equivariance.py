import torch

from bondnet import AtomicData, EdgeCenteredEquivariantModel


def _configuration() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([1, 6, 8, 1], dtype=torch.long)
    positions = torch.tensor([
        [0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [-0.3, 1.2, 0.4], [0.2, -0.5, 1.3]
    ], dtype=torch.float64)
    return atomic_numbers, positions


def _data(atomic_numbers: torch.Tensor, positions: torch.Tensor, cutoff_radius: float) -> AtomicData:
    return AtomicData(atomic_numbers, positions, cutoff_radius=cutoff_radius)


def _orthogonal_matrix(reflection: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    if reflection:
        matrix[:, 0] *= -1
    return matrix


def test_energy_and_force_rotation_equivariance() -> None:
    torch.manual_seed(4)
    model = EdgeCenteredEquivariantModel(number_of_channels=8, number_of_radial_functions=4).double().eval()
    atomic_numbers, positions = _configuration()
    rotation = _orthogonal_matrix()
    original = model(_data(atomic_numbers, positions, model.cutoff_radius), compute_forces=True)
    transformed = model(_data(atomic_numbers, positions @ rotation.T, model.cutoff_radius), compute_forces=True)
    torch.testing.assert_close(original["energy"], transformed["energy"], rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(original["forces"] @ rotation.T, transformed["forces"], rtol=1.0e-5, atol=1.0e-6)


def test_energy_and_force_reflection_equivariance() -> None:
    torch.manual_seed(5)
    model = EdgeCenteredEquivariantModel(number_of_channels=8, number_of_radial_functions=4).double().eval()
    atomic_numbers, positions = _configuration()
    reflection = _orthogonal_matrix(reflection=True)
    original = model(_data(atomic_numbers, positions, model.cutoff_radius), compute_forces=True)
    transformed = model(_data(atomic_numbers, positions @ reflection.T, model.cutoff_radius), compute_forces=True)
    torch.testing.assert_close(original["energy"], transformed["energy"], rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(original["forces"] @ reflection.T, transformed["forces"], rtol=1.0e-5, atol=1.0e-6)


def test_non_scalar_edge_channels_affect_energy() -> None:
    torch.manual_seed(11)
    model = EdgeCenteredEquivariantModel(number_of_channels=8, number_of_radial_functions=4).double().eval()
    atomic_numbers, positions = _configuration()
    energy = model(_data(atomic_numbers, positions, model.cutoff_radius))["energy"]
    energy.backward()
    non_scalar_gradient = model.interactions[0].source_channel_mixing[1].weight.grad
    assert non_scalar_gradient is not None
    assert torch.linalg.vector_norm(non_scalar_gradient) > 0
