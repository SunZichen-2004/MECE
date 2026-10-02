import torch

from bondnet import AtomicData, EdgeCenteredEquivariantModel


def test_toy_energy_overfit() -> None:
    torch.manual_seed(9)
    model = EdgeCenteredEquivariantModel(
        number_of_elements=10, number_of_channels=12, number_of_radial_functions=6,
        number_of_interactions=2, cutoff_radius=4.0,
    )
    atomic_numbers = torch.tensor([1, 6, 8, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-0.2, 1.1, 0.1], [0.1, -0.4, 1.2]])
    data = AtomicData(atomic_numbers, positions, cutoff_radius=model.cutoff_radius)
    target_energy = torch.tensor([1.75])
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
    initial_loss = None
    for _ in range(120):
        optimizer.zero_grad()
        loss = (model(data)["energy"] - target_energy).square().mean()
        if initial_loss is None:
            initial_loss = loss.detach()
        loss.backward()
        optimizer.step()
    assert loss.detach() < initial_loss * 1.0e-3
