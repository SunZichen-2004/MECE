import torch

from bondnet import AtomicData, EdgeCenteredEquivariantModel


def main() -> None:
    torch.manual_seed(7)
    model = EdgeCenteredEquivariantModel()
    atomic_numbers = torch.tensor([1, 6, 8, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [-0.2, 1.2, 0.2], [0.2, -0.4, 1.3]])
    data = AtomicData(atomic_numbers, positions, cutoff_radius=model.cutoff_radius)
    target_energy = torch.tensor([-2.5])
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    for training_step in range(201):
        optimizer.zero_grad()
        predicted_energy = model(data)["energy"]
        loss = (predicted_energy - target_energy).square().mean()
        loss.backward()
        optimizer.step()
        if training_step % 20 == 0:
            print(f"training step {training_step:4d} | loss {loss.item():.8f}")
    prediction = model(data, compute_forces=True)
    print("energy", prediction["energy"].detach().tolist())
    print("forces", prediction["forces"].detach().tolist())


if __name__ == "__main__":
    main()
