import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

from bondnet import AtomicData, BondNet
from bondnet.data.three_bpa import load_three_bpa_parquet


def make_data(configuration: dict[str, torch.Tensor], cutoff_radius: float, device: torch.device) -> AtomicData:
    return AtomicData(
        configuration["atomic_numbers"].to(device),
        configuration["positions"].to(device),
        cutoff_radius=cutoff_radius,
    )


def evaluate(
    model: BondNet,
    configurations: list[dict[str, torch.Tensor]],
    energy_offset: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    absolute_energy_errors = []
    absolute_force_errors = []
    start_time = time.perf_counter()
    model.eval()
    for configuration in configurations:
        data = make_data(configuration, model.cutoff_radius, device)
        prediction = model(data, compute_forces=True)
        target_energy = configuration["energy"].to(device) - energy_offset
        absolute_energy_errors.append((prediction["energy"][0] - target_energy).abs().detach().cpu())
        absolute_force_errors.append(
            (prediction["forces"] - configuration["forces"].to(device)).abs().mean().detach().cpu()
        )
    elapsed = time.perf_counter() - start_time
    number_of_atoms = configurations[0]["atomic_numbers"].numel()
    return {
        "energy_mae_ev": torch.stack(absolute_energy_errors).mean().item(),
        "energy_mae_mev_per_atom": 1000.0 * torch.stack(absolute_energy_errors).mean().item() / number_of_atoms,
        "force_mae_mev_per_angstrom": 1000.0 * torch.stack(absolute_force_errors).mean().item(),
        "seconds_per_configuration": elapsed / len(configurations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal reproducible BondNet experiment on 3BPA")
    parser.add_argument("--train-file", type=Path, default=Path("data/3bpa/train_300K.parquet"))
    parser.add_argument("--test-file", type=Path, default=Path("data/3bpa/test_300K.parquet"))
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--valid-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/3bpa_smoke/results.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    all_training = load_three_bpa_parquet(args.train_file, args.train_size + args.valid_size)
    training = all_training[: args.train_size]
    validation = all_training[args.train_size :]
    test = load_three_bpa_parquet(args.test_file, args.valid_size)
    energy_offset = torch.stack([item["energy"] for item in training]).mean().to(device)
    model = BondNet(
        number_of_channels=args.channels,
        number_of_radial_functions=6,
        maximum_angular_momentum=1,
        number_of_interactions=1,
        maximum_correlation_order=2,
        cutoff_radius=args.cutoff,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        for index in torch.randperm(len(training), generator=generator).tolist():
            configuration = training[index]
            data = make_data(configuration, args.cutoff, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data, compute_forces=True)
            target_energy = configuration["energy"].to(device) - energy_offset
            energy_loss = (prediction["energy"][0] - target_energy).square() / data.positions.shape[0]
            force_loss = (prediction["forces"] - configuration["forces"].to(device)).square().mean()
            loss = energy_loss + args.force_weight * force_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        print(f"epoch {epoch + 1}/{args.epochs} loss={loss.item():.6f}")
    training_seconds = time.perf_counter() - start_time
    peak_resident_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    peak_resident_memory_mb = peak_resident_memory / (1024.0 * 1024.0) if sys.platform == "darwin" else peak_resident_memory / 1024.0
    results = {
        "kind": "minimal_smoke_test_not_full_benchmark",
        "seed": args.seed,
        "device": str(device),
        "train_size": len(training),
        "valid_size": len(validation),
        "test_size": len(test),
        "epochs": args.epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "training_seconds_per_epoch": training_seconds / args.epochs,
        "peak_process_rss_mb": peak_resident_memory_mb,
        "validation": evaluate(model, validation, energy_offset, device),
        "test_300K": evaluate(model, test, energy_offset, device),
        "command": " ".join(os.sys.argv),
    }
    if device.type == "mps":
        results["peak_mps_allocated_mb"] = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
