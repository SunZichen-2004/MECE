from pathlib import Path

import torch


def load_three_bpa_parquet(path: str | Path, maximum_configurations: int | None = None) -> list[dict[str, torch.Tensor]]:
    """Load ColabFit 3BPA parquet rows as small torch dictionaries.

    Returned positions are in angstrom, energies in eV, and forces in eV/angstrom.
    ``pyarrow`` is an optional experiment dependency and is imported lazily.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError("3BPA parquet loading requires `pip install bondnet[experiment]`") from error
    columns = ["atomic_numbers", "positions", "energy", "atomic_forces"]
    table = parquet.read_table(path, columns=columns)
    if maximum_configurations is not None:
        table = table.slice(0, maximum_configurations)
    configurations = []
    for row in table.to_pylist():
        configurations.append({
            "atomic_numbers": torch.tensor(row["atomic_numbers"], dtype=torch.long),
            "positions": torch.tensor(row["positions"], dtype=torch.get_default_dtype()),
            "energy": torch.tensor(row["energy"], dtype=torch.get_default_dtype()),
            "forces": torch.tensor(row["atomic_forces"], dtype=torch.get_default_dtype()),
        })
    return configurations
