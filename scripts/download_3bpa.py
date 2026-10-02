"""Download the public ColabFit/Hugging Face 3BPA parquet mirror."""

from pathlib import Path
from urllib.request import urlretrieve


FILES = {
    "train_300K.parquet": "https://huggingface.co/datasets/colabfit/3BPA_train_300K/resolve/main/co/co_0.parquet?download=true",
    "test_300K.parquet": "https://huggingface.co/datasets/colabfit/3BPA_test_300K/resolve/main/co/co_0.parquet?download=true",
}


def main() -> None:
    output_directory = Path(__file__).resolve().parents[1] / "data" / "3bpa"
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        destination = output_directory / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"already present: {destination}")
            continue
        print(f"downloading {url} -> {destination}")
        urlretrieve(url, destination)


if __name__ == "__main__":
    main()
