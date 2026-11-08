from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.eurosat_data import class_distribution, create_eurosat_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica existenta si functionalitatea Dataset-ului/DataLoader-ului pentru EuroSAT"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Cale catre folderul EuroSAT RGB (folder cu cele 10 clase)",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="artifacts/splits/eurosat_split_70_15_15_seed42.json",
        help="Fisier JSON pentru salvarea/incarcarea split-urilor fixe",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assert_batch_ok(images: torch.Tensor, labels: torch.Tensor, split_name: str, image_size: int) -> None:
    if images.ndim != 4:
        raise AssertionError(f"[{split_name}] Batch images should have 4 dims, got {images.ndim}")
    if labels.ndim != 1:
        raise AssertionError(f"[{split_name}] Batch labels should have 1 dim, got {labels.ndim}")
    if images.shape[1:] != (3, image_size, image_size):
        raise AssertionError(
            f"[{split_name}] Expected image shape (B, 3, {image_size}, {image_size}), got {tuple(images.shape)}"
        )
    if images.dtype != torch.float32:
        raise AssertionError(f"[{split_name}] Images should be float32, got {images.dtype}")
    if labels.dtype not in (torch.int64, torch.long):
        raise AssertionError(f"[{split_name}] Labels should be int64/long, got {labels.dtype}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)

    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Folderul de date nu exista sau nu este director: {data_path}")

    class_dirs = [p for p in data_path.iterdir() if p.is_dir()]
    if len(class_dirs) != 10:
        raise ValueError(
            "Structura dataset-ului pare invalida. "
            f"Sunt necesare 10 foldere de clase, dar au fost gasite {len(class_dirs)} in {data_path}."
        )

    dataloaders = create_eurosat_dataloaders(
        dataset_root=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        split_file=args.split_file,
    )

    print("=== Verificare Exista Dataset + DataLoader Functional ===")
    print(f"Data dir: {data_path}")
    print(f"Clase detectate ({len(dataloaders.class_names)}): {dataloaders.class_names}")

    split_indices = dataloaders.split_indices
    n_train = len(split_indices["train"])
    n_val = len(split_indices["val"])
    n_test = len(split_indices["test"])
    n_total = n_train + n_val + n_test

    print(f"Split-uri: train={n_train}, val={n_val}, test={n_test}, total={n_total}")
    print(
        "Proportii realizate: "
        f"train={100.0 * n_train / n_total:.2f}%, "
        f"val={100.0 * n_val / n_total:.2f}%, "
        f"test={100.0 * n_test / n_total:.2f}%"
    )
    print(f"Fisier split: {args.split_file}")

    target_map = dataloaders.train_loader.dataset.base_dataset.targets
    for split_name in ["train", "val", "test"]:
        dist = class_distribution(target_map, split_indices[split_name])
        print(f"Distributie clase [{split_name}]: {dist}")

    split_loader_pairs = [
        ("train", dataloaders.train_loader),
        ("val", dataloaders.val_loader),
        ("test", dataloaders.test_loader),
    ]
    for split_name, loader in split_loader_pairs:
        images, labels = next(iter(loader))
        assert_batch_ok(images, labels, split_name=split_name, image_size=args.image_size)
        print(
            f"Batch [{split_name}] OK: images={tuple(images.shape)}, "
            f"labels={tuple(labels.shape)}, dtype={images.dtype}/{labels.dtype}"
        )

    print("Toate verificarile au trecut cu succes.")


if __name__ == "__main__":
    main()