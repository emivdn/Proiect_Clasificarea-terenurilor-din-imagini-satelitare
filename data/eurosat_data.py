from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransformedSubset(Dataset):
    """Subset wrapper that applies a split-specific transform at access time."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass
class EuroSATDataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: List[str]
    split_indices: Dict[str, List[int]]


def build_transforms(
    image_size: int = 64,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns train and eval transforms for EuroSAT RGB images."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, eval_transform


def stratified_split_indices(
    targets: Sequence[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Creates deterministic stratified train/val/test index splits."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    indices_by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, cls in enumerate(targets):
        indices_by_class[int(cls)].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for cls, cls_indices in indices_by_class.items():
        cls_copy = cls_indices[:]
        rng.shuffle(cls_copy)

        n = len(cls_copy)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_part = cls_copy[:n_train]
        val_part = cls_copy[n_train : n_train + n_val]
        test_part = cls_copy[n_train + n_val : n_train + n_val + n_test]

        if len(train_part) + len(val_part) + len(test_part) != n:
            raise RuntimeError(f"Split size mismatch for class {cls}")

        train_indices.extend(train_part)
        val_indices.extend(val_part)
        test_indices.extend(test_part)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def save_split_indices(split_indices: Dict[str, List[int]], split_file: Path) -> None:
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)


def load_split_indices(split_file: Path) -> Dict[str, List[int]]:
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "train": [int(x) for x in data["train"]],
        "val": [int(x) for x in data["val"]],
        "test": [int(x) for x in data["test"]],
    }


def validate_split_indices(split_indices: Dict[str, List[int]], dataset_size: int) -> None:
    all_indices = split_indices["train"] + split_indices["val"] + split_indices["test"]
    if len(all_indices) != dataset_size:
        raise ValueError("Split size does not match dataset size")

    unique_indices = set(all_indices)
    if len(unique_indices) != dataset_size:
        raise ValueError("Splits are not disjoint or do not cover all samples")

    if min(unique_indices) < 0 or max(unique_indices) >= dataset_size:
        raise ValueError("Found split index out of dataset bounds")


def class_distribution(targets: Sequence[int], indices: Sequence[int]) -> Dict[int, int]:
    return dict(Counter(int(targets[i]) for i in indices))


def create_eurosat_dataloaders(
    dataset_root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int = 64,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_file: Optional[str] = None,
    pin_memory: bool = True,
) -> EuroSATDataLoaders:
    """
    Creates EuroSAT DataLoaders with deterministic stratified split.

    Expected dataset structure:
    dataset_root/
      AnnualCrop/
      Forest/
      ... (10 class folders)
    """
    root_path = Path(dataset_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root_path}")

    base_dataset = datasets.ImageFolder(root=str(root_path), transform=None)
    if len(base_dataset.classes) != 10:
        raise ValueError(
            f"Expected 10 classes for EuroSAT, found {len(base_dataset.classes)} in {root_path}"
        )

    train_transform, eval_transform = build_transforms(image_size=image_size)

    if split_file:
        split_path = Path(split_file)
        if split_path.exists():
            split_indices = load_split_indices(split_path)
        else:
            split_indices = stratified_split_indices(
                base_dataset.targets,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            save_split_indices(split_indices, split_path)
    else:
        split_indices = stratified_split_indices(
            base_dataset.targets,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    validate_split_indices(split_indices, dataset_size=len(base_dataset))

    train_dataset = TransformedSubset(base_dataset, split_indices["train"], transform=train_transform)
    val_dataset = TransformedSubset(base_dataset, split_indices["val"], transform=eval_transform)
    test_dataset = TransformedSubset(base_dataset, split_indices["test"], transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return EuroSATDataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=base_dataset.classes,
        split_indices=split_indices,
    )