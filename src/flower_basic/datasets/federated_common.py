from __future__ import annotations

"""Shared helpers for federated split loading and manifest aggregation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SplitLoader = Callable[[str | Path], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class ClientDataLoaders:
    """Prepared loaders and split counts for a local FL client."""

    train_loader: DataLoader
    val_loader: DataLoader | None
    num_train_samples: int
    num_val_samples: int = 0
    num_test_samples: int = 0
    input_dim: int = 0


def resolve_split_paths(
    node_dir: Path, subject_id: str | None = None
) -> tuple[Path, Path, Path]:
    """Resolve train/val/test split paths for an aggregated or per-subject client."""
    if subject_id:
        subject_dir = node_dir / f"subject_{subject_id}"
        return (
            subject_dir / "train.npz",
            subject_dir / "val.npz",
            subject_dir / "test.npz",
        )

    return node_dir / "train.npz", node_dir / "val.npz", node_dir / "test.npz"


def load_optional_split(
    split_path: Path,
    load_split: SplitLoader,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Best-effort split loader used for optional validation/test artifacts."""
    if not split_path.exists():
        return None, None

    try:
        features, labels, _ = load_split(split_path)
    except Exception:
        return None, None

    if features.size == 0:
        return None, None
    return features, labels


def build_client_data(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    *,
    load_split: SplitLoader,
    batch_size: int,
    generator: torch.Generator | None = None,
    train_shuffle: bool = True,
    eval_batch_size: int = 256,
) -> ClientDataLoaders:
    """Build deterministic client loaders and split counts from NPZ artifacts."""
    train_features, train_labels, _ = load_split(train_file)
    if train_features.size == 0:
        raise RuntimeError("Train split is empty for this node. Check subject assignments.")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features).float(),
            torch.from_numpy(train_labels).long(),
        ),
        batch_size=batch_size,
        shuffle=train_shuffle,
        generator=generator,
    )

    val_loader = None
    val_features, val_labels = load_optional_split(val_file, load_split)
    if val_features is not None and val_labels is not None:
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(val_features).float(),
                torch.from_numpy(val_labels).long(),
            ),
            batch_size=eval_batch_size,
            shuffle=False,
        )

    test_features, _ = load_optional_split(test_file, load_split)

    return ClientDataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        num_train_samples=len(train_features),
        num_val_samples=0 if val_features is None else len(val_features),
        num_test_samples=0 if test_features is None else len(test_features),
        input_dim=int(train_features.shape[1]),
    )


def load_manifest_eval_data(
    manifest_path: Path,
    *,
    load_split: SplitLoader,
    tag: str,
    split_name: str = "test",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and concatenate one split from all manifest nodes."""
    print(f"{tag} Loading evaluation data from manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    all_features = []
    all_labels = []

    for node_id in nodes.keys():
        split_path = base / node_id / f"{split_name}.npz"
        if not split_path.exists():
            print(f"{tag}   Node {node_id}: {split_name}.npz NOT FOUND at {split_path}")
            continue

        features, labels, _ = load_split(split_path)
        if features.size == 0:
            print(f"{tag}   Node {node_id}: {split_name}.npz is empty")
            continue

        all_features.append(features)
        all_labels.append(labels)
        print(f"{tag}   Node {node_id}: loaded {features.shape[0]} {split_name} samples")

    if not all_features:
        print(f"{tag} WARNING: No {split_name} data found for centralized evaluation!")
        return None

    total_features = np.concatenate(all_features, axis=0)
    total_labels = np.concatenate(all_labels, axis=0)
    print(
        f"{tag} Total evaluation data: {total_features.shape[0]} samples, "
        f"{total_features.shape[1]} features"
    )
    return total_features, total_labels


def load_manifest_split_counts(
    manifest_path: Path,
    *,
    load_split: SplitLoader,
) -> tuple[int, int, int]:
    """Aggregate train/val/test sample counts from all nodes in a manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    totals = {"train": 0, "val": 0, "test": 0}

    for node_id in nodes.keys():
        for split_name in totals:
            split_path = base / node_id / f"{split_name}.npz"
            if not split_path.exists():
                continue
            features, _, _ = load_split(split_path)
            totals[split_name] += int(features.shape[0])

    return totals["train"], totals["val"], totals["test"]
