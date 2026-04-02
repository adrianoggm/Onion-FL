from __future__ import annotations

"""Pure local training and evaluation helpers."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainRoundResult:
    """Summary of one local training round."""

    avg_loss: float
    num_samples: int
    batch_count: int


@dataclass(frozen=True)
class EvalResult:
    """Summary of a local evaluation pass."""

    loss: float
    accuracy: float
    num_samples: int


@dataclass(frozen=True)
class DetailedEvalResult:
    """Summary of an evaluation pass with optional confusion-matrix details."""

    loss: float
    accuracy: float
    num_samples: int
    confusion_matrix: np.ndarray | None = None
    labels: list[int] | None = None


def train_classifier_round(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    *,
    local_epochs: int = 1,
) -> TrainRoundResult:
    """Train a classifier for the requested number of local epochs."""
    model.train()
    total_loss = 0.0
    num_samples = 0
    batch_count = 0

    for _ in range(max(1, int(local_epochs))):
        for features, targets in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = int(features.size(0))
            total_loss += float(loss.item()) * batch_size
            num_samples += batch_size
            batch_count += 1

    return TrainRoundResult(
        avg_loss=total_loss / max(num_samples, 1),
        num_samples=num_samples,
        batch_count=batch_count,
    )


def evaluate_classifier(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
) -> EvalResult:
    """Evaluate a classifier on a labeled dataset."""
    prev_training = model.training
    model.eval()

    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for features, targets in data_loader:
            logits = model(features)
            loss = criterion(logits, targets)
            batch_size = int(features.size(0))
            total_loss += float(loss.item()) * batch_size
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == targets).sum().item())
            count += batch_size

    if prev_training:
        model.train()

    return EvalResult(
        loss=total_loss / max(count, 1),
        accuracy=correct / max(count, 1),
        num_samples=count,
    )


def evaluate_classifier_arrays(
    model: torch.nn.Module,
    data: tuple[np.ndarray, np.ndarray],
    *,
    batch_size: int = 256,
    include_confusion_matrix: bool = False,
) -> DetailedEvalResult:
    """Evaluate a classifier on array-backed labeled data."""
    features, labels = data
    prev_training = model.training
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).long(),
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    count = 0
    all_predictions = [] if include_confusion_matrix else None
    all_labels = [] if include_confusion_matrix else None

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            batch_size = int(batch_features.size(0))
            predictions = torch.argmax(logits, dim=1)

            total_loss += float(loss.item()) * batch_size
            correct += int((predictions == batch_labels).sum().item())
            count += batch_size

            if include_confusion_matrix:
                all_predictions.append(predictions.cpu())
                all_labels.append(batch_labels.cpu())

    if prev_training:
        model.train()

    confusion_matrix = None
    label_values = None
    if include_confusion_matrix:
        y_true = (
            torch.cat(all_labels).numpy()
            if all_labels
            else np.array([], dtype=np.int64)
        )
        y_pred = (
            torch.cat(all_predictions).numpy()
            if all_predictions
            else np.array([], dtype=np.int64)
        )
        label_values = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        if not label_values:
            label_values = [0, 1]

        label_index = {label: idx for idx, label in enumerate(label_values)}
        confusion_matrix = np.zeros((len(label_values), len(label_values)), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            confusion_matrix[label_index[true_label], label_index[pred_label]] += 1

    return DetailedEvalResult(
        loss=total_loss / max(count, 1),
        accuracy=correct / max(count, 1),
        num_samples=count,
        confusion_matrix=confusion_matrix,
        labels=label_values,
    )
