from __future__ import annotations

"""Pure local training and evaluation helpers."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


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
