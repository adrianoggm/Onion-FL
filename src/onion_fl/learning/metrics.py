from __future__ import annotations

"""Evaluation metrics and how reports are combined (spec §10.3).

A metric plugin turns logits and labels into named scalars, so they fit in a
``Payload``: ``recall.<class>`` and ``confusion.<true>.<pred>`` included.
Reports from several nodes are combined weighted by their samples, except the
confusion counts, which are summed.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from onion_fl.core.registry import Registry

metrics = Registry("metric")
SUMMED = ("confusion.",)


def _confusion(logits: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes))
    np.add.at(matrix, (y, logits.argmax(axis=1)), 1)
    return matrix


@metrics.register("loss", title="Pérdida", description="Entropía cruzada media.")
class Loss:
    def compute(self, logits, y, n_classes):
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return {"loss": float(-log_probs[np.arange(len(y)), y].mean())}


@metrics.register("accuracy", title="Exactitud", description="Fracción de aciertos.")
class Accuracy:
    def compute(self, logits, y, n_classes):
        return {"accuracy": float((logits.argmax(axis=1) == y).mean())}


@metrics.register(
    "macro_f1",
    title="F1 macro",
    description="Media del F1 de cada clase; una clase sin aciertos ni predicciones cuenta 0.",
)
class MacroF1:
    def compute(self, logits, y, n_classes):
        matrix = _confusion(logits, y, n_classes)
        tp = np.diag(matrix)
        precision = np.divide(
            tp,
            matrix.sum(axis=0),
            out=np.zeros(n_classes),
            where=matrix.sum(axis=0) > 0,
        )
        recall = np.divide(
            tp,
            matrix.sum(axis=1),
            out=np.zeros(n_classes),
            where=matrix.sum(axis=1) > 0,
        )
        both = precision + recall
        f1 = np.divide(
            2 * precision * recall, both, out=np.zeros(n_classes), where=both > 0
        )
        return {"macro_f1": float(f1.mean())}


@metrics.register(
    "recall_per_class",
    title="Sensibilidad por clase",
    description="recall.<clase> para cada clase presente en las etiquetas.",
)
class RecallPerClass:
    def compute(self, logits, y, n_classes):
        matrix = _confusion(logits, y, n_classes)
        support = matrix.sum(axis=1)
        return {
            f"recall.{c}": float(matrix[c, c] / support[c])
            for c in range(n_classes)
            if support[c]
        }


@metrics.register(
    "confusion_matrix",
    title="Matriz de confusión",
    description="confusion.<real>.<predicha>; al agregar se suman.",
)
class ConfusionMatrix:
    def compute(self, logits, y, n_classes):
        matrix = _confusion(logits, y, n_classes)
        return {
            f"confusion.{i}.{j}": float(matrix[i, j])
            for i in range(n_classes)
            for j in range(n_classes)
        }


def compute(
    logits: np.ndarray, y: np.ndarray, n_classes: int, names: Sequence[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        out |= metrics.create(name).compute(
            np.asarray(logits), np.asarray(y), n_classes
        )
    return out


def evaluate(
    model: nn.Module, data: Any, names: Sequence[str] = ("loss", "accuracy")
) -> tuple[dict[str, float], int]:
    """Metrics of ``model`` on ``data`` (``X``, ``y``, ``n_classes``) and the sample count."""
    y = np.asarray(data.y, np.int64)
    if not len(y):
        return {}, 0
    return compute(predict(model, data.X), y, data.n_classes, names), len(y)


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Logits of ``model`` on ``X``, in evaluation mode."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X, np.float32))).numpy()
    model.train(was_training)
    return logits.astype(np.float64)


def reduce_reports(
    reports: Sequence[tuple[Mapping[str, float], float]],
) -> tuple[dict[str, float], float]:
    """Combine ``(metrics, samples)`` reports: sample-weighted means, confusion counts summed."""
    reports = [(m, n) for m, n in reports if n > 0]
    total = sum(n for _, n in reports)
    if not total:
        return {}, 0
    out: dict[str, float] = {}
    for key in sorted({k for m, _ in reports for k in m}):
        holders = [(m[key], n) for m, n in reports if key in m]
        if key.startswith(SUMMED):
            out[key] = float(sum(v for v, _ in holders))
        else:
            out[key] = float(
                sum(v * n for v, n in holders) / sum(n for _, n in holders)
            )
    return out, total
